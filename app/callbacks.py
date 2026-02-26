"""
callbacks.py — All Dash callback definitions for the ElectroAnatomyCloud app.

Every interactive behaviour of the UI is defined here as a @app.callback.
This module is imported by app.py solely for its side-effect of registering
callbacks; it does not define any layout.

Session state
-------------
All persistent state between callbacks is stored in `dcc.Store(id='pipeline-store')`.
The store holds a JSON-serializable dict (see app.py for the full schema).

Heavy Python objects (Mesh, PointCloud, sitk.Image, numpy arrays) are NOT stored
in the browser-facing dcc.Store. Instead:
  - The CT volume numpy array is cached in a server-side diskcache.Cache
    (key = session_id derived from a UUID in the store).
  - Mesh and PointCloud objects are reconstructed from the stored vertex/triangle
    lists on each callback invocation (cheap — just a numpy array wrap).

Background processing (region grow)
-------------------------------------
Region growing takes ~8–14 min. To avoid blocking the Dash server, it runs in a
concurrent.futures.ThreadPoolExecutor. A dcc.Interval polls for completion every
second and updates the UI when done.

Adding a new pipeline step
---------------------------
1. Increment TOTAL_STEPS in pipeline.py.
2. Add the step label to STEP_LABELS in pipeline.py.
3. Add a new branch in _run_step() for the new step number.
4. If the step needs new parameters, add them to param_controls.py and include
   the corresponding Input in advance_pipeline_step().
"""

import os
import json
import uuid
import threading
import traceback

import numpy as np
import diskcache

from dash import Input, Output, State, callback, no_update, ctx
import dash_bootstrap_components as dbc
from dash import html

import pipeline
from components.viewer_3d import build_alignment_figure, build_empty_figure
from components.ct_slicer import build_slice_figure
# app object is not imported here — Dash 4.x @callback decorator registers
# callbacks globally without needing an explicit app reference.

# ── Server-side cache (CT volumes + background job results) ────────────────────
_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '.cache')
_cache = diskcache.Cache(_CACHE_DIR)

# Background job registry: session_id -> {'done': bool, 'result': ..., 'error': str|None}
_bg_jobs: dict[str, dict] = {}
_bg_lock = threading.Lock()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _collect_params(
    seg_label, exclude_labels, intensity_tol, max_dist_mult, icp_iters, icp_algo
) -> dict:
    """Bundle sidebar control values into a params dict for pipeline functions."""
    return {
        'seg_label': seg_label or 2,
        'exclude_labels': exclude_labels or [1, 3, 4, 5, 6],
        'intensity_tolerance': intensity_tol or 200,
        'max_distance_multiplier': max_dist_mult or 1.5,
        'icp_iterations': icp_iters or 100,
        'icp_algorithm': icp_algo or 'P2P',
    }


def _stage_enabled(stages_enabled: list | None, stage: str) -> bool:
    """Return True if the given stage key is in the enabled stages list."""
    if stages_enabled is None:
        return True   # default: all enabled
    return stage in stages_enabled


def _store_from_data(store: dict) -> dict:
    """Return a fresh copy of the store dict (avoids mutating callback inputs)."""
    return dict(store)


def _get_or_create_session_id(store: dict) -> str:
    sid = store.get('session_id')
    if not sid:
        sid = str(uuid.uuid4())
    return sid


def _rebuild_mesh(store: dict):
    """Reconstruct a Mesh object from stored vertex/triangle arrays."""
    verts = np.array(store['mesh_vertices'])
    tris = np.array(store['mesh_triangles'])
    m = pipeline.Mesh(vertices=verts, triangles=tris)
    if store.get('voltages'):
        m.voltages = {int(row[0]): (row[1], row[2]) for row in store['voltages']}
    return m


def _rebuild_shell(store: dict):
    """Reconstruct a PointCloud from stored CT vertex array."""
    return pipeline.PointCloud(np.array(store['ct_vertices']))


def _mesh_to_store_verts(mesh) -> list:
    return mesh.get_vertices().tolist()


def _voltages_to_list(voltages: dict | None) -> list | None:
    """Serialise voltages dict {vid: (uni, bi)} -> list of [vid, uni, bi]."""
    if not voltages:
        return None
    return [[vid, v[0], v[1]] for vid, v in voltages.items()]


def _bipolar_list(store: dict) -> list | None:
    """Extract ordered bipolar voltages aligned to current mesh_vertices order."""
    if not store.get('voltages') or not store.get('mesh_vertices'):
        return None
    # voltages are stored by vertex id; mesh vertex order may differ
    # For simplicity, return voltages in vertex-id order if available
    try:
        rows = store['voltages']
        return [row[2] for row in rows]   # bipolar is index 2
    except Exception:
        return None


def _metrics_card(metrics: dict) -> dbc.Card:
    """Build the alignment quality report card shown at step 6."""
    rows = [
        dbc.Row([
            dbc.Col(html.Span(k.replace('_', ' ').title(), className='text-muted small'), width=8),
            dbc.Col(html.Span(f"{v:.3f}" if isinstance(v, float) else str(v),
                              className='fw-bold small text-end'), width=4),
        ], className='mb-1')
        for k, v in metrics.items()
    ]
    return dbc.Card([
        dbc.CardHeader("Alignment Quality Report", className='fw-bold'),
        dbc.CardBody(rows),
    ], color='dark', outline=True, className='mt-2')


# ── Callback 1: Validate file paths and enable Run button ──────────────────────

@callback(
    Output('status-mesh', 'children'),
    Output('status-mesh', 'className'),
    Output('status-seg', 'children'),
    Output('status-seg', 'className'),
    Output('status-ct', 'children'),
    Output('status-ct', 'className'),
    Output('btn-next-step', 'disabled'),
    Input('input-mesh-path', 'value'),
    Input('input-seg-path', 'value'),
    Input('input-ct-path', 'value'),
)
def validate_and_enable_button(mesh_path, seg_path, ct_path):
    """
    Check that all three file paths exist. Show green/red badges.
    Enable the Run Next Step button only when all three paths are valid.
    """
    def _badge(path):
        if not path:
            return '?', 'badge bg-secondary ms-1'
        if os.path.exists(path):
            return 'OK', 'badge bg-success ms-1'
        return 'X', 'badge bg-danger ms-1'

    m_text, m_cls = _badge(mesh_path)
    s_text, s_cls = _badge(seg_path)
    c_text, c_cls = _badge(ct_path)

    all_valid = (
        mesh_path and os.path.exists(mesh_path) and
        seg_path and os.path.exists(seg_path) and
        ct_path and os.path.exists(ct_path)
    )
    btn_disabled = not all_valid
    return m_text, m_cls, s_text, s_cls, c_text, c_cls, btn_disabled


# ── Callback 2: Advance pipeline step ─────────────────────────────────────────

@callback(
    Output('pipeline-store', 'data'),
    Output('step-status', 'children'),
    Output('grow-poll-interval', 'disabled'),
    Input('btn-next-step', 'n_clicks'),
    Input('btn-reset', 'n_clicks'),
    Input('grow-poll-interval', 'n_intervals'),
    State('pipeline-store', 'data'),
    State('input-mesh-path', 'value'),
    State('input-seg-path', 'value'),
    State('input-ct-path', 'value'),
    State('param-seg-label', 'value'),
    State('param-exclude-labels', 'value'),
    State('param-intensity-tolerance', 'value'),
    State('param-max-dist-mult', 'value'),
    State('param-icp-iterations', 'value'),
    State('param-icp-algorithm', 'value'),
    State('param-stages-enabled', 'value'),
    State('param-grown-cache-path', 'value'),
    prevent_initial_call=True,
)
def advance_pipeline_step(
    n_clicks_next, n_clicks_reset, n_intervals,
    store, mesh_path, seg_path, ct_path,
    seg_label, exclude_labels, intensity_tol, max_dist_mult, icp_iters, icp_algo,
    stages_enabled, grown_cache_path,
):
    """
    Main pipeline state machine. Each invocation either advances one step or resets.

    Steps:
      0 -> 1 : Load files + region grow (or skip/load cache if 'grow' is unchecked)
      1 -> 2 : (polling) Region grow complete
      2 -> 3 : COM alignment (or skip if 'com' is unchecked)
      3 -> 4 : PCA axis alignment (or skip if 'pca' is unchecked)
      4 -> 5 : Axial rotation sweep (or skip if 'axial' is unchecked)
      5 -> 6 : ICP refinement + metrics (or skip if 'icp' is unchecked)

    Any stage can be disabled by unchecking it in the Pipeline Stages checklist.
    Region growing can be skipped by providing a cached grown mask path — the mask
    is loaded directly, saving ~14 minutes.
    """
    triggered = ctx.triggered_id

    # ── Reset ──────────────────────────────────────────────────────────────────
    if triggered == 'btn-reset':
        sid = store.get('session_id')
        if sid:
            _cache.delete(f'ct_array_{sid}')
            with _bg_lock:
                _bg_jobs.pop(sid, None)
        empty_store = {
            'step': 0, 'mesh_path': None, 'seg_path': None, 'ct_path': None,
            'params': {}, 'mesh_vertices': None, 'mesh_triangles': None,
            'ct_vertices': None, 'metrics': None, 'ct_meta': None,
            'voltages': None, 'session_id': None,
        }
        return empty_store, "Pipeline reset.", True

    store = _store_from_data(store)
    params = _collect_params(
        seg_label, exclude_labels, intensity_tol, max_dist_mult, icp_iters, icp_algo
    )
    current_step = store.get('step', 0)
    sid = _get_or_create_session_id(store)
    store['session_id'] = sid

    # ── Poll for background region-grow completion ─────────────────────────────
    if triggered == 'grow-poll-interval':
        with _bg_lock:
            job = _bg_jobs.get(sid)
        if job is None:
            # no_update for store — avoids triggering the figure rebuild
            return no_update, "Waiting for region grow job...", False
        if not job['done']:
            # no_update for store — the figure stays untouched (camera preserved)
            return no_update, "Region growing... (this takes ~8–14 min)", False
        # Job finished
        if job['error']:
            store['step'] = 0
            return store, f"Region grow failed: {job['error']}", True
        # Success: commit results to store and advance to step 2
        grown_mask, ct_array = job['result']
        shell = pipeline.build_shell(grown_mask)
        _cache.set(f'ct_array_{sid}', ct_array)
        store['ct_vertices'] = shell.get_vertices().tolist()
        store['ct_meta'] = {'shape': list(ct_array.shape)}
        store['step'] = 2
        with _bg_lock:
            _bg_jobs.pop(sid, None)
        return store, "Region grow complete. Click 'Run Next Step' for COM alignment.", True

    # ── Next Step button ───────────────────────────────────────────────────────
    if triggered != 'btn-next-step':
        return no_update, no_update, no_update

    try:
        if current_step == 0:
            # ── Step 0: Load mesh + build CT shell ─────────────────────────────
            store['mesh_path'] = mesh_path
            store['seg_path'] = seg_path
            store['ct_path'] = ct_path
            store['params'] = params

            mesh = pipeline.load_mesh(mesh_path)
            store['mesh_vertices'] = mesh.get_vertices().tolist()
            store['mesh_triangles'] = mesh.get_triangles().tolist()
            store['voltages'] = _voltages_to_list(mesh.voltages)

            if not _stage_enabled(stages_enabled, 'grow'):
                # ── Fast path: skip region growing ─────────────────────────────
                import SimpleITK as sitk
                if grown_cache_path and os.path.exists(grown_cache_path):
                    # Load pre-computed grown mask from cache file
                    grown_mask = pipeline.load_grown_mask(grown_cache_path)
                    status_msg = f"Region growing skipped — loaded cache: {os.path.basename(grown_cache_path)}"
                else:
                    # Fall back to raw segmentation label (no growing)
                    grown_mask = pipeline.load_seg(seg_path, params.get('seg_label', 2))
                    status_msg = "Region growing skipped — using raw segmentation label (no grown mask)."

                shell = pipeline.build_shell(grown_mask)
                ct = pipeline.load_ct(ct_path)
                ct_array = sitk.GetArrayFromImage(ct)
                _cache.set(f'ct_array_{sid}', ct_array)
                store['ct_vertices'] = shell.get_vertices().tolist()
                store['ct_meta'] = {'shape': list(ct_array.shape)}
                store['step'] = 2
                return store, status_msg + " Click Next for COM alignment.", True

            else:
                # ── Slow path: run region growing in background thread ──────────
                ct = pipeline.load_ct(ct_path)

                def _grow_worker():
                    try:
                        grown = pipeline.grow_region(ct, seg_path, params)
                        import SimpleITK as sitk
                        ct_array = sitk.GetArrayFromImage(ct)
                        with _bg_lock:
                            _bg_jobs[sid] = {'done': True, 'result': (grown, ct_array), 'error': None}
                    except Exception as e:
                        with _bg_lock:
                            _bg_jobs[sid] = {'done': True, 'result': None, 'error': str(e)}

                with _bg_lock:
                    _bg_jobs[sid] = {'done': False, 'result': None, 'error': None}
                threading.Thread(target=_grow_worker, daemon=True).start()
                store['step'] = 1
                return store, "Region growing started — this takes ~8–14 min...", False

        elif current_step == 2:
            # ── Step 2: COM alignment ───────────────────────────────────────────
            if not _stage_enabled(stages_enabled, 'com'):
                store['step'] = 3
                return store, "COM alignment skipped. Click Next for PCA alignment.", True
            mesh = _rebuild_mesh(store)
            shell = _rebuild_shell(store)
            verts = pipeline.run_stage1_com(mesh, shell)
            store['mesh_vertices'] = verts.tolist()
            store['step'] = 3
            return store, "COM alignment done. Click Next for PCA axis alignment.", True

        elif current_step == 3:
            # ── Step 3: PCA axis alignment ──────────────────────────────────────
            if not _stage_enabled(stages_enabled, 'pca'):
                store['step'] = 4
                return store, "PCA alignment skipped. Click Next for axial rotation.", True
            mesh = _rebuild_mesh(store)
            shell = _rebuild_shell(store)
            verts = pipeline.run_stage2_pca(mesh, shell)
            store['mesh_vertices'] = verts.tolist()
            store['step'] = 4
            return store, "PCA alignment done. Click Next for axial rotation sweep.", True

        elif current_step == 4:
            # ── Step 4: Axial rotation sweep ────────────────────────────────────
            if not _stage_enabled(stages_enabled, 'axial'):
                store['step'] = 5
                return store, "Axial rotation skipped. Click Next for ICP refinement.", True
            mesh = _rebuild_mesh(store)
            shell = _rebuild_shell(store)
            verts = pipeline.run_stage3_axial(mesh, shell)
            store['mesh_vertices'] = verts.tolist()
            store['step'] = 5
            return store, "Axial rotation done. Click Next for ICP refinement.", True

        elif current_step == 5:
            # ── Step 5: ICP refinement + metrics ────────────────────────────────
            if not _stage_enabled(stages_enabled, 'icp'):
                store['step'] = 6
                store['metrics'] = {}
                return store, "ICP refinement skipped. Alignment complete.", True
            mesh = _rebuild_mesh(store)
            shell = _rebuild_shell(store)
            metrics = pipeline.run_icp(mesh, shell, store.get('params', params))
            store['mesh_vertices'] = mesh.get_vertices().tolist()
            store['metrics'] = metrics
            store['step'] = 6
            sym = metrics.get('symmetric_mean_dist_mm', float('nan'))
            h95 = metrics.get('hausdorff_95pct_mm', float('nan'))
            return store, f"Done! sym_mean={sym:.2f} mm, h95={h95:.2f} mm", True

        elif current_step == 6:
            return store, "Alignment complete. Use the CT slicer below or reset to start over.", True

    except Exception as e:
        return store, f"Error at step {current_step}: {e}\n{traceback.format_exc()[:400]}", True

    return store, no_update, True


# ── Callback 3: Update 3D figure ───────────────────────────────────────────────

@callback(
    Output('viewer-3d', 'figure'),
    Output('step-breadcrumb', 'children'),
    Input('pipeline-store', 'data'),
)
def render_3d_figure(store):
    """
    Rebuild the 3D alignment figure whenever the pipeline store changes.
    Uses uirevision='alignment-viewer' to preserve the camera angle between updates.
    """
    step = store.get('step', 0)
    label = pipeline.STEP_LABELS.get(step, f"Step {step}")
    breadcrumb = f"Step {step} of {pipeline.TOTAL_STEPS} — {label}"

    ct_verts = np.array(store['ct_vertices']) if store.get('ct_vertices') else None
    mesh_verts = np.array(store['mesh_vertices']) if store.get('mesh_vertices') else None
    mesh_tris = np.array(store['mesh_triangles']) if store.get('mesh_triangles') else None
    voltages = _bipolar_list(store)

    if ct_verts is None and mesh_verts is None:
        return build_empty_figure(), breadcrumb

    fig = build_alignment_figure(
        ct_vertices=ct_verts,
        mesh_vertices=mesh_verts,
        mesh_triangles=mesh_tris,
        voltages=voltages,
        title=label,
    )
    return fig, breadcrumb


# ── Callback 4: Metrics card ───────────────────────────────────────────────────

@callback(
    Output('metrics-card', 'children'),
    Input('pipeline-store', 'data'),
)
def update_metrics_card(store):
    """Show alignment quality report when pipeline reaches step 6."""
    if store.get('step', 0) < 6 or not store.get('metrics'):
        return []
    return _metrics_card(store['metrics'])


# ── Callback 5: Enable CT slicer at step 6 ────────────────────────────────────

@callback(
    Output('btn-toggle-slicer', 'disabled'),
    Output('ct-slice-slider', 'max'),
    Output('ct-slice-slider', 'value'),
    Input('pipeline-store', 'data'),
)
def enable_ct_slicer(store):
    """Enable the CT slicer button and configure the slider range when done."""
    if store.get('step', 0) < 6:
        return True, 100, 50
    ct_meta = store.get('ct_meta') or {}
    shape = ct_meta.get('shape', [100, 100, 100])
    nz = shape[0]
    return False, nz - 1, nz // 2


@callback(
    Output('ct-slicer-collapse', 'is_open'),
    Input('btn-toggle-slicer', 'n_clicks'),
    State('ct-slicer-collapse', 'is_open'),
    prevent_initial_call=True,
)
def toggle_slicer(n_clicks, is_open):
    """Toggle the CT slicer collapse panel."""
    return not is_open


# ── Callback 6: Render CT slice ────────────────────────────────────────────────

@callback(
    Output('ct-slice-figure', 'figure'),
    Output('ct-slice-label', 'children'),
    Input('ct-axis-radio', 'value'),
    Input('ct-slice-slider', 'value'),
    State('pipeline-store', 'data'),
    prevent_initial_call=True,
)
def render_ct_slice(axis, index, store):
    """Fetch a CT slice from the server-side cache and render it as a Heatmap."""
    sid = store.get('session_id')
    if not sid:
        return no_update, ""
    ct_array = _cache.get(f'ct_array_{sid}')
    if ct_array is None:
        return no_update, "CT volume not loaded"

    index = int(index or 0)
    ranges = pipeline.ct_slice_range(ct_array)
    lo, hi = ranges.get(axis, (0, 100))
    index = max(lo, min(hi, index))

    slice_arr = pipeline.get_ct_slice(ct_array, axis, index)
    label = f"{axis.capitalize()} slice {index} / {hi}"
    return build_slice_figure(slice_arr, axis, index), label
