"""
param_controls.py — Parameter controls sidebar component.

Renders sliders, dropdowns, and radio buttons for all experimentally tuned
pipeline parameters. Default values match the current best configuration
determined for patient 1L (see MEMORY.md / alignment metrics table).

Architecture role
-----------------
  app/app.py imports build_param_controls_panel() and places it in the sidebar.
  app/callbacks.py reads these control values (via their IDs) when advancing
  each pipeline step, bundling them into a params dict passed to pipeline.py.

To add a new parameter
-----------------------
1. Add a control below with a unique ID (prefix 'param-').
2. Add that ID as an Input in the advance_pipeline_step callback in callbacks.py.
3. Add handling for the new key in the relevant pipeline.py function.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc

# All pipeline stages in order, with their IDs.  Checked = enabled.
STAGE_OPTIONS = [
    {'label': 'Region Growing', 'value': 'grow'},
    {'label': 'COM Alignment',  'value': 'com'},
    {'label': 'PCA Alignment',  'value': 'pca'},
    {'label': 'Axial Rotation', 'value': 'axial'},
    {'label': 'ICP Refinement', 'value': 'icp'},
]
ALL_STAGES = [s['value'] for s in STAGE_OPTIONS]


def _labeled_control(label: str, tooltip: str, control: html.Div) -> html.Div:
    """Wrap a control with a label and a Bootstrap popover tooltip."""
    tid = f"tooltip-{label.lower().replace(' ', '-')}"
    return html.Div([
        html.Div([
            dbc.Label(label, className='fw-semibold mb-1 me-1'),
            html.Span(
                "?",
                id=tid,
                className='badge bg-info rounded-circle',
                style={'cursor': 'help', 'fontSize': '0.65rem'},
            ),
            dbc.Tooltip(
                tooltip,
                target=tid,
                placement='right',
                className='small',
                style={'maxWidth': '320px'},
            ),
        ], className='d-flex align-items-center'),
        control,
    ], className='mb-3')


def build_param_controls_panel() -> html.Div:
    """
    Build the parameter controls panel.

    Returns
    -------
    html.Div
        A Dash layout div with controls for all tunable pipeline parameters.

    Control IDs (used in callbacks.py)
    ------------------------------------
    - 'param-stages-enabled'     : checklist — which stages to run (values from STAGE_OPTIONS)
    - 'param-grown-cache-path'   : str input — path to a pre-grown mask .nii.gz (skip region grow)
    - 'param-seg-label'          : int dropdown — which TotalSegmentator label is the LA
    - 'param-exclude-labels'     : checklist — labels to exclude from region growing
    - 'param-intensity-tolerance': slider — ±HU for ConnectedThresholdImageFilter
    - 'param-max-dist-mult'      : slider — grown mask radius cap multiplier
    - 'param-icp-iterations'     : slider — max ICP iterations
    - 'param-icp-algorithm'      : radio — ICP variant (P2P or P2Plane)
    """
    label_options = [
        {'label': '1 — Myocardium', 'value': 1},
        {'label': '2 — Left Atrium (default)', 'value': 2},
        {'label': '3 — Left Ventricle', 'value': 3},
        {'label': '4 — Right Atrium', 'value': 4},
        {'label': '5 — Right Ventricle', 'value': 5},
        {'label': '6 — Aorta', 'value': 6},
        {'label': '7 — Pulmonary Artery', 'value': 7},
    ]
    exclude_options = [
        {'label': '1 Myocardium', 'value': 1},
        {'label': '3 Left Ventricle', 'value': 3},
        {'label': '4 Right Atrium', 'value': 4},
        {'label': '5 Right Ventricle', 'value': 5},
        {'label': '6 Aorta', 'value': 6},
    ]

    return html.Div([
        html.H6("Pipeline Stages", className='text-uppercase text-muted mb-2 mt-1 fw-bold'),
        html.P(
            "Uncheck a stage to skip it. Useful when reloading from a cached grown mask.",
            className='small text-muted mb-2',
        ),

        dcc.Checklist(
            id='param-stages-enabled',
            options=STAGE_OPTIONS,
            value=ALL_STAGES,           # all enabled by default
            className='small',
            inputStyle={'marginRight': '6px'},
            labelStyle={'display': 'block', 'marginBottom': '4px'},
        ),

        # Cache path input — always shown; most useful when 'grow' is unchecked
        html.Div([
            html.Div([
                dbc.Label(
                    "Grown mask cache path",
                    className='fw-semibold mb-1 mt-2 me-1',
                    style={'fontSize': '0.82rem'},
                ),
                html.Span(
                    "?",
                    id='tooltip-grown-cache',
                    className='badge bg-info rounded-circle',
                    style={'cursor': 'help', 'fontSize': '0.65rem'},
                ),
                dbc.Tooltip(
                    "Path to a pre-computed grown+trimmed mask (.nii.gz). "
                    "When Region Growing is unchecked, this mask is loaded directly "
                    "instead of running the ~14-min grow step. "
                    "Leave blank to fall back to the raw segmentation label without growing.",
                    target='tooltip-grown-cache',
                    placement='right',
                    className='small',
                    style={'maxWidth': '320px'},
                ),
            ], className='d-flex align-items-center'),
            dbc.Input(
                id='param-grown-cache-path',
                type='text',
                placeholder='/data/results/grown_mask.nii.gz',
                debounce=True,
                size='sm',
                className='mt-1',
                style={'fontSize': '0.78rem'},
            ),
        ], className='mb-3'),

        html.Hr(className='my-2'),
        html.H6("Pipeline Parameters", className='text-uppercase text-muted mb-3 mt-1 fw-bold'),

        _labeled_control(
            "Target Label",
            "TotalSegmentator heartchambers_highres label index for the left atrium. "
            "Default: 2 (LA).",
            dcc.Dropdown(
                id='param-seg-label',
                options=label_options,
                value=2,
                clearable=False,
                className='form-select-sm',
            ),
        ),

        _labeled_control(
            "Exclude Labels",
            "Other cardiac structures to wall off during region growing. "
            "Voxels belonging to these labels are set to a sentinel HU value "
            "so the growing algorithm cannot cross into them.",
            dcc.Checklist(
                id='param-exclude-labels',
                options=exclude_options,
                value=[1, 3, 4, 5, 6],
                className='small',
                inputStyle={'marginRight': '6px'},
                labelStyle={'display': 'block'},
            ),
        ),

        _labeled_control(
            "Intensity Tolerance (HU)",
            "ConnectedThresholdImageFilter grows into voxels within ±tolerance HU "
            "of the seed region mean intensity. Lower values = tighter boundary. "
            "Default: 200 HU.",
            html.Div([
                dcc.Slider(
                    id='param-intensity-tolerance',
                    min=50, max=400, step=10, value=200,
                    marks={50: '50', 200: '200', 400: '400'},
                    tooltip={'placement': 'bottom', 'always_visible': False},
                ),
            ]),
        ),

        _labeled_control(
            "Max Distance Multiplier",
            "Caps the grown mask radius at N × (max seed boundary distance from LA COM). "
            "1.0 = exact seed boundary; 1.5 = 50% further (covers pulmonary vein protrusions). "
            "Default: 1.5.",
            dcc.Slider(
                id='param-max-dist-mult',
                min=1.0, max=2.5, step=0.05, value=1.5,
                marks={1.0: '1.0', 1.5: '1.5', 2.0: '2.0', 2.5: '2.5'},
                tooltip={'placement': 'bottom', 'always_visible': False},
            ),
        ),

        _labeled_control(
            "ICP Iterations",
            "Maximum number of Iterative Closest Point iterations for final refinement. "
            "100 is typically sufficient for P2P; 50 for P2Plane. "
            "Default: 100.",
            dcc.Slider(
                id='param-icp-iterations',
                min=10, max=200, step=10, value=100,
                marks={10: '10', 100: '100', 200: '200'},
                tooltip={'placement': 'bottom', 'always_visible': False},
            ),
        ),

        _labeled_control(
            "ICP Algorithm",
            "P2P (point-to-point, trimesh): robust, slightly lower mean accuracy. "
            "P2Plane (point-to-plane, Open3D): marginally better symmetric mean distance "
            "but slightly higher hausdorff 95th percentile. Default: P2P.",
            dbc.RadioItems(
                id='param-icp-algorithm',
                options=[
                    {'label': 'Point-to-Point (P2P)', 'value': 'P2P'},
                    {'label': 'Point-to-Plane (P2Plane)', 'value': 'P2Plane'},
                ],
                value='P2P',
                className='small',
                inputStyle={'marginRight': '6px'},
            ),
        ),

    ], className='p-3 border rounded bg-light')
