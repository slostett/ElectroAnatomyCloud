# ElectroAnatomyCloud — Developer Guide

## Overview

This package aligns EAM (electro-anatomical mapping) mesh data from Biosense Webster
CARTO (.mesh files) to CT-derived left atrial segmentations (.nii.gz), then overlays
voltage maps on CT anatomy.

The primary interface is a Dash web application running at `http://127.0.0.1:8050`.

---

## Architecture

```
ElectroAnatomyCloud/
│
├── app/                         ← Dash web application
│   ├── app.py                   ← Layout + server entry point
│   ├── callbacks.py             ← All @app.callback definitions + pipeline state machine
│   ├── pipeline.py              ← Stateless backend wrappers (no Dash dependencies)
│   └── components/
│       ├── file_inputs.py       ← File path input panel
│       ├── param_controls.py    ← Parameter sliders/dropdowns sidebar
│       ├── viewer_3d.py         ← Plotly 3D figure builder
│       └── ct_slicer.py         ← CT cross-section slice viewer
│
├── EAM/                         ← Core alignment library
│   ├── display.py               ← PointCloud and Mesh classes
│   │   ├── PointCloud           ← Base class (vertices, prealignment, ICP)
│   │   │   ├── stage1_com()     ← COM translation
│   │   │   ├── stage2_pca()     ← PCA axis alignment + 180° flip check
│   │   │   ├── stage3_axial()   ← 1D axial rotation sweep
│   │   │   ├── structured_prealign()  ← Calls all three stages in sequence
│   │   │   ├── register_mesh_icp()    ← Trimesh P2P ICP
│   │   │   └── register_mesh_icp_p2plane()  ← Open3D P2Plane ICP
│   │   └── Mesh(PointCloud)     ← Adds triangles, voltages, rasterization
│   ├── register.py              ← Pipeline functions (region grow, metrics, etc.)
│   ├── graph.py                 ← Plotly visualization functions
│   └── main.py                  ← Legacy CLI entry point
│
├── Dockerfile                   ← Docker image definition
├── docker-compose.yml           ← Multi-container orchestration + volume mounts
├── requirements.txt             ← Pinned Python dependencies
└── pyproject.toml               ← Package metadata + build config
```

### Data Flow

```
User inputs paths in browser
        │
        ▼
callbacks.py::advance_pipeline_step()
        │
        ├── Step 1: pipeline.load_mesh() → Mesh
        │           pipeline.grow_region() → sitk.Image  [background thread, ~10 min]
        │           pipeline.build_shell() → PointCloud
        │
        ├── Step 2: mesh.stage1_com(shell)   → vertices snapshot
        ├── Step 3: mesh.stage2_pca(shell)   → vertices snapshot
        ├── Step 4: mesh.stage3_axial(shell) → vertices snapshot
        ├── Step 5: pipeline.run_icp(mesh, shell) → metrics dict
        │
        └── Step 6: render_3d_figure() → go.Figure displayed in browser
```

### Session State

`dcc.Store(id='pipeline-store')` holds a JSON-serializable dict in the browser.
The CT volume numpy array is cached server-side in `diskcache.Cache` at `ElectroAnatomyCloud/.cache/`.

---

## Running the App

### Bare Python (development)

```bash
cd ElectroAnatomyCloud
pip install -r requirements.txt   # first time only
python app/app.py
# Open http://127.0.0.1:8050
```

### Docker (recommended for distribution)

```bash
# 1. Copy and edit environment file
cp .env.example .env
# Edit .env to point to your local data directories

# 2. Build and start
docker-compose up --build -d

# 3. Open http://127.0.0.1:8050
#    IMPORTANT: use 127.0.0.1, not localhost — on Windows, WSL's wslrelay
#    may intercept localhost on IPv6 and hang the connection.

# 4. Stop
docker-compose down
```

### Docker Troubleshooting

| Problem | Fix |
|---------|-----|
| Browser hangs on `localhost:8050` | Use `127.0.0.1:8050` instead. WSL's wslrelay intercepts IPv6 localhost. |
| Container is a zombie, won't stop | `docker rm -f eam-alignment` then `docker-compose up -d` |
| App takes ~20s to respond after restart | Open3D import is slow on first load — wait for Flask log to appear |
| Port 8050 already in use | `docker rm -f eam-alignment` or check `netstat -ano \| grep 8050` |
| Stale diskcache causing errors | `docker volume rm electroanatomycloud_eam-cache` then restart |

### Quick Restart

```bash
docker restart eam-alignment
```

### Full Rebuild (after code changes)

```bash
docker rm -f eam-alignment
cd ElectroAnatomyCloud
docker-compose up --build -d
```

### Using the UI

1. Enter the full paths to your files (inside the container, data is mounted under `/data/`):
   - **EAM Mesh**: e.g., `/data/mesh/1L/Patient .../2-LA FAM.mesh`
   - **Segmentation**: e.g., `/data/seg/1L_1.nii.gz`
   - **Raw CT**: e.g., `/data/ct/raw_nii/1L_1.3.12...nii.gz`
2. **Skip region growing** (recommended for repeat runs):
   - Uncheck "Region Growing" in the Pipeline Stages checklist
   - Enter a cached grown mask path, e.g., `/data/results/1L_grown_trimmed_1p5.nii.gz`
   - This reduces Step 1 from ~14 min to ~5 seconds
3. Adjust parameters in the sidebar if needed (see Parameter Reference below).
4. Click **Run Next Step** to advance through the pipeline.
   - Any stage can be skipped by unchecking it in Pipeline Stages.
   - Step 1 (region grow) takes ~8–14 min; subsequent steps are <5 seconds each.
5. At step 6, the Alignment Quality Report shows the final metrics.
6. Click **CT Slice Viewer** to browse axial/coronal/sagittal slices.

---

## Pipeline Stages

Each stage can be independently enabled/disabled via the Pipeline Stages checklist in
the sidebar. Disabled stages are skipped and the mesh position is left unchanged.

### Step 1: Region Growing

**Purpose:** Extend the TotalSegmentator LA label to include pulmonary vein protrusions
that the segmentator does not label.

**Algorithm:**
1. Load the multi-label segmentation and extract the target label (default: 2 = LA).
2. Compute the mean HU intensity of the seed region on the (optionally smoothed) CT.
3. Set voxels belonging to excluded labels (myocardium, LV, RA, RV, aorta) to a sentinel
   value (-10000 HU) so they act as growth barriers.
4. Run SimpleITK `ConnectedThresholdImageFilter` with bounds = mean ± intensity_tolerance.
5. Cap the grown mask at `max_distance_multiplier × max_seed_boundary_distance` from the
   LA center of mass to prevent runaway growth into the lungs.
6. Remove thin disconnected bridges via connected-component analysis (`trim_disconnected_bridges`).
7. Extract the surface shell as a point cloud using marching cubes.

**Runtime:** ~8–14 min. Can be skipped by providing a pre-computed grown mask path.

### Step 2: COM Alignment

**Purpose:** Translate the EAM mesh so its center of mass coincides with the CT shell's.

**Algorithm:**
```
offset = mean(CT_vertices) - mean(mesh_vertices)
mesh_vertices += offset
```

**Runtime:** <1 second.

### Step 3: PCA Axis Alignment

**Purpose:** Rotate the mesh so its long axis aligns with the CT shell's long axis.

**Algorithm:**
1. Compute PCA first principal component on both CT shell and mesh vertices.
   - Vertices are trimmed to exclude the top 2% furthest from COM (outlier veins)
     to prevent vein tips from skewing the axis.
2. Compute the Rodrigues rotation matrix to rotate the mesh axis onto the CT axis.
3. Apply the rotation about the mesh COM.
4. Check the 180° ambiguity: compute symmetric KDTree cost for both orientations
   and keep the one with lower cost.

**Runtime:** <1 second.

### Step 4: Axial Rotation Sweep

**Purpose:** Find the optimal rotation about the CT long axis (the one axis PCA
cannot resolve).

**Algorithm:**
1. Coarse sweep: try 72 rotations at 5° intervals (0°–355°) about the CT long axis,
   centered on the CT COM.
2. Score each rotation by symmetric mean KDTree distance (mean of mesh→CT + CT→mesh
   nearest-neighbour distances).
3. Golden-section refinement: take the best coarse angle and refine within ±5° using
   golden-section search to ~0.01° precision.

**Runtime:** ~2–4 seconds.

### Step 5: ICP Refinement

**Purpose:** Fine registration via Iterative Closest Point.

**Algorithm (P2P — default):**
- Trimesh's ICP implementation. Point-to-point cost function.
- 100 iterations. Returns 4×4 transformation matrix + mean correspondence distance.

**Algorithm (P2Plane — alternative):**
- Open3D's ICP with point-to-plane cost function (requires surface normals).
- Typically converges in ~50 iterations.
- ~0.15 mm better symmetric mean but ~0.6 mm worse hausdorff 95th percentile
  (because the CT shell may not cover all mesh vein tips, and P2Plane slides
  points along the surface normal into uncovered regions).

**Runtime:** ~2–5 seconds.

### Step 6: Done

Alignment metrics are computed and displayed. CT slice viewer is unlocked.

---

## Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| Target Label | 2 | TotalSegmentator label index for the LA (heartchambers_highres: 1=myocardium, 2=LA, 3=LV, 4=RA, 5=RV, 6=aorta, 7=PA) |
| Exclude Labels | [1,3,4,5,6] | Labels set to sentinel HU to prevent region growing from crossing into other structures |
| Intensity Tolerance | 200 HU | ±HU range for ConnectedThresholdImageFilter |
| Max Distance Multiplier | 1.5 | Grown mask radius cap = N × max seed boundary distance from LA COM. 1.5 covers pulmonary veins. |
| ICP Iterations | 100 | Max ICP iterations (100 for P2P, 50 sufficient for P2Plane) |
| ICP Algorithm | P2P | Point-to-point (trimesh, robust) or Point-to-plane (Open3D, marginally better sym_mean) |

### Key Insight: Max Distance Multiplier

The LA label from TotalSegmentator does not include pulmonary veins. A multiplier of 1.0
reproduces only the LA body; increasing it allows the grown mask to extend into the vein
protrusions, which significantly reduces `hausdorff_95pct_mm`:

| Multiplier | sym_mean_mm | h95_mm |
|------------|-------------|--------|
| 1.1× | 4.05 | 16.1 |
| 1.4× | 3.18 | 11.7 |
| 1.5× | 3.10 | 11.5 |

---

## Alignment Metrics

`compute_alignment_metrics()` in `EAM/register.py` uses bidirectional KDTree nearest-neighbour:

| Metric | Description |
|--------|-------------|
| `symmetric_mean_dist_mm` | Mean of (mesh→CT + CT→mesh) nearest distances. **Primary metric.** |
| `hausdorff_95pct_mm` | 95th percentile of max nearest-neighbour distance. Measures tail coverage. |
| `mean_surface_dist_mm` | Unidirectional mean (mesh→CT). Lower than symmetric if CT shell is larger. |
| `rms_dist_mm` | Root mean square of mesh→CT distances. |

---

## How to Add a New Pipeline Stage

1. **Add a method to `PointCloud` in `EAM/display.py`:**
   ```python
   def stage_N_myname(self, other: 'PointCloud') -> np.ndarray:
       """Docstring: what this does."""
       # ... transform self.vertices in-place ...
       return self.vertices.copy()
   ```

2. **Add a wrapper in `app/pipeline.py`:**
   ```python
   def run_stage_N_myname(mesh: Mesh, shell: PointCloud) -> np.ndarray:
       """NumPy-style docstring."""
       return mesh.stage_N_myname(shell)
   ```

3. **Update `STEP_LABELS` and `TOTAL_STEPS` in `app/pipeline.py`:**
   ```python
   STEP_LABELS = {
       ...
       5: "Step 5: My New Stage",
       6: "Step 6: ICP Refinement",
       7: "Done — Alignment complete",
   }
   TOTAL_STEPS = 7
   ```

4. **Add the step branch in `app/callbacks.py::advance_pipeline_step()`:**
   ```python
   elif current_step == 5:
       mesh = _rebuild_mesh(store)
       shell = _rebuild_shell(store)
       verts = pipeline.run_stage_N_myname(mesh, shell)
       store['mesh_vertices'] = verts.tolist()
       store['step'] = 6
       return store, "My stage done.", True
   ```

---

## How to Add a New Parameter Control

1. **Add the control in `app/components/param_controls.py`** inside `build_param_controls_panel()`:
   ```python
   _labeled_control(
       "My Parameter",
       "Tooltip explaining what it does.",
       dcc.Slider(id='param-my-param', min=0, max=100, value=50, ...),
   ),
   ```

2. **Add the Input to `advance_pipeline_step` in `app/callbacks.py`:**
   ```python
   State('param-my-param', 'value'),
   ```
   And include it in the function signature and in `_collect_params()`.

3. **Handle the new key in `app/pipeline.py`** in the relevant function:
   ```python
   my_value = params.get('my_param', 50)
   ```

---

## File Formats

| Format | Source | Notes |
|--------|--------|-------|
| `.mesh` | Biosense Webster CARTO | ASCII; parsed by `parse_biosense()` in display.py |
| `.xml` | St. Jude Velocity | Parsed by `parse_st_jude_xml()` in display.py |
| `.nii.gz` | TotalSegmentator / CT scanner | Multi-label or raw HU volume |

---

## Clinical Use Warning

This software is intended for **research and academic purposes only**. It is not validated
for clinical use. See `LICENSE.md` for the full academic license terms.
