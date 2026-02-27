# ElectroAnatomyCloud

Alignment of electro-anatomical mapping (EAM) data from the Biosense Webster CARTO system with CT-derived cardiac segmentations. The pipeline registers `.mesh` point clouds to binary label maps (`.nii.gz`) produced by [TotalSegmentator](https://github.com/wasserth/TotalSegmentator), enabling voltage map overlays on CT anatomy.

![CT segmentation (SimpleITK)](https://github.com/slostett/ElectroAnatomyCloud/blob/main/sitk.png)
![Aligned point cloud](https://github.com/slostett/ElectroAnatomyCloud/blob/main/cloud.png)

> **Note:** No EAM or alignment data is included in this repository as it constitutes protected health information (PHI). The images above use an open-source CT segmented with TotalSegmentator. If you have open-source EAM data, please open an issue.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Configure Data Paths](#2-configure-data-paths)
  - [3. Build and Start the Container](#3-build-and-start-the-container)
  - [4. Open the Application](#4-open-the-application)
- [Using the Application](#using-the-application)
- [Docker Reference](#docker-reference)
  - [Start / Stop / Restart](#start--stop--restart)
  - [Rebuild After Code Changes](#rebuild-after-code-changes)
  - [View Logs](#view-logs)
  - [Reset Cache](#reset-cache)
  - [Troubleshooting](#troubleshooting)
- [Running Without Docker](#running-without-docker)
- [Pipeline Overview](#pipeline-overview)
- [CLI Usage](#cli-usage)
- [Library Usage](#library-usage)
- [Supported File Formats](#supported-file-formats)
- [License](#license)

---

## Features

- Interactive web UI (Dash/Plotly) for step-by-step EAM-to-CT alignment
- Region growing to extend segmentation labels into pulmonary veins
- Multi-stage prealignment: center-of-mass, PCA axis matching, axial rotation sweep
- ICP refinement (point-to-point and point-to-plane)
- Quantitative alignment metrics (symmetric mean distance, Hausdorff 95th percentile)
- CT slice viewer (axial, coronal, sagittal)
- Per-stage enable/disable toggles with checkpoint caching for fast reruns
- Dockerized deployment with volume-mounted patient data

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows, macOS, or Linux)
- [Docker Compose](https://docs.docker.com/compose/) (included with Docker Desktop)
- Patient data on your local filesystem:
  - EAM `.mesh` files exported from CARTO
  - Segmentation `.nii.gz` label maps (e.g., from TotalSegmentator `heartchambers_highres`. This will most likely need to be done using a research computing cluster and is **out of scope** for this application and repo!)
  - Raw CT `.nii.gz` volumes

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/slostett/ElectroAnatomyCloud.git
cd ElectroAnatomyCloud
```

### 2. Configure Data Paths

Copy the example environment file and edit it to point to your local data directories:

```bash
cp .env.example .env
```

Open `.env` in a text editor and set the four paths:

```dotenv
# Path to the folder containing EAM .mesh files
MESH_DIR=C:/Users/yourname/data/EAM

# Path to the folder containing segmentation .nii.gz files
SEG_DIR=C:/Users/yourname/data/segmentations

# Path to the folder containing raw CT .nii.gz files
CT_DIR=C:/Users/yourname/data/CT

# Path for output files (writable)
RESULTS_DIR=C:/Users/yourname/data/results
```

These directories are mounted into the container under `/data/`:

| Host directory | Container path | Access |
|----------------|---------------|--------|
| `MESH_DIR`     | `/data/mesh`  | read-only |
| `SEG_DIR`      | `/data/seg`   | read-only |
| `CT_DIR`       | `/data/ct`    | read-only |
| `RESULTS_DIR`  | `/data/results` | read-write |

### 3. Build and Start the Container

```bash
docker-compose up --build -d
```

The first build downloads dependencies and may take several minutes. Subsequent builds use Docker layer caching and are much faster.

### 4. Open the Application

Navigate to **http://127.0.0.1:8050** in your browser.

> **Important (Windows):** Use `127.0.0.1`, not `localhost`. On Windows with WSL, the `wslrelay` process may intercept `localhost` over IPv6 and cause the page to hang indefinitely.

---

## Using the Application

1. **Enter file paths.** All paths are relative to the container's `/data/` mount:
   - **EAM Mesh:** e.g., `/data/mesh/1L/Patient .../2-LA FAM.mesh`
   - **Segmentation:** e.g., `/data/seg/1L_1.nii.gz`
   - **Raw CT:** e.g., `/data/ct/raw_nii/1L_1.3.12...nii.gz`

2. **Configure pipeline stages.** In the sidebar, toggle individual stages on or off:
   - Region Growing, COM Alignment, PCA Alignment, Axial Rotation, ICP Refinement
   - To skip region growing on repeat runs, uncheck it and provide a cached grown mask path (e.g., `/data/results/1L_grown_trimmed_1p5.nii.gz`). This reduces step 1 from ~14 minutes to seconds.

3. **Adjust parameters.** Hover over the **?** badges next to each parameter for a description. Defaults work well for standard left atrial cases.

4. **Click "Run Next Step"** to advance through the pipeline one stage at a time. The 3D viewer updates after each step so you can inspect intermediate results.

5. **Review alignment metrics.** After the final step, the alignment quality report displays symmetric mean distance and Hausdorff 95th percentile.

6. **Browse CT slices.** Click the CT Slice Viewer tab to inspect axial, coronal, and sagittal cross-sections.

---

## Docker Reference

### Start / Stop / Restart

```bash
# Start (if already built)
docker-compose up -d

# Stop
docker-compose down

# Restart the container without rebuilding
docker restart eam-alignment
```

### Rebuild After Code Changes

```bash
docker rm -f eam-alignment
docker-compose up --build -d
```

### View Logs

```bash
# Follow live logs
docker logs -f eam-alignment

# Last 50 lines
docker logs --tail 50 eam-alignment
```

### Reset Cache

If you encounter stale cache errors, remove the persistent diskcache volume:

```bash
docker-compose down
docker volume rm electroanatomycloud_eam-cache
docker-compose up -d
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Browser hangs on `localhost:8050` | Use `127.0.0.1:8050` instead. WSL's wslrelay intercepts IPv6 localhost. |
| Container is a zombie and won't stop | `docker rm -f eam-alignment` then `docker-compose up -d` |
| App takes ~20 seconds to respond after restart | Open3D import is slow on first load. Wait for the Flask log line to appear in `docker logs`. |
| Port 8050 already in use | `docker rm -f eam-alignment`, or check what's using the port: `netstat -ano | grep 8050` |
| Stale diskcache causing errors | See [Reset Cache](#reset-cache) above. |

---

## Running Without Docker

For local development without Docker:

```bash
cd ElectroAnatomyCloud
pip install -r requirements.txt
python app/app.py
```

Open **http://127.0.0.1:8050**. File paths in the UI will reference your local filesystem directly (no `/data/` prefix needed).

Requires Python 3.10+ with the following packages: numpy, scipy, SimpleITK, scikit-learn, scikit-image, trimesh, open3d, plotly, dash, dash-bootstrap-components, tqdm, diskcache.

---

## Pipeline Overview

The alignment pipeline executes in six steps. Each stage can be independently enabled or disabled in the UI sidebar.

## Intuition (KEY!)

The intuition behind the algorithm is as follows:

1. Grow the CT extracted region to a size and shape that resembles the EAM. TotalSegmentator does not include pulmonary vein segmentation, so these regions must be added to via region growing. Other segmented structures prevent growth into non-PV structures.
2. Move EAM to CT center of mass. EAM must move to CT (not vice-versa) to preserve location in CT at the end.
3. Align long axis via PCA. Now, we've got the structures aligned, save for rotation. They're on the same line. We try flipping both ways, because in theory we could have a 180 degree mismatch.
4. Rotate until we get a decent match.
5. Improve slightly via 1 of 2 iterative closest point (ICP) algorithms.

### Step 1 — Region Growing

Extends the TotalSegmentator left atrial label into pulmonary vein protrusions that the segmentator does not label.

1. Extract the target label (default: label 2 = LA) from the multi-label segmentation.
2. Compute the mean HU intensity of the seed region on the CT volume.
3. Set excluded-label voxels (myocardium, LV, RA, RV, aorta) to a sentinel value to act as growth barriers.
4. Run SimpleITK `ConnectedThresholdImageFilter` with bounds = mean HU +/- intensity tolerance.
5. Cap the grown mask at `max_distance_multiplier` x the maximum seed boundary distance from the LA center of mass.
6. Remove thin disconnected bridges via connected-component analysis.
7. Extract the surface shell as a point cloud using marching cubes.

Runtime: ~8-14 minutes. Can be skipped by providing a pre-computed grown mask.

### Step 2 — Center-of-Mass Alignment

Translates the EAM mesh so its center of mass coincides with the CT shell's center of mass.

Runtime: <1 second.

### Step 3 — PCA Axis Alignment

Rotates the mesh so its principal axis aligns with the CT shell's principal axis. Vertices are trimmed to exclude the top 2% furthest from COM (outlier vein tips) before computing PCA. A 180-degree ambiguity check selects the orientation with the lower symmetric KDTree cost.

Runtime: <1 second.

### Step 4 — Axial Rotation Sweep

Finds the optimal rotation about the CT long axis (the one axis PCA cannot resolve). A coarse sweep tests 72 rotations at 5-degree intervals, scored by symmetric mean KDTree distance. Golden-section refinement narrows the best angle to ~0.01-degree precision.

Runtime: ~2-4 seconds.

### Step 5 — ICP Refinement

Fine registration via Iterative Closest Point. Two algorithms are available:

- **Point-to-point (default):** Trimesh ICP, 100 iterations. Robust to incomplete CT shell coverage.
- **Point-to-plane:** Open3D ICP with surface normals, ~50 iterations. Marginally better symmetric mean distance but worse tail coverage (Hausdorff 95th percentile) when the CT shell does not cover all mesh vein tips.

Runtime: ~2-5 seconds.

### Step 6 — Done

Alignment metrics are computed and the CT slice viewer is unlocked.

---

## CLI Usage

For scripted or batch processing without the web UI:

```bash
cd ElectroAnatomyCloud
python EAM/main.py \
  --meshpath "path/to/mesh.mesh" \
  --segpath "path/to/segmentation.nii.gz" \
  --output_path "path/to/output.nii.gz" \
  --segchannel 2 \
  --kmeans_alignment \
  --plot
```

| Flag | Description |
|------|-------------|
| `--meshpath` | Path to the CARTO `.mesh` file |
| `--segpath` | Path to the segmentation `.nii.gz` label map |
| `--output_path` | Where to write the aligned output |
| `--segchannel` | Label index for the target chamber (2 = LA) |
| `--kmeans_alignment` | Enable k-means cluster prealignment |
| `--euler_transform` | Apply a known Euler rotation, e.g., `"(0, 0, 0)"` |
| `--compute_euler_transform` | Brute-force search for the optimal Euler rotation (slow) |
| `--plot` | Open the Plotly viewer after alignment |

---

## Library Usage

```python
from EAM.display import Mesh

# Load a CARTO mesh
mesh = Mesh(meshpath="path/to/mesh.mesh")

# Optionally load voltage data from XML exports
mesh.initialize_voltages("path/to/folder/with/xml")

# Visualize
mesh.plot()
```

---

## Supported File Formats

| Format | Source | Description |
|--------|--------|-------------|
| `.mesh` | Biosense Webster CARTO | ASCII format containing vertices, triangles, and optional attributes |
| `.xml` | St. Jude Velocity | Per-point voltage data parsed by `parse_st_jude_xml()` |
| `.nii.gz` | TotalSegmentator / CT scanner | NIfTI volumes: multi-label segmentations or raw HU data |

---

## License

ElectroAnatomyCloud is released under the [ElectroAnatomyCloud Academic Public License](LICENSE.md). Free for noncommercial, academic, and research use. Contact the author for commercial licensing.

This software is intended for **research and academic purposes only**. It is not validated for clinical use.

2025, Stephen Lostetter
