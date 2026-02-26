"""
pipeline.py — Stateless backend wrapper for the EAM-CT alignment pipeline.

This module provides thin, well-documented functions that wrap the core logic
in EAM/display.py and EAM/register.py for use by the Dash application.
It has NO Dash dependencies and can be imported and tested independently.

Architecture role
-----------------
  app/pipeline.py  <--calls--  EAM/display.py  (Mesh, PointCloud classes)
                   <--calls--  EAM/register.py (region_grow_from_mask, metrics, ...)
  app/callbacks.py <--calls--  app/pipeline.py

To add a new pipeline stage
----------------------------
1. Add a function here with a clean signature and NumPy-style docstring.
2. Add the corresponding PointCloud/Mesh method in EAM/display.py if needed.
3. Add a new step entry in STEP_LABELS below.
4. Wire the step in callbacks.py::advance_pipeline_step().
"""

import sys
import os
import numpy as np
import SimpleITK as sitk

# Allow importing from the EAM sibling package when running from the app/ directory
_EAM_DIR = os.path.join(os.path.dirname(__file__), '..', 'EAM')
if _EAM_DIR not in sys.path:
    sys.path.insert(0, _EAM_DIR)

from display import Mesh, PointCloud
from register import (
    region_grow_from_mask,
    trim_disconnected_bridges,
    extract_label_channel,
    compute_alignment_metrics,
)

# ── Step metadata ──────────────────────────────────────────────────────────────
# Maps step number -> human-readable name shown in the UI breadcrumb.
STEP_LABELS = {
    0: "Idle — Load files to begin",
    1: "Step 1: Region Growing",
    2: "Step 2: COM Alignment",
    3: "Step 3: PCA Axis Alignment",
    4: "Step 4: Axial Rotation",
    5: "Step 5: ICP Refinement",
    6: "Done — Alignment complete",
}

TOTAL_STEPS = 6  # steps 1-6 are active pipeline stages


# ── Data loading ───────────────────────────────────────────────────────────────

def load_mesh(mesh_path: str) -> Mesh:
    """
    Load an EAM mesh from a Biosense Webster .mesh file.

    Parameters
    ----------
    mesh_path : str
        Absolute path to the .mesh file.

    Returns
    -------
    Mesh
        Loaded mesh with vertices and triangles. Voltages are loaded automatically
        from the [VerticesColorsSection] of the .mesh file if present.

    Raises
    ------
    FileNotFoundError
        If mesh_path does not exist.
    ValueError
        If the file format cannot be detected.
    """
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    mesh = Mesh(meshpath=mesh_path)
    try:
        mesh.initialize_voltages()
    except Exception:
        pass  # voltages are optional
    return mesh


def load_seg(seg_path: str, seg_label: int = 2) -> sitk.Image:
    """
    Load a TotalSegmentator label map and extract a single label as a binary mask.

    Parameters
    ----------
    seg_path : str
        Path to the multi-label NIfTI segmentation (.nii.gz).
    seg_label : int
        Label index to extract. Default 2 = left atrium (TotalSegmentator
        heartchambers_highres label convention).

    Returns
    -------
    sitk.Image
        Binary mask (uint8, value 1 where label == seg_label).

    Raises
    ------
    FileNotFoundError
        If seg_path does not exist.
    """
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
    seg_image = sitk.ReadImage(seg_path)
    return extract_label_channel(seg_image, label_id=seg_label)


def load_ct(ct_path: str) -> sitk.Image:
    """
    Load a raw CT volume as a float32 SimpleITK image.

    Parameters
    ----------
    ct_path : str
        Path to the CT NIfTI file (.nii.gz).

    Returns
    -------
    sitk.Image
        Raw CT volume (float32).

    Raises
    ------
    FileNotFoundError
        If ct_path does not exist.
    """
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"CT file not found: {ct_path}")
    return sitk.ReadImage(ct_path, sitk.sitkFloat32)


# ── Region growing ─────────────────────────────────────────────────────────────

def grow_region(ct_sitk: sitk.Image, seg_path: str, params: dict) -> sitk.Image:
    """
    Run region growing on the CT volume seeded from the LA label in the segmentation.

    Uses SimpleITK ConnectedThresholdImageFilter with intensity bounds derived from
    the seed region mean ± intensity_tolerance HU. After growing, thin voxel bridges
    to non-LA structures are removed by trim_disconnected_bridges().

    Parameters
    ----------
    ct_sitk : sitk.Image
        Raw CT volume (float32). Pass the smoothed CT if available.
    seg_path : str
        Path to the multi-label segmentation NIfTI. Used to determine seed region
        and exclusion zones.
    params : dict
        Pipeline parameters. Relevant keys:
          - 'seg_label' (int): Label index for the LA. Default 2.
          - 'exclude_labels' (list[int]): Labels to exclude from growing. Default [1,3,4,5,6].
          - 'intensity_tolerance' (int): ±HU around seed mean. Default 200.
          - 'max_distance_multiplier' (float): Grown mask radius cap as a multiple of
            the max seed boundary distance from LA COM. Default 1.5.
          - 'bridge_trim_n' (int): Min voxels per radial ring to keep a component. Default 1000.

    Returns
    -------
    sitk.Image
        Binary grown+trimmed mask (uint8).

    Notes
    -----
    This step is the slowest in the pipeline (~8–14 min without a cache).
    The Dash app caches the result to disk so subsequent runs are instant.
    """
    label_id = params.get('seg_label', 2)
    exclude_label_ids = params.get('exclude_labels', [1, 3, 4, 5, 6])
    intensity_tolerance = params.get('intensity_tolerance', 200)
    max_distance_multiplier = params.get('max_distance_multiplier', 1.5)
    bridge_trim_n = params.get('bridge_trim_n', 1000)

    grown = region_grow_from_mask(
        ct_sitk,
        seg_path,
        label_id=label_id,
        exclude_label_ids=exclude_label_ids,
        intensity_tolerance=intensity_tolerance,
        max_distance_multiplier=max_distance_multiplier,
    )
    trimmed = trim_disconnected_bridges(grown, n=bridge_trim_n)
    return trimmed


def load_grown_mask(cache_path: str) -> sitk.Image:
    """
    Load a pre-computed grown+trimmed mask from a .nii.gz file.

    Use this to skip the ~14-min region growing step when a cached result is
    available. Pass the path via the 'Grown mask cache path' UI field.

    Parameters
    ----------
    cache_path : str
        Absolute path to the binary mask NIfTI file (.nii.gz).

    Returns
    -------
    sitk.Image
        Binary mask (uint8) ready to pass to build_shell().

    Raises
    ------
    FileNotFoundError
        If cache_path does not exist.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Grown mask cache not found: {cache_path}")
    return sitk.ReadImage(cache_path)


def build_shell(grown_mask: sitk.Image) -> PointCloud:
    """
    Extract the surface shell of a binary mask as a PointCloud.

    Parameters
    ----------
    grown_mask : sitk.Image
        Binary mask (uint8) to extract the shell from.

    Returns
    -------
    PointCloud
        Surface shell point cloud in physical (mm) coordinates.

    Notes
    -----
    Uses 6-connectivity: a voxel is on the shell if it is foreground and has
    at least one background face-neighbour.
    """
    return PointCloud(grown_mask)


# ── Alignment stages ───────────────────────────────────────────────────────────

def run_stage1_com(mesh: Mesh, shell: PointCloud) -> np.ndarray:
    """
    Stage 1: translate EAM mesh center-of-mass to the CT shell COM.

    Parameters
    ----------
    mesh : Mesh
        Moving EAM mesh. Modified in-place.
    shell : PointCloud
        Fixed CT shell. Not modified.

    Returns
    -------
    np.ndarray
        Copy of mesh vertices after the translation (Nx3, mm).
    """
    return mesh.stage1_com(shell)


def run_stage2_pca(mesh: Mesh, shell: PointCloud) -> np.ndarray:
    """
    Stage 2: align mesh long axis (PCA) to CT long axis, with 180-degree flip check.

    Parameters
    ----------
    mesh : Mesh
        Moving EAM mesh. Modified in-place.
    shell : PointCloud
        Fixed CT shell. Not modified.

    Returns
    -------
    np.ndarray
        Copy of mesh vertices after the rotation (Nx3, mm).
    """
    return mesh.stage2_pca(shell)


def run_stage3_axial(mesh: Mesh, shell: PointCloud) -> np.ndarray:
    """
    Stage 3: 1D axial rotation sweep about the CT long axis to minimise symmetric
    KDTree distance (72 x 5-degree candidates + golden-section refinement).

    Parameters
    ----------
    mesh : Mesh
        Moving EAM mesh. Modified in-place.
    shell : PointCloud
        Fixed CT shell. Not modified.

    Returns
    -------
    np.ndarray
        Copy of mesh vertices after the rotation (Nx3, mm).
    """
    return mesh.stage3_axial(shell)


def run_icp(mesh: Mesh, shell: PointCloud, params: dict) -> dict:
    """
    Final ICP refinement of the EAM mesh against the CT shell.

    Parameters
    ----------
    mesh : Mesh
        Moving EAM mesh. Modified in-place.
    shell : PointCloud
        Fixed CT shell. Not modified.
    params : dict
        Pipeline parameters. Relevant keys:
          - 'icp_iterations' (int): Max ICP iterations. Default 100.
          - 'icp_algorithm' (str): 'P2P' (point-to-point, default) or 'P2Plane'
            (point-to-plane via Open3D — marginally better sym_mean, slightly worse h95).

    Returns
    -------
    dict
        Alignment quality metrics from compute_alignment_metrics():
          - symmetric_mean_dist_mm (primary metric)
          - hausdorff_95pct_mm
          - mean_surface_dist_mm
          - rms_dist_mm
          - hausdorff_dist_mm
          - n_mesh_points, n_ct_points
    """
    iterations = params.get('icp_iterations', 100)
    algorithm = params.get('icp_algorithm', 'P2P')

    if algorithm == 'P2Plane':
        mesh.register_mesh_icp_p2plane(shell, iterations=iterations)
    else:
        mesh.register_mesh_icp(shell, iterations=iterations)

    return compute_alignment_metrics(mesh.get_vertices(), shell.get_vertices())


# ── CT slicer ──────────────────────────────────────────────────────────────────

def get_ct_slice(ct_array: np.ndarray, axis: str, index: int) -> np.ndarray:
    """
    Extract a 2D cross-section from a 3D CT volume array.

    Parameters
    ----------
    ct_array : np.ndarray
        CT volume in (z, y, x) axis order (as returned by sitk.GetArrayFromImage()).
    axis : str
        Slice plane: 'axial' (z), 'coronal' (y), or 'sagittal' (x).
    index : int
        Slice index along the chosen axis (0-based).

    Returns
    -------
    np.ndarray
        2D array of HU values for the requested slice.

    Raises
    ------
    ValueError
        If axis is not one of 'axial', 'coronal', 'sagittal'.
    """
    if axis == 'axial':
        return ct_array[index, :, :]
    elif axis == 'coronal':
        return ct_array[:, index, :]
    elif axis == 'sagittal':
        return ct_array[:, :, index]
    else:
        raise ValueError(f"axis must be 'axial', 'coronal', or 'sagittal', got '{axis}'")


def ct_slice_range(ct_array: np.ndarray) -> dict:
    """
    Return the valid slice index range for each axis of a CT volume.

    Parameters
    ----------
    ct_array : np.ndarray
        CT volume in (z, y, x) axis order.

    Returns
    -------
    dict
        {'axial': (0, nz-1), 'coronal': (0, ny-1), 'sagittal': (0, nx-1)}
    """
    nz, ny, nx = ct_array.shape
    return {
        'axial':    (0, nz - 1),
        'coronal':  (0, ny - 1),
        'sagittal': (0, nx - 1),
    }
