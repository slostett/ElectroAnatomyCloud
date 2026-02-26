from graph import *
import numpy as np
import SimpleITK as sitk
import re
from sklearn.decomposition import PCA
from scipy.ndimage import convolve
from itertools import product
from scipy.spatial.transform import Rotation as R
from trimesh.registration import icp
from concurrent.futures import ProcessPoolExecutor, as_completed
from display import PointCloud, Mesh


def load_mesh_data(meshpath, with_vertex_ids=True):
    """
    Load mesh data from a .mesh file and extract vertices and triangles.

    Args:
        meshpath (str): Path to the .mesh file.
        with_vertex_ids (bool): If True, assumes vertex lines start with IDs.

    Returns:
        tuple: (vertices, triangles), where both are NumPy arrays.
    """
    vertices = []
    triangles = []
    reading_vertices = False
    reading_triangles = False

    with open(meshpath, "rb") as f:
        data = f.read().decode("ascii", errors="ignore").split('\n')

    for line in data:
        line = line.strip()
        if not line or line.startswith(";") or line.startswith("#"):
            continue
        if "[VerticesSection]" in line:
            reading_vertices = True
            reading_triangles = False
            continue
        elif "[TrianglesSection]" in line:
            reading_triangles = True
            reading_vertices = False
            continue
        elif "[" in line and "]" in line:
            reading_vertices = False
            reading_triangles = False
            continue

        # Parse vertices
        if reading_vertices:
            if with_vertex_ids:
                match = re.match(r'\s*(\d+)\s*=\s*([-\.\d]+)\s+([-\.\d]+)\s+([-\.\d]+)', line)
                if match:
                    v_id, x, y, z = map(float, match.groups())
                    vertices.append((int(v_id), x, y, z))
            else:
                match = re.match(r'\s*\d+\s*=\s*([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)', line)
                if match:
                    x, y, z = map(float, match.groups())
                    vertices.append((x, y, z))

        # Parse triangles
        if reading_triangles:
            match = re.match(r'\s*\d+\s*=\s*(\d+)\s+(\d+)\s+(\d+)', line)
            if match:
                v1, v2, v3 = map(int, match.groups())
                triangles.append((v1, v2, v3))

    vertices = np.array(vertices)
    if with_vertex_ids and vertices.shape[1] == 4:
        vertices = vertices[:, 1:]  # Drop vertex IDs if present

    return np.array(vertices), np.array(triangles)


def extract_label_channel(image: sitk.Image, label_id: int) -> sitk.Image:
    """
    Extract a binary mask for a specific label ID from a label map.

    Args:
        image (sitk.Image): Labeled segmentation image.
        label_id (int): Label value to extract.

    Returns:
        sitk.Image: Binary image where selected label is 1, others are 0.
    """
    mask = sitk.BinaryThreshold(image, lowerThreshold=label_id, upperThreshold=label_id, insideValue=1, outsideValue=0)
    mask.CopyInformation(image)
    return mask


def register_mesh_icp(fixed, moving):
    """
    Perform Iterative Closest Point (ICP) alignment of two point clouds.

    Args:
        fixed (PointCloud): Target point cloud.
        moving (PointCloud): Point cloud to align.

    Returns:
        tuple: (transformation matrix, transformed points, final cost)
    """
    if type(fixed) == PointCloud:
        fixed = fixed.get_vertices()
    if type(moving) == PointCloud:
        moving = moving.get_vertices()

    matrix, transformed, cost = icp(
        moving,
        fixed,
        scale=False,
        reflection=False,
        max_iterations=100
    )
    print("Structures aligned with cost:", cost)
    return matrix, transformed, cost


def get_long_axis(points: np.ndarray):
    """
    Finds the long axis of a 3D point cloud using PCA.

    Parameters:
    - points: (N, 3) numpy array of 3D coordinates.

    Returns:
    - center: The centroid of the point cloud (3,)
    - direction: Unit vector along the long axis (3,)
    """
    assert points.ndim == 2 and points.shape[1] == 3, "Input must be (N, 3) array"

    # Center the data
    center = points.mean(axis=0)
    centered_points = points - center

    # PCA
    pca = PCA(n_components=3)
    pca.fit(centered_points)

    # First principal component = long axis direction
    direction = pca.components_[0]

    return direction


def prealign(vertices, image_vertices):

    def find_rotation_matrix(pc_direction, target_direction):
        """
        Rotates and translates a 3D point cloud so that its long axis aligns with the target direction.

        Parameters:
        - points: (N, 3) numpy array of 3D points
        - pc_center: center of point cloud (3,)
        - pc_direction: long axis of point cloud (3,)
        - target_center: center of target (e.g. image) (3,)
        - target_direction: long axis of target (3,)

        Returns:
        - aligned_points: transformed point cloud (N, 3)
        """

        # Normalize directions
        pc_dir = pc_direction / np.linalg.norm(pc_direction)
        tgt_dir = target_direction / np.linalg.norm(target_direction)

        # Compute rotation from pc_dir -> tgt_dir
        v = np.cross(pc_dir, tgt_dir)
        c = np.dot(pc_dir, tgt_dir)
        if np.allclose(v, 0):  # already aligned or anti-aligned
            if c > 0:
                R_mat = np.eye(3)
            else:  # 180-degree rotation
                # Find arbitrary orthogonal axis
                axis = np.eye(3)[np.argmin(np.abs(pc_dir))]
                v = np.cross(pc_dir, axis)
                R_mat = R.from_rotvec(np.pi * v / np.linalg.norm(v)).as_matrix()
        else:
            skew = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R_mat = np.eye(3) + skew + (skew @ skew) * ((1 - c) / (np.linalg.norm(v) ** 2))

            return R_mat

    com = np.mean(vertices, axis=0)
    image_com = np.mean(image_vertices, axis=0)

    mesh.cluster_points_kmeans(n_clusters=5)
    mesh.merge_n_closest_clusters(3)
    direction = get_long_axis(vertices)
    image_direction = get_long_axis(image_vertices)  #TODO
    r_mat = find_rotation_matrix(direction, image_direction)

    rotated_vertices = (vertices - com) @ r_mat.T
    aligned_vertices = rotated_vertices + image_com

    return aligned_vertices


def sitk_binary_shell(sitk_image):
    from scipy.ndimage import convolve
    import numpy as np

    binary_array = sitk.GetArrayFromImage(sitk_image).astype(np.uint8)

    # 6-connectivity kernel
    kernel = np.zeros((3, 3, 3), dtype=int)
    kernel[1, 1, 0] = kernel[1, 1, 2] = 1
    kernel[1, 0, 1] = kernel[1, 2, 1] = 1
    kernel[0, 1, 1] = kernel[2, 1, 1] = 1

    neighbor_zero_count = convolve(binary_array == 0, kernel, mode='constant', cval=0)
    shell_mask = (binary_array == 1) & (neighbor_zero_count > 0)

    # Get voxel indices (z, y, x) → reorder to (x, y, z)
    indices = np.argwhere(shell_mask)
    indices = indices[:, [2, 1, 0]]

    # Physical coordinate conversion
    spacing = np.array(sitk_image.GetSpacing())
    origin = np.array(sitk_image.GetOrigin())
    direction = np.array(sitk_image.GetDirection()).reshape(3, 3)

    physical_coords = (indices * spacing) @ direction.T + origin

    return physical_coords


def _icp_with_rotation(angles, moving, fixed):
    r = R.from_euler('xyz', angles, degrees=True).as_matrix()
    rotated = moving @ r.T

    try:
        matrix, _, cost = icp(
            rotated,
            fixed,
            scale=False,
            reflection=False,
            max_iterations=30
        )

        aligned = (np.hstack([rotated, np.ones((rotated.shape[0], 1))]) @ matrix.T)[:, :3]
        return cost, angles, matrix, aligned
    except Exception as e:
        return np.inf, angles, None, None


def euler_search_icp(fixed, moving, angles_deg=(0, 45, 90, 135, 180, 225, 270, 315), max_workers=None):
    angle_combos = list(product(angles_deg, repeat=3))
    total = len(angle_combos)
    print(f"Searching over {total} rotation combinations...")

    best_cost = np.inf
    best_result = None

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_icp_with_rotation, angles, moving, fixed): angles
            for angles in angle_combos
        }

        for i, future in enumerate(as_completed(futures), 1):
            cost, angles, matrix, aligned = future.result()
            print(f"[{i}/{total}] Rotation {angles} -> Cost: {cost:.4f}")

            if cost < best_cost:
                print(f"    [BEST] New best found at {angles}")
                best_cost = cost
                best_result = (aligned, matrix, angles)

    if best_result:
        aligned, matrix, angles = best_result
        print(f"\n Best ICP result at angles: {angles} with cost {best_cost:.4f}")
        return aligned, matrix, angles
    else:
        raise RuntimeError("ICP failed on all rotations.")


def apply_euler_transform(points: np.ndarray, angles_deg: tuple, center: np.ndarray = None) -> np.ndarray:
    """
    Applies an Euler rotation about the center of a point cloud, preserving location.

    Args:
        points (np.ndarray): (N, 3) point cloud.
        angles_deg (tuple): Euler angles (X, Y, Z) in degrees.
        center (np.ndarray or None): Optional (3,) array to rotate about. Defaults to centroid.

    Returns:
        np.ndarray: Rotated point cloud with preserved position.
    """
    if center is None:
        center = points.mean(axis=0)

    # Translate to origin
    shifted = points - center

    # Apply rotation
    rot_matrix = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()
    rotated = shifted @ rot_matrix.T

    # Translate back
    return rotated + center


def mesh_to_sitk(vertices, triangles, sitk_reference_image, step_mm=0.05):
    """
    Rasterizes triangle surfaces into a binary voxel image using a reference SimpleITK image.

    Args:
        vertices (array-like): Nx3 array of (x, y, z) physical coordinates.
        triangles (array-like): Mx3 array of indices into the vertices array.
        sitk_reference_image (sitk.Image): Reference image for spacing/origin/direction.
        step_mm (float): Sampling step size in mm for filling triangle faces.

    Returns:
        sitk.Image: Sitk outline of the mesh.
    """
    import SimpleITK as sitk
    import numpy as np

    size = sitk_reference_image.GetSize()
    image = sitk.Image(size, sitk.sitkUInt8)
    image.CopyInformation(sitk_reference_image)

    def barycentric_fill(p1, p2, p3, step):
        filled_points = []
        v0 = p2 - p1
        v1 = p3 - p1
        for a in np.arange(0, 1 + step, step):
            for b in np.arange(0, 1 - a + step, step):
                point = p1 + a * v0 + b * v1
                filled_points.append(point)
        return filled_points

    count = 0
    for tri in triangles:
        pts = np.array([vertices[i] for i in tri])
        for pt in barycentric_fill(pts[0], pts[1], pts[2], step_mm / max(sitk_reference_image.GetSpacing())):
            try:
                idx = image.TransformPhysicalPointToIndex(tuple(pt))
                if all(0 <= i < s for i, s in zip(idx, size)):
                    image[idx] = 1
                    count += 1
            except RuntimeError:
                continue

    print(f"Set {count} voxels to 1 from {len(triangles)} triangles.")

    return image

def _evaluate_translation_shift(fixed, moving, shift_vector):
    try:
        shifted = moving + shift_vector
        matrix, _, cost = icp(shifted, fixed, scale=False, reflection=False, max_iterations=100)
        aligned = np.hstack([shifted, np.ones((shifted.shape[0], 1))]) @ matrix.T
        aligned = aligned[:, :3]
        return cost, shift_vector, aligned
    except Exception as e:
        return np.inf, shift_vector, moving

def translation_gradient_descent_icp(fixed, moving, initial_shift=5.0, epsilon=1e-3, max_iter=100, max_workers=None):
    '''
    Tries shifting in 6 orthogonal directions and seeing if alignment improves. if it does, continue the process
    :param fixed: PointCloud or np.array
    :param moving: PointCloud or np.array
    :param initial_shift: float or int, inital offset for translations
    :param epsilon: when to stop
    :param max_iter: max # of iterations
    :param max_workers: max # of cpu cores to use, None means it will use all
    :return:
    '''
    if type(fixed) == PointCloud:
        fixed = fixed.get_vertices()
    if type(moving) == PointCloud:
        moving = moving.get_vertices()
    directions = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])

    current_shift = initial_shift
    current_translation = np.zeros(3)
    current_points = moving.copy()

    # Initial cost
    matrix, _, best_cost = icp(current_points, fixed, scale=False, reflection=False, max_iterations=100)
    current_points = np.hstack([current_points, np.ones((current_points.shape[0], 1))]) @ matrix.T
    current_points = current_points[:, :3]

    print(f"[0] Initial ICP cost: {best_cost:.4f}")

    for i in range(1, max_iter + 1):
        # Propose all 6 directional shifts
        shift_vectors = [current_translation + d * current_shift for d in directions]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_evaluate_translation_shift, fixed, moving, shift)
                for shift in shift_vectors
            ]
            results = [f.result() for f in as_completed(futures)]

        # Find best result (including current position)
        best_result = min(results, key=lambda r: r[0])
        min_cost, best_translation, best_points = best_result

        cost_drop = best_cost - min_cost
        cost_fraction = cost_drop / best_cost if best_cost > 0 else 0

        if min_cost < best_cost:
            print(f"[{i}] Improved cost: {min_cost:.4f} (Δ={cost_drop:.4f}, scaled shift={cost_fraction:.4f})")
            best_cost = min_cost
            current_translation = best_translation
            current_points = best_points
            current_shift *= (1 - cost_fraction)
        else:
            print(f"[{i}] No improvement (best cost: {best_cost:.4f})")
            if current_shift < 1e-3 or cost_fraction < epsilon:
                print("Converged.")
                break
            current_shift *= 0.5

    return current_points, current_translation, best_cost


def cluster_points_kmeans(points: np.ndarray, n_clusters=3):
    """
    Cluster a point cloud into `n_clusters` using KMeans.

    Parameters:
        points (np.ndarray): Nx3 array of 3D coordinates.
        n_clusters (int): Number of clusters (default 3).

    Returns:
        labels (np.ndarray): Array of cluster labels (shape N).
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(points)
    return labels


def merge_n_closest_clusters(points: np.ndarray, labels: np.ndarray, n_merge: int = 3):
    """
    Merge the `n_merge` clusters whose centroids are mutually closest to each other.

    Parameters:
        points (np.ndarray): Nx3 array of coordinates.
        labels (np.ndarray): Cluster labels (e.g., from KMeans).
        n_merge (int): Number of mutually closest clusters to merge.

    Returns:
        new_labels (np.ndarray): Labels with the closest `n_merge` merged into 0;
                                 others retain their original label (offset to avoid collision).
    """
    from itertools import combinations
    from scipy.spatial.distance import pdist, squareform
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    if n_merge >= k:
        raise ValueError("n_merge must be less than total number of clusters.")

    # Step 1: Compute centroids
    centroids = np.array([points[labels == l].mean(axis=0) for l in unique_labels])

    # Step 2: Find the combination of n_merge centroids with minimal internal pairwise distance
    dist_matrix = squareform(pdist(centroids))
    best_combo = None
    best_total_dist = np.inf

    for combo in combinations(range(k), n_merge):
        submatrix = dist_matrix[np.ix_(combo, combo)]
        total_dist = submatrix.sum() / 2  # symmetric matrix, so divide by 2
        if total_dist < best_total_dist:
            best_total_dist = total_dist
            best_combo = combo

    # Step 3: Remap labels — merged group becomes 0; others get offset original labels
    merged_label_indices = np.array(best_combo)
    merged_cluster_ids = unique_labels[merged_label_indices]

    new_labels = np.full_like(labels, fill_value=-1)
    new_labels[np.isin(labels, merged_cluster_ids)] = 0  # merged group becomes 0

    next_label = 1
    for old_label in unique_labels:
        if old_label not in merged_cluster_ids:
            new_labels[labels == old_label] = next_label
            next_label += 1

    return new_labels


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 transformation matrix to an Nx3 array of points.
    """
    assert matrix.shape == (4, 4), "Expected 4x4 homogeneous matrix"

    # Convert points to homogeneous coordinates: Nx3 → Nx4
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])  # Nx4

    # Apply transformation
    transformed = points_homogeneous @ matrix.T  # Apply transform

    return transformed[:, :3]  # Return only XYZ


def region_grow_from_mask(
    ct,
    mask_path,
    label_id=0,
    exclude_label_ids=None,
    intensity_tolerance=80,
    smoothing_sigma=0,
    max_distance_multiplier=None
):
    """
    Region grow a CT volume using a labeled mask region as seed, with exclusion zones and distance limiting.
    Ensures that the original label_id region is always included in the final mask.

    Args:
        ct_path (str): Path to CT image (nii.gz).
        mask_path (str): Path to label mask (nii.gz).
        label_id (int): Channel used as seed.
        exclude_label_ids (int or list[int]): Channels used as exclusion barriers (growth stops at these).
        intensity_tolerance (float): +/- allowed intensity range from seed region.
        smoothing_sigma (float): CT smoothing sigma (in mm).
        max_distance_multiplier (float): Limits max grow radius to this multiple of max seed distance.

    Returns:
        sitk.Image: Binary grown mask (sitk.Image, same shape as CT).
    """
    import numpy as np
    import SimpleITK as sitk
    from scipy.ndimage import center_of_mass

    mask_multi = sitk.ReadImage(mask_path)
    seed_mask = extract_label_channel(mask_multi, label_id)

    # === Exclusion Zones ===
    exclude_array = np.zeros_like(sitk.GetArrayFromImage(seed_mask), dtype=np.uint8)
    if exclude_label_ids is not None:
        if isinstance(exclude_label_ids, int):
            exclude_label_ids = [exclude_label_ids]
        for ex_id in exclude_label_ids:
            ex_mask = sitk.GetArrayFromImage(extract_label_channel(mask_multi, ex_id))
            exclude_array |= (ex_mask > 0).astype(np.uint8)

    # Optional smoothing
    if smoothing_sigma > 0:
        ct = sitk.SmoothingRecursiveGaussian(ct, sigma=smoothing_sigma)
        ct = sitk.CurvatureFlow(ct, timeStep=0.125, numberOfIterations=5)

    ct_array = sitk.GetArrayFromImage(ct)
    seed_array = sitk.GetArrayFromImage(seed_mask)

    # Apply exclusion barrier
    ct_array[exclude_array > 0] = -1e6

    # Update CT
    ct = sitk.GetImageFromArray(ct_array)
    ct.CopyInformation(seed_mask)

    # Seed indices
    seed_indices = np.argwhere(seed_array > 0)
    if len(seed_indices) == 0:
        raise ValueError("No non-zero seed region found in seed mask.")

    mean_intensity = np.mean(ct_array[seed_array > 0])

    # Region Growing
    rg_filter = sitk.ConnectedThresholdImageFilter()
    rg_filter.SetLower(float(mean_intensity - intensity_tolerance))
    rg_filter.SetUpper(float(mean_intensity + intensity_tolerance))
    rg_filter.SetReplaceValue(1)

    seed_points = [tuple(int(i) for i in reversed(idx)) for idx in seed_indices]
    rg_filter.SetSeedList(seed_points)
    grown_mask = rg_filter.Execute(ct)

    # === Distance Constraint (Optional) ===
    if max_distance_multiplier is not None:
        spacing = np.array(ct.GetSpacing())[::-1]  # z, y, x
        grown_array = sitk.GetArrayFromImage(grown_mask)
        grown_indices = np.argwhere(grown_array > 0)

        com_voxel = np.array(center_of_mass(seed_array))
        distances = np.linalg.norm((grown_indices - com_voxel) * spacing, axis=1)
        max_seed_distance = np.max(np.linalg.norm((seed_indices - com_voxel) * spacing, axis=1))

        allowed = distances <= (max_distance_multiplier * max_seed_distance)
        allowed_mask = np.zeros_like(grown_array, dtype=np.uint8)
        allowed_mask[tuple(grown_indices[allowed].T)] = 1
        grown_mask = sitk.GetImageFromArray(allowed_mask)
        grown_mask.CopyInformation(ct)

    # === Ensure original region is preserved ===
    final_array = sitk.GetArrayFromImage(grown_mask) | (seed_array > 0).astype(np.uint8)
    final_mask = sitk.GetImageFromArray(final_array)
    final_mask.CopyInformation(ct)

    return final_mask


def trim_disconnected_bridges(sitk_image, n=5):
    """
    Trim regions in a binary mask (SimpleITK.Image) that are connected to the main region
    by only a narrow bridge (< n contiguous voxels).

    Args:
        sitk_image (sitk.Image): Binary mask (1 = structure, 0 = background).
        n (int): Minimum number of layer voxels required to keep growing.

    Returns:
        sitk.Image: Filtered binary mask.
    """
    from collections import deque
    from scipy.ndimage import center_of_mass, label
    from skimage.draw import disk
    from scipy.ndimage import binary_dilation
    arr = sitk.GetArrayFromImage(sitk_image).astype(np.uint8)
    shape = arr.shape
    zdim, ydim, xdim = shape

    if np.sum(arr) == 0:
        return sitk.Image(sitk_image.GetSize(), sitk.sitkUInt8)

    # Compute center column and origin voxel
    com_f = center_of_mass(arr)
    candidates = np.argwhere(arr > 0)
    com = tuple(candidates[np.argmin(np.linalg.norm(candidates - com_f, axis=1))])
    com_z, com_y, com_x = com

    max_radius = int(np.ceil(np.sqrt(ydim ** 2 + xdim ** 2)))
    included = np.zeros_like(arr, dtype=bool)
    included[:, com_y, com_x] = arr[:, com_y, com_x].astype(bool)

    # Create array of radial distances from center column (only yx plane)
    yy, xx = np.meshgrid(np.arange(ydim), np.arange(xdim), indexing='ij')
    radial_dist = np.sqrt((yy - com_y) ** 2 + (xx - com_x) ** 2)
    radial_dist = np.broadcast_to(radial_dist[None, :, :], arr.shape)

    # Start growing in cylindrical layers
    from tqdm import tqdm
    for r in tqdm(range(1, max_radius), desc="Growing by radius"):
        ring_mask = (np.floor(radial_dist) == r) & (arr > 0)
        if not np.any(ring_mask):
            continue

        # Only consider ring voxels that touch previously included voxels
        grow_mask = np.zeros_like(arr, dtype=bool)
        padded = np.pad(included, 1)
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == dy == dx == 0:
                        continue
                    shifted = padded[1 + dz:1 + dz + zdim, 1 + dy:1 + dy + ydim, 1 + dx:1 + dx + xdim]
                    grow_mask |= shifted
        border_mask = ring_mask & grow_mask

        coords = np.argwhere(border_mask)
        if coords.size == 0:
            continue

        # Get bounding box
        minz, miny, minx = coords.min(axis=0)
        maxz, maxy, maxx = coords.max(axis=0) + 1

        subvolume = border_mask[minz:maxz, miny:maxy, minx:maxx]
        labeled_sub, num = label(subvolume, structure=np.ones((3, 3, 3)))

        for comp_id in range(1, num + 1):
            comp_mask = (labeled_sub == comp_id)
            comp_size = np.sum(comp_mask)

            if comp_size >= n:
                included[minz:maxz, miny:maxy, minx:maxx] |= comp_mask

    # Convert to sitk
    output_image = sitk.GetImageFromArray(included.astype(np.uint8))
    output_image.CopyInformation(sitk_image)
    return output_image

def compute_alignment_metrics(mesh_vertices: np.ndarray, ct_vertices: np.ndarray) -> dict:
    """
    Compute surface distance metrics between an aligned mesh and a CT shell point cloud.

    Args:
        mesh_vertices (np.ndarray): Nx3 array of aligned mesh vertex positions (mm).
        ct_vertices (np.ndarray): Mx3 array of CT shell surface points (mm).

    Returns:
        dict: Metric name → float value. All distances are in the same units as the inputs (typically mm).
              Keys: mean_surface_dist_mm, symmetric_mean_dist_mm, rms_dist_mm,
                    hausdorff_dist_mm, hausdorff_95pct_mm, n_mesh_points, n_ct_points.
    """
    from scipy.spatial import KDTree

    tree_ct = KDTree(ct_vertices)
    tree_mesh = KDTree(mesh_vertices)

    d_mesh_to_ct, _ = tree_ct.query(mesh_vertices)
    d_ct_to_mesh, _ = tree_mesh.query(ct_vertices)

    return {
        "mean_surface_dist_mm":   float(np.mean(d_mesh_to_ct)),
        "symmetric_mean_dist_mm": float((np.mean(d_mesh_to_ct) + np.mean(d_ct_to_mesh)) / 2),
        "rms_dist_mm":            float(np.sqrt(np.mean(d_mesh_to_ct ** 2))),
        "hausdorff_dist_mm":      float(max(np.max(d_mesh_to_ct), np.max(d_ct_to_mesh))),
        "hausdorff_95pct_mm":     float(max(np.percentile(d_mesh_to_ct, 95), np.percentile(d_ct_to_mesh, 95))),
        "n_mesh_points":          int(len(mesh_vertices)),
        "n_ct_points":            int(len(ct_vertices)),
    }


def skeletonize_sitk_image(sitk_img: sitk.Image) -> sitk.Image:
    """
    Skeletonize a SimpleITK binary image using 3D skeletonization.

    Args:
        sitk_img (sitk.Image): Input binary image (label or mask).

    Returns:
        sitk.Image: Skeletonized binary image.
    """
    # Convert to NumPy (z, y, x)
    from skimage.morphology import skeletonize
    img_np = sitk.GetArrayFromImage(sitk_img).astype(bool)

    # Skeletonize
    skeleton_np = skeletonize(img_np)

    # Convert back to sitk.Image (SimpleITK expects (x, y, z))
    skeleton_sitk = sitk.GetImageFromArray(skeleton_np.astype(np.uint8))
    skeleton_sitk.CopyInformation(sitk_img)

    return skeleton_sitk

def emphasize_periphery(pc: 'PointCloud', scale_factor=1.0, base=np.e) -> 'PointCloud':
    """
    Transforms a PointCloud by exaggerating points farther from the center of mass.

    Each point is scaled exponentially based on its distance from the center:
        new_point = COM + unit_vector * exp(scale_factor * distance)

    Args:
        pc (PointCloud): Original point cloud.
        scale_factor (float): Controls how aggressively to exaggerate distances.
        base (float): Exponential base (e.g., e or 2).

    Returns:
        PointCloud: Transformed point cloud in the exaggerated space.
    """
    vertices = np.array(pc.get_vertices())
    center = np.mean(vertices, axis=0)

    # Displacement vectors
    directions = vertices - center
    distances = np.linalg.norm(directions, axis=1)

    # Avoid division by zero
    unit_vectors = np.divide(
        directions,
        distances[:, np.newaxis],
        out=np.zeros_like(directions),
        where=distances[:, np.newaxis] != 0
    )

    # Exponentially scaled distances
    scaled_distances = base ** (scale_factor * distances)

    # New positions
    new_vertices = center + unit_vectors * scaled_distances[:, np.newaxis]

    return PointCloud(new_vertices)


if __name__ == "__main__":
    # === User input paths ===
    meshpath = "C:/Users/steph/Documents/UNC Cardiac Imaging/EAM data/1L/Patient 2025_07_08/PVI/Export_PVI-08_19_2025-15-41-50/2-LA FAM.mesh"
    #folder = "C:/Users/steph/Documents/UNC Cardiac Imaging/EAM data/ExportData28_02_25 16_19_56/Patient 2025_02_28/AF/Export_AF-02_28_2025-16-01-43/6-1-sinus.mesh"
    nii_path = "C:/Users/steph/Documents/UNC Cardiac Imaging/CT Data/1L_1.nii.gz"
    ct_path = "C:/Users/steph/Documents/UNC Cardiac Imaging/CT Data/raw_nii/1L_1.3.12.2.1107.5.1.4.105354.30000025070712213494500039033.nii.gz"

    #meshpath_2 = "C:/Users/steph/Downloads/Atrium_L.nii.gz"

    # === Load mesh ===
    #vertices, triangles = load_mesh_data(folder, with_vertex_ids=True)
    mesh = Mesh(meshpath=meshpath)
    mesh.initialize_voltages()

    # === Load NIfTI and extract label ===

    la_seg_image = sitk.ReadImage(nii_path)
    ct = sitk.ReadImage(ct_path, sitk.sitkFloat32)
    '''
    smoothed = sitk.CurvatureAnisotropicDiffusion(
        ct,
        timeStep=.0075,
        conductanceParameter=50,
        numberOfIterations=50
    )
    sitk.WriteImage(smoothed, 'C:/users/steph/downloads/smoothed.nii.gz')
    '''
    smoothed = sitk.ReadImage('C:/Users/steph/Documents/UNC Cardiac Imaging/CT Data/smoothed/smoothed.nii.gz', sitk.sitkFloat32)

    la_mask = extract_label_channel(la_seg_image, label_id=2)

    grown_cache = 'C:/Users/steph/Documents/UNC Cardiac Imaging/results/1L_grown_trimmed_1p5.nii.gz'
    import os
    if os.path.exists(grown_cache):
        print('Loading cached grown mask...')
        la_mask_grown = sitk.ReadImage(grown_cache)
    else:
        print('Growing image')
        la_mask_grown = region_grow_from_mask(smoothed, nii_path, intensity_tolerance=200, label_id=2, exclude_label_ids=[1, 3, 4, 5, 6], max_distance_multiplier=1.5)
        print('Trimming')
        la_mask_grown = trim_disconnected_bridges(la_mask_grown, 1000)
        sitk.WriteImage(la_mask_grown, grown_cache)
        print(f'Saved grown mask to cache: {grown_cache}')

    la_shell = PointCloud(la_mask)
    print(len(la_shell.vertices))
    la_shell = PointCloud(la_mask_grown)
    print(len(la_shell.vertices))

    #print(vertices)
    #print(triangles)

    print('Prealigning (structured 3-stage)')
    mesh.structured_prealign(la_shell)

    print('Final ICP refinement')
    mesh.register_mesh_icp(la_shell, iterations=100)


    #vertices_aligned, transform, angles = euler_search_icp(fixed=la_shell, moving=vertices)

    #registered_verts, _, _ = translation_gradient_descent_icp(la_shell, vertices_transformed, initial_shift=10.0)

    metrics = compute_alignment_metrics(mesh.get_vertices(), la_shell.get_vertices())
    print("\n=== Alignment Quality Report ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print("================================\n")

    print('Converting back to SITK')
    print(len(mesh.get_vertices()))
    mesh_img = mesh.mesh_to_sitk(la_mask_grown, step_mm=0.05)

    plot_sitk_images(mesh_img, la_mask_grown)
    #show_multiple_images_3d([mesh, la_mask])

    sitk.WriteImage(mesh_img, 'C:/users/steph/downloads/EAM.nii.gz')

    #meshpath = '/nas/longleaf/home/slostett/cardiacfibrosis/6-1-sinus.mesh'
    #nii_path = "/nas/longleaf/home/slostett/cardiacfibrosis/results_totalseg.nii"
