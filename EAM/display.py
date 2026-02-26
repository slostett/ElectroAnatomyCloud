from graph import *
import numpy as np
import re
import xml.etree.ElementTree as ET
import os
from pathlib import Path
from trimesh import Trimesh
import SimpleITK as sitk
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from trimesh.registration import icp

def find_mesh_files(root_dir):
    files = [str(path) for path in Path(root_dir).rglob("*.mesh")]
    if len(files) == 0:
        print('Zero files found. Make sure you\'ve unzipped subfolders. Use 7zip if you can\'t unzip the normal way.')
    print(f'{len(files)} mesh files found in your directory or subfolders:\n')
    for file in files:
        print(file)
    return files

class PointCloud:
    def __init__(self, vertices, labels=None):
        if type(vertices) == sitk.Image:
            self.vertices = self.sitk_binary_shell(vertices)
        else:
            self.vertices = np.array(vertices)

        self.labels = labels

    def get_vertices(self):
        return self.vertices

    def get_labels(self):
        return self.labels

    def plot(self):
        from graph import plot_shell_point_cloud
        plot_shell_point_cloud(self.vertices)

    @staticmethod
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

    def cluster_points_kmeans(self, n_clusters=3):
        """
        Cluster a point cloud into `n_clusters` using KMeans.

        Parameters:
            points (np.ndarray): Nx3 array of 3D coordinates.
            n_clusters (int): Number of clusters (default 3).

        Returns:
            labels (np.ndarray): Array of cluster labels (shape N).
        """

        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        labels = kmeans.fit_predict(self.vertices)
        self.labels=labels

    def merge_n_closest_clusters(self, n_merge: int = 3):
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
        unique_labels = np.unique(self.labels)
        k = len(unique_labels)
        if n_merge >= k:
            raise ValueError("n_merge must be less than total number of clusters.")

        # Step 1: Compute centroids
        centroids = np.array([self.vertices[self.labels == l].mean(axis=0) for l in unique_labels])

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

        new_labels = np.full_like(self.labels, fill_value=-1)
        new_labels[np.isin(self.labels, merged_cluster_ids)] = 0  # merged group becomes 0

        next_label = 1
        for old_label in unique_labels:
            if old_label not in merged_cluster_ids:
                new_labels[self.labels == old_label] = next_label
                next_label += 1

        self.labels = new_labels
        return new_labels

    def apply_transform(self, matrix: np.ndarray) -> np.ndarray:
        """
        Apply a 4x4 transformation matrix to an Nx3 array of points.
        """
        assert matrix.shape == (4, 4), "Expected 4x4 homogeneous matrix"

        # Convert points to homogeneous coordinates: Nx3 → Nx4
        ones = np.ones((self.vertices.shape[0], 1))
        points_homogeneous = np.hstack([self.vertices, ones])  # Nx4

        # Apply transformation
        transformed = points_homogeneous @ matrix.T  # Apply transform

        self.vertices = transformed[:, :3]  # Return only XYZ
        return self.vertices

    def apply_euler_transform(self, angles_deg: tuple, center: np.ndarray = None) -> np.ndarray:
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
            center = self.vertices.mean(axis=0)

        # Translate to origin
        shifted = self.vertices - center

        # Apply rotation
        rot_matrix = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()
        rotated = shifted @ rot_matrix.T

        # Translate back
        self.vertices = rotated + center
        return self.vertices

    def get_long_axis(self):
        """
        Finds the long axis of a 3D point cloud using PCA.

        Parameters:
        - points: (N, 3) numpy array of 3D coordinates.

        Returns:
        - center: The centroid of the point cloud (3,)
        - direction: Unit vector along the long axis (3,)
        """
        if len(np.unique(self.labels)) > 1:
            vertices = self.vertices[self.labels == 0]
        else:
            vertices = self.get_vertices()
        assert vertices.ndim == 2 and vertices.shape[1] == 3, "Input must be (N, 3) array"

        # Center the data
        center = vertices.mean(axis=0)
        centered_points = vertices - center

        # PCA
        pca = PCA(n_components=3)
        pca.fit(centered_points)

        # First principal component = long axis direction
        direction = pca.components_[0]

        return direction

    def prealign(self, other, k_means=True):
        '''
        Transforms the vertices of this PointCloud or Mesh to be close to those of the other.
        Note that self will move, other is fixed.
        :param other: PointCloud, Mesh, fixed.
        :return: None, self.vertices will be updated.
        '''

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
        if k_means:
            self.cluster_points_kmeans(5)
            self.merge_n_closest_clusters(3)
            com = np.mean(self.vertices[self.labels == 0], axis=0)
        else:
            com = np.mean(self.vertices, axis=0)

        image_com = np.mean(other.vertices, axis=0)

        direction = self.get_long_axis()
        image_direction = other.get_long_axis()
        r_mat = find_rotation_matrix(direction, image_direction)

        rotated_vertices = (self.vertices - com) @ r_mat.T
        self.vertices = rotated_vertices + image_com

    # ── Shared helpers for the 3-stage prealign ──────────────────────────────
    @staticmethod
    def _pca_axis(pts):
        """Return first principal component (unit vector) of pts (Nx3), trimming top 2% outliers."""
        com = pts.mean(axis=0)
        dists = np.linalg.norm(pts - com, axis=1)
        keep = dists <= np.percentile(dists, 98)
        trimmed = pts[keep]
        centered = trimmed - trimmed.mean(axis=0)
        pca = PCA(n_components=3)
        pca.fit(centered)
        return pca.components_[0]

    @staticmethod
    def _rodrigues(axis, theta_rad):
        """Rodrigues rotation matrix: rotate by theta_rad about unit axis."""
        u = axis / np.linalg.norm(axis)
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        skew = np.array([[0, -u[2], u[1]],
                          [u[2], 0, -u[0]],
                          [-u[1], u[0], 0]])
        return c * np.eye(3) + s * skew + (1 - c) * np.outer(u, u)

    @staticmethod
    def _sym_cost(pts_a, pts_b):
        """Symmetric mean nearest-neighbour distance (mm) between two point clouds."""
        from scipy.spatial import KDTree
        tree_b = KDTree(pts_b)
        tree_a = KDTree(pts_a)
        d_ab, _ = tree_b.query(pts_a)
        d_ba, _ = tree_a.query(pts_b)
        return (d_ab.mean() + d_ba.mean()) / 2.0

    @staticmethod
    def _rotate_about_axis(pts, axis, theta_rad, center):
        """Rotate pts about an axis passing through center by theta_rad radians."""
        R_mat = PointCloud._rodrigues(axis, theta_rad)
        return (pts - center) @ R_mat.T + center

    def stage1_com(self, other: 'PointCloud') -> np.ndarray:
        """
        Stage 1 prealignment: translate mesh center-of-mass to CT shell COM.

        Parameters
        ----------
        other : PointCloud
            Fixed CT shell point cloud. Not modified.

        Returns
        -------
        np.ndarray
            Copy of self.vertices after the translation (for visualization snapshots).

        Notes
        -----
        Modifies self.vertices in-place. Call this before stage2_pca().
        """
        ct_com = other.get_vertices().mean(axis=0)
        self.vertices += (ct_com - self.vertices.mean(axis=0))
        print(f"[Prealign] Stage 1: COM alignment complete. "
              f"Mesh COM now at {self.vertices.mean(axis=0).round(1)}")
        return self.vertices.copy()

    def stage2_pca(self, other: 'PointCloud') -> np.ndarray:
        """
        Stage 2 prealignment: align mesh PCA long axis to CT long axis via Rodrigues rotation.
        Includes automatic 180-degree flip disambiguation using symmetric KDTree distance.

        Parameters
        ----------
        other : PointCloud
            Fixed CT shell point cloud. Not modified.

        Returns
        -------
        np.ndarray
            Copy of self.vertices after the rotation (for visualization snapshots).

        Notes
        -----
        Modifies self.vertices in-place. Call stage1_com() before this method.
        """
        ct_pts = other.get_vertices()
        ct_com = ct_pts.mean(axis=0)
        ct_axis = self._pca_axis(ct_pts)
        mesh_axis = self._pca_axis(self.vertices)

        # Rodrigues rotation: mesh_axis -> ct_axis
        v = np.cross(mesh_axis, ct_axis)
        c = np.dot(mesh_axis, ct_axis)
        if np.allclose(v, 0):
            if c > 0:
                R2 = np.eye(3)
            else:
                perp = np.eye(3)[np.argmin(np.abs(mesh_axis))]
                rv = np.cross(mesh_axis, perp)
                R2 = self._rodrigues(rv / np.linalg.norm(rv), np.pi)
        else:
            skew = np.array([[0, -v[2], v[1]],
                              [v[2], 0, -v[0]],
                              [-v[1], v[0], 0]])
            R2 = np.eye(3) + skew + skew @ skew * ((1 - c) / (np.linalg.norm(v) ** 2))

        center = ct_com
        self.vertices = (self.vertices - center) @ R2.T + center

        # 180-degree flip disambiguation
        flipped = self._rotate_about_axis(self.vertices, ct_axis, np.pi, center)
        cost_normal = self._sym_cost(self.vertices, ct_pts)
        cost_flipped = self._sym_cost(flipped, ct_pts)
        if cost_flipped < cost_normal:
            print(f"[Prealign] Stage 2: 180-deg flip applied "
                  f"(cost {cost_normal:.3f} -> {cost_flipped:.3f})")
            self.vertices = flipped
        else:
            print(f"[Prealign] Stage 2: no flip needed "
                  f"(cost {cost_normal:.3f} vs flipped {cost_flipped:.3f})")
        return self.vertices.copy()

    def stage3_axial(self, other: 'PointCloud') -> np.ndarray:
        """
        Stage 3 prealignment: 1D rotation sweep about the CT long axis to minimise symmetric
        KDTree distance. Uses a 5-degree grid (72 candidates) followed by golden-section
        refinement within ±15 degrees of the best grid candidate.

        This implicitly aligns the pulmonary vein protrusions on the mesh to their
        corresponding positions on the CT shell.

        Parameters
        ----------
        other : PointCloud
            Fixed CT shell point cloud. Not modified.

        Returns
        -------
        np.ndarray
            Copy of self.vertices after the rotation (for visualization snapshots).

        Notes
        -----
        Modifies self.vertices in-place. Call stage2_pca() before this method.
        The CT long axis (not the mesh axis) is used as the rotation axis because
        the CT shell is complete and its principal axis is more reliable.
        """
        from scipy.optimize import minimize_scalar

        ct_pts = other.get_vertices()
        ct_com = ct_pts.mean(axis=0)
        ct_axis = self._pca_axis(ct_pts)
        center = ct_com

        print("[Prealign] Stage 3: Axial rotation sweep (72 x 5-deg steps)")
        thetas = np.deg2rad(np.arange(0, 360, 5))
        costs = np.array([
            self._sym_cost(self._rotate_about_axis(self.vertices, ct_axis, t, center), ct_pts)
            for t in thetas
        ])
        best_idx = int(np.argmin(costs))
        best_theta = thetas[best_idx]
        print(f"[Prealign] Stage 3: grid best at {np.rad2deg(best_theta):.1f} deg "
              f"(cost {costs[best_idx]:.3f})")

        lo = best_theta - np.deg2rad(15)
        hi = best_theta + np.deg2rad(15)

        def _cost_scalar(theta):
            return self._sym_cost(
                self._rotate_about_axis(self.vertices, ct_axis, theta, center), ct_pts
            )

        result = minimize_scalar(_cost_scalar, bounds=(lo, hi), method='bounded',
                                 options={'xatol': np.deg2rad(0.1)})
        refined_theta = result.x
        print(f"[Prealign] Stage 3: refined to {np.rad2deg(refined_theta):.2f} deg "
              f"(cost {result.fun:.3f})")

        self.vertices = self._rotate_about_axis(self.vertices, ct_axis, refined_theta, center)
        return self.vertices.copy()

    def structured_prealign(self, other: 'PointCloud') -> None:
        """
        3-stage geometric prealignment of self (EAM mesh) to other (CT shell).
        Self is moved; other is fixed. Updates self.vertices in-place.

        Delegates to the three individual stage methods for modularity:
          - stage1_com()   : center-of-mass translation
          - stage2_pca()   : long-axis alignment (Rodrigues + 180-deg disambiguation)
          - stage3_axial() : 1D rotation sweep about CT long axis

        Parameters
        ----------
        other : PointCloud
            Fixed CT shell. Not modified.

        Notes
        -----
        To visualize intermediate states (e.g. in a step-by-step UI), call the
        stage methods individually rather than this combined method.
        """
        self.stage1_com(other)
        self.stage2_pca(other)
        self.stage3_axial(other)
        print("[Prealign] Done.")

    def register_mesh_icp_p2plane(self, other, iterations=50, normal_radius=5.0, max_correspondence=10.0):
        """
        Point-to-plane ICP using Open3D. Converges faster and more accurately than point-to-point
        for smooth anatomical surfaces (atrium wall). Normals are estimated from the surface geometry.

        Args:
            other (PointCloud): Fixed CT shell point cloud.
            iterations (int): Max ICP iterations (50 is typically sufficient for P2P-plane).
            normal_radius (float): Radius in mm for normal estimation neighbourhood.
            max_correspondence (float): Max correspondence distance in mm.

        Returns:
            tuple: (4x4 transform matrix as np.ndarray, final symmetric_mean_dist_mm)
        """
        import open3d as o3d
        from scipy.spatial import KDTree

        def _to_o3d(pts):
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            return pc

        moving_pts = self.get_vertices()
        fixed_pts = other.get_vertices()

        # Voxel-downsample the CT shell to roughly match mesh density
        if len(fixed_pts) > 2 * len(moving_pts):
            target_n = int(1.5 * len(moving_pts))
            pts_range = fixed_pts.max(axis=0) - fixed_pts.min(axis=0)
            voxel_size = pts_range.max() / (target_n ** (1/3)) * 0.5
            o3d_fixed_full = _to_o3d(fixed_pts)
            o3d_fixed_ds = o3d_fixed_full.voxel_down_sample(voxel_size=voxel_size)
            fixed_pts_icp = np.asarray(o3d_fixed_ds.points)
            print(f"[P2Plane ICP] CT shell downsampled: {len(fixed_pts)} -> {len(fixed_pts_icp)} points")
        else:
            fixed_pts_icp = fixed_pts

        o3d_moving = _to_o3d(moving_pts)
        o3d_fixed = _to_o3d(fixed_pts_icp)

        # Estimate normals (required for point-to-plane)
        search_param = o3d.geometry.KDTreeSearchParamRadius(radius=normal_radius)
        o3d_moving.estimate_normals(search_param=search_param)
        o3d_fixed.estimate_normals(search_param=search_param)

        # Orient normals consistently inward (for a closed LA surface, orient away from COM)
        o3d_moving.orient_normals_towards_camera_location(
            camera_location=np.asarray(o3d_moving.points).mean(axis=0)
        )
        o3d_fixed.orient_normals_towards_camera_location(
            camera_location=np.asarray(o3d_fixed.points).mean(axis=0)
        )

        result = o3d.pipelines.registration.registration_icp(
            source=o3d_moving,
            target=o3d_fixed,
            max_correspondence_distance=max_correspondence,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations)
        )

        matrix = np.array(result.transformation)
        self.apply_transform(matrix)

        # Report symmetric cost on full CT shell
        tree_ct = KDTree(other.get_vertices())
        tree_mesh = KDTree(self.get_vertices())
        d_m2c, _ = tree_ct.query(self.get_vertices())
        d_c2m, _ = tree_mesh.query(other.get_vertices())
        sym_cost = (d_m2c.mean() + d_c2m.mean()) / 2.0
        print(f"[P2Plane ICP] Done. Fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.4f}, "
              f"sym_mean={sym_cost:.3f} mm")
        return matrix, sym_cost

    def register_mesh_icp(self, other, iterations=100, focus_outer_fraction=None, plot=False):
        """
        Perform Iterative Closest Point (ICP) alignment of two point clouds.

        Args:
            self (PointCloud): moving point cloud.
            other (PointCloud): fixed point cloud.
            iterations (int): Maximum number of ICP iterations.
            focus_outer_fraction (float or None): If set (e.g., 0.5), keeps only points further than this fraction
                                                  of the max distance from the center of mass to emphasize outer structure.

        Returns:
            tuple: (transformation matrix, transformed points, final cost)
        """

        def filter_outer_points(points, fraction):
            com = np.mean(points, axis=0)
            dists = np.linalg.norm(points - com, axis=1)
            max_dist = np.max(dists)
            mask = dists >= (fraction * max_dist)
            return points[mask]

        moving = self.get_vertices()
        fixed = other.get_vertices()

        # Apply outer shell filtering if specified
        if focus_outer_fraction is not None:
            moving = filter_outer_points(moving, focus_outer_fraction)
            fixed = filter_outer_points(fixed, focus_outer_fraction)


        # Balance point count
        if len(moving) > len(fixed):
            moving = moving[np.random.choice(len(moving), size=len(fixed), replace=False)]
        elif len(fixed) > len(moving):
            fixed = fixed[np.random.choice(len(fixed), size=len(moving), replace=False)]

        from graph import show_multiple_images_3d
        from display import PointCloud
        if focus_outer_fraction and plot:
            show_multiple_images_3d([PointCloud(moving), PointCloud(fixed)], colors=['red', 'blue'])

        print(f"Filtered moving points: {len(moving)}, fixed points: {len(fixed)}")

        matrix, transformed, cost = icp(
            moving,
            fixed,
            scale=False,
            reflection=False,
            max_iterations=iterations
        )
        print("PointCloud aligned to target with cost:", cost)
        self.apply_transform(matrix)
        return matrix, transformed, cost


class Mesh(PointCloud):
    def __init__(self, vertices=None, triangles=None, meshpath=None):
        '''
        Mesh can either be manually loaded in or be immediately passed a path name via the meshpath
        keyword argument to load quickly.
        :param vertices: iterable
        :param triangles: iterable
        :param meshpath: string path to file if quickload
        '''
        assert (vertices is not None and triangles is not None) or meshpath is not None, "Mesh initialized incorrectly. If you are quickloading from filename, you need meshpath="
        PointCloud.__init__(self, vertices)
        self.meshpath = meshpath
        if meshpath is not None:
            vertices, triangles = load_mesh_data(meshpath)
            self.vertices, self.triangles = np.array(vertices), np.array(triangles)
        else:
            self.vertices, self.triangles = np.array(vertices), np.array(triangles)
        self.voltages = None

    def get_triangles(self):
        return self.triangles

    def get_voltages(self):
        return self.voltages

    def initialize_voltages(self, voltages=None, xml_dir=None):
        if voltages:
            self.voltages = voltages
        else:
            #self.voltages = load_voltage_data_from_xml(self.meshpath, xml_dir)
            self.voltages = load_voltage_colors_from_mesh(self.meshpath)

    def remove_unused_vertices(self):
        """
        Remove vertices not referenced by any triangle and update triangle indices accordingly.

        Parameters:
            vertices (list of tuples): List of (x, y, z) coordinates.
            triangles (list of tuples): List of (i1, i2, i3) index triples referencing the vertices.

        Returns:
            new_vertices (list of tuples): Filtered list of used vertices.
            new_triangles (list of tuples): Updated triangle indices.
        """
        # Step 1: Find all used vertex indices
        vertices = self.vertices
        triangles = self.triangles
        # Step 1: Get unique indices used in triangles
        used_indices = np.unique(triangles)

        # Step 2: Create a mapping from old to new indices
        index_map = -np.ones(vertices.shape[0], dtype=int)
        index_map[used_indices] = np.arange(len(used_indices))

        # Step 3: Apply mapping to triangles
        new_triangles = index_map[triangles]

        # Step 4: Filter the vertices
        new_vertices = vertices[used_indices]

        self.vertices, self.triangles = new_vertices, new_triangles


    def plot(self, voltage_type='Bipolar'):
        if self.voltages is None:
            print('Plotting without voltages. If you have voltages, call .initialize_voltages first')
            plot_meshes_3d([Trimesh(vertices=self.vertices, faces=self.triangles)])
        if self.voltages is not None:
            if voltage_type == "Impedance":
                print(f'Plotting with {voltage_type}s. Work in progress')
            else:
                print(f'Plotting with {voltage_type} voltages.')
            from graph import plot_voltages_3d_color_adjust
            plot_voltages_3d_color_adjust(self.vertices, self.triangles, self.voltages, voltage_type)

    def mesh_to_sitk(self, sitk_reference_image, step_mm=0.05):
        """
        Rasterizes triangle surfaces into a binary voxel image using a reference SimpleITK image.

        Args:
            sitk_reference_image (sitk.Image): Reference image for spacing/origin/direction.
            step_mm (float): Sampling step size in mm for filling triangle faces.

        Returns:
            sitk.Image: Sitk outline of the mesh.
        """

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
        for tri in self.triangles:
            pts = np.array([self.vertices[i] for i in tri])
            for pt in barycentric_fill(pts[0], pts[1], pts[2], step_mm / max(sitk_reference_image.GetSpacing())):
                try:
                    idx = image.TransformPhysicalPointToIndex(tuple(pt))
                    if all(0 <= i < s for i, s in zip(idx, size)):
                        image[idx] = 1
                        count += 1
                except RuntimeError as e:
                    print("Transform failed for point:", pt)
                    print("Error:", e)
                    continue

        print(f"Set {count} voxels to 1 from {len(self.triangles)} triangles.")

        return image


def parse_biosense(lines, with_vertex_ids=False):
    vertices = []
    triangles = []
    reading_vertices = False
    reading_triangles = False

    for line in lines:
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

        if reading_triangles:
            match = re.match(r'\s*\d+\s*=\s*(\d+)\s+(\d+)\s+(\d+)', line)
            if match:
                v1, v2, v3 = map(int, match.groups())
                triangles.append((v1, v2, v3))

    return np.array(vertices), np.array(triangles)


def parse_st_jude_xml(filepath, with_vertex_ids=False, include_voltage=False):
    import xml.etree.ElementTree as ET
    tree = ET.parse(filepath)
    root = tree.getroot()

    vertices_text = root.find(".//Vertices").text.strip()
    vertices_raw = list(map(float, vertices_text.split()))
    vertices = list(zip(vertices_raw[0::3], vertices_raw[1::3], vertices_raw[2::3]))
    if with_vertex_ids:
        vertices = [(i, x, y, z) for i, (x, y, z) in enumerate(vertices)]

    polygons_text = root.find(".//Polygons").text.strip()
    polygons_raw = list(map(int, polygons_text.split()))
    triangles = [(i1 - 1, i2 - 1, i3 - 1) for i1, i2, i3 in
                 zip(polygons_raw[0::3], polygons_raw[1::3], polygons_raw[2::3])]

    voltages = None
    if include_voltage:
        map_data_tag = root.find(".//Map_data")
        if map_data_tag is not None and map_data_tag.text:
            voltages = np.array(list(map(float, map_data_tag.text.strip().split())))
            if len(voltages) != len(vertices):
                raise ValueError("Mismatch between number of voltages and vertices")

    return np.array(vertices), np.array(triangles), voltages


def load_mesh_data(meshpath, with_vertex_ids=False, format_hint=None):
    '''
    Loads mesh data from Biosense, St. Jude, or other known formats.

    :param meshpath: Path to the mesh file (.mesh or .xml).
    :param with_vertex_ids: Whether to return vertex IDs (if available).
    :param format_hint: Optional override ("biosense", "st_jude").
    :return: (vertices, triangles) as numpy arrays.
    '''

    # Determine format
    if format_hint == "st_jude" or meshpath.endswith(".xml"):
        return parse_st_jude_xml(meshpath)

    with open(meshpath, "rb") as f:
        lines = f.read().decode("ascii", errors="ignore").splitlines()

    if format_hint == "biosense" or ("[VerticesSection]" in lines and "[TrianglesSection]" in lines):
        return parse_biosense(lines)

    raise ValueError("Could not detect mesh format. Use format_hint='biosense' or 'st_jude'.")


def load_voltage_data_from_xml(meshpath, xml_dir=None):
    '''
    :param xml_dir: String, directory where corresponding "points_export.xml" can be found
    :param meshpath: String, where mesh file is
    :return: dict: int -> (float, float, int); point id -> (unipolar voltage, bipolar voltage, impedance rate)
    '''
    voltage_map = {}

    # Get mesh file prefix (remove .mesh, add _Points_Export.xml)
    mesh_prefix = Path(meshpath).stem
    if xml_dir is None:
        xml_dir = Path(meshpath).parent

    points_filename = f"{mesh_prefix}_Points_Export.xml"
    points_export_file = os.path.join(xml_dir, points_filename)

    print(f"Using mesh file: {meshpath}")
    print(f"Using XML points export file: {points_export_file}")

    if not os.path.exists(points_export_file):
        raise FileNotFoundError(f"Points export file not found: {points_export_file}")

    # Load mapping from point IDs to filenames
    tree = ET.parse(points_export_file)
    root = tree.getroot()

    point_map = {}
    for point in root.findall(".//Point"):
        point_id = int(point.attrib["ID"])
        file_name = point.attrib.get("File_Name")
        if file_name:
            point_map[point_id] = file_name

    # Load voltages and impedance rate for each point
    for point_id, filename in point_map.items():
        file_path = os.path.join(xml_dir, filename)
        if os.path.exists(file_path):
            tree = ET.parse(file_path)
            root = tree.getroot()

            voltages = root.find(".//Voltages")
            impedance = root.find(".//Impedances")

            if voltages is not None:
                unipolar = float(voltages.attrib.get("Unipolar", 0))
                bipolar = float(voltages.attrib.get("Bipolar", 0))
            else:
                unipolar, bipolar = 0.0, 0.0

            if impedance is not None:
                rate = int(impedance.attrib.get("Rate", 0))
            else:
                rate = 0

            voltage_map[point_id] = (unipolar, bipolar, rate)

    return voltage_map


def parse_st_jude_multivolume_xml(filepath, with_vertex_ids=False):
    '''
    Parses a St. Jude XML file with multiple Volumes.

    Returns:
        A dict: {volume_name: (vertices, triangles)}
    '''
    import xml.etree.ElementTree as ET
    tree = ET.parse(filepath)
    root = tree.getroot()

    volumes = {}
    for volume in root.findall(".//Volume"):
        name = volume.get("name", "UnnamedVolume")

        # Parse vertices
        vertices_text = volume.find("Vertices").text.strip()
        vertices_raw = list(map(float, vertices_text.split()))
        vertices = list(zip(vertices_raw[0::3], vertices_raw[1::3], vertices_raw[2::3]))
        if with_vertex_ids:
            vertices = [(i, x, y, z) for i, (x, y, z) in enumerate(vertices)]

        # Parse triangles
        polygons_text = volume.find("Polygons").text.strip()
        polygons_raw = list(map(int, polygons_text.split()))
        triangles = [(i1 - 1, i2 - 1, i3 - 1) for i1, i2, i3 in zip(polygons_raw[0::3], polygons_raw[1::3], polygons_raw[2::3])]

        volumes[name] = (np.array(vertices), np.array(triangles))

    return volumes


def load_structured_mesh_with_voltage(structure_path, voltage_path):
    from collections import defaultdict
    import xml.etree.ElementTree as ET

    # Step 1: Load flat canonical mesh from voltage file
    vertices_all, triangles_all, voltages = parse_st_jude_xml(voltage_path, include_voltage=True)

    # Step 2: Load structured geometry from the structure file
    tree = ET.parse(structure_path)
    root = tree.getroot()

    meshes = {}
    vertex_map = {tuple(v): i for i, v in enumerate(vertices_all)}  # for fast lookup

    for volume in root.findall(".//Volume"):
        name = volume.get("name", "UnnamedVolume")

        # Parse this volume's vertices
        vertices_text = volume.find("Vertices").text.strip()
        v_raw = list(map(float, vertices_text.split()))
        v_list = list(zip(v_raw[0::3], v_raw[1::3], v_raw[2::3]))

        # Map to global indices (with np.allclose fallback if needed)
        idx_map = []
        for v in v_list:
            i = vertex_map.get(tuple(v))
            if i is None:
                # fallback: slow fuzzy match
                i = next((j for j, v_ref in enumerate(vertices_all) if np.allclose(v, v_ref, atol=1e-5)), None)
                if i is None:
                    raise ValueError(f"Vertex {v} not found in global vertex list.")
            idx_map.append(i)

        # Parse this volume's triangles
        poly_text = volume.find("Polygons").text.strip()
        poly_raw = list(map(int, poly_text.split()))
        polys_local = list(zip(poly_raw[0::3], poly_raw[1::3], poly_raw[2::3]))

        # Adjust to global indices (1-based → 0-based → remapped)
        triangles = []
        for t in polys_local:
            try:
                i1 = idx_map[t[0] - 1]
                i2 = idx_map[t[1] - 1]
                i3 = idx_map[t[2] - 1]
                triangles.append((i1, i2, i3))
            except IndexError:
                continue  # skip malformed

        vertices = [vertices_all[i] for i in set(idx_map)]
        local_indices = {old_i: new_i for new_i, old_i in enumerate(sorted(set(idx_map)))}
        triangles = [(local_indices[i1], local_indices[i2], local_indices[i3]) for (i1, i2, i3) in triangles]
        voltages_local = np.array([voltages[i] for i in sorted(set(idx_map))])

        m = Mesh(vertices=vertices, triangles=triangles)
        m.voltages = voltages_local
        meshes[name] = m

    return meshes


def load_voltage_colors_from_mesh(filepath):
    voltages = {}
    in_section = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if "[VerticesColorsSection]" in line:
                in_section = True
                continue

            if in_section:
                if not line or line.startswith(";"):
                    continue
                if line.startswith("[") and not "[VerticesColorsSection]" in line:
                    break

                match = re.match(r"(\d+)\s*=\s*(.*)", line)
                if match:
                    vertex_id = int(match.group(1))
                    values_str = match.group(2)
                    values = re.findall(r"[-+]?\d*\.\d+|\d+", values_str)
                    if len(values) >= 2:
                        unipolar = float(values[0])
                        bipolar = float(values[1])
                        voltages[vertex_id] = (unipolar, bipolar)

    # Summary statistics
    if voltages:
        unipolars = [v[0] for v in voltages.values() if v[0] > -1000]
        bipolars = [v[1] for v in voltages.values() if v[1] > -1000]
        print(f"Loaded {len(voltages)} voltages")
    else:
        print("No voltages found.")

    return voltages


import matplotlib.pyplot as plt

def plot_voltage_histogram(voltages, bins=50):
    """
    Plots histograms of unipolar and bipolar voltages.

    Parameters:
        voltages (dict): Dictionary of form {index: (unipolar, bipolar)}.
        bins (int): Number of histogram bins (default 50).
    """
    if not voltages:
        print("No voltage data to plot.")
        return

    unipolars = [v[0] for v in voltages.values()]
    bipolars = [v[1] for v in voltages.values()]

    plt.figure()
    plt.hist(unipolars, bins=bins, alpha=0.7, label='Unipolar')
    plt.hist(bipolars, bins=bins, alpha=0.7, label='Bipolar')
    plt.xlabel('Voltage')
    plt.ylabel('Frequency')
    plt.title('Voltage Distribution Histogram')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    #folder = "C:/Users/steph/Documents/UNC Cardiac Imaging/EAM data/ExportData29_04_25 13_58_54"
    folder = "C:/Users/steph/Documents/UNC Cardiac Imaging/EAM data/1L/Patient 2025_07_08/PVI/Export_PVI-08_19_2025-15-41-50"
    spath = "C:/Users/steph/Documents/UNC Cardiac Imaging/EAM data/Velocity_Export/f9f00471-2de7-46fa-a78a-f50415576216/2025_02_28_17_10_53/Model_Groups.xml"
    #vpath = "C:/Users/steph/Documents/UNC Cardiac Imaging/EAM data/Velocity_Export/f9f00471-2de7-46fa-a78a-f50415576216/2025_02_28_17_10_53/Contact_Mapping_Model.xml"
    ct_volume_path = "C:/Users/steph/Downloads/results_0000403E.nii.gz"
    mesh_files = find_mesh_files(folder)

    #print(mesh_files)
    file = mesh_files[2]
    print(file)
    mesh = Mesh(meshpath=file)
    mesh.initialize_voltages()
    #test_voltages = {x: (x, x) for x in mesh.voltages.keys()}
    #mesh.voltages = test_voltages
    #print(mesh.voltages)
    #plot_voltage_histogram(mesh.voltages)
    mesh.plot()
    #mesh.plot()
    '''
    #structures = load_structured_mesh_with_voltage(spath, vpath)
    #left = structures['Left']
    #left.plot()


    
    mesh = Mesh(meshpath_1)
    mesh.initalize_voltages(xml_folder_1)
    mesh.plot(voltage_type='Unipolar')
    mesh.apply_euler_transform((90,0,0))
    mesh.plot(voltage_type='Unipolar')

    la_test_image = sitk.ReadImage(meshpath_2)
    # plot_sitk_image_3d(la_test_image)
    # test_shell = sitk_binary_shell(la_test_image)
    # test_shell = PointCloud(test_shell)
    # test_shell.plot()

    #plot_3d()
    from register import sitk_binary_shell
    #vertices, triangles = load_mesh_data(meshpath_3, True)


    #plot_voltages_3d(vertices, np.array([]), meshpath_2, xml_folder_1, "Bipolar")
    #sitk_image_1 = mesh_to_sitk(vertices, triangles)
    #vertices, triangles = load_mesh_data(meshpath_2, True)
    #sitk_image_2 = mesh_to_sitk(vertices, triangles)
    #show_multiple_sitk_images_3d([sitk_image_1, sitk_image_2], colors=['red', 'blue'])
    #plot_voltages_3d_color_adjust(meshpath, xml_folder, 'Bipolar')
    '''