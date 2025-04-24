import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import re
import xml.etree.ElementTree as ET
import os
from pathlib import Path
from trimesh import Trimesh
from graph import *
import SimpleITK as sitk
from sklearn.cluster import KMeans


class PointCloud:
    def __init__(self, vertices, labels=None):
        self.vertices = np.array(vertices)
        self.labels = labels

    def plot(self):
        plot_shell_point_cloud(self.vertices)


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
        return labels

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



class Mesh(PointCloud):
    def __init__(self, vertices=None, triangles=None, meshpath=None):
        '''
        Mesh can either be manually loaded in or be immediately passed a path name via the meshpath
        keyword argument to load quickly.
        :param vertices: iterable
        :param triangles: iterable
        :param meshpath: string path to file if quickload
        '''
        PointCloud.__init__(self, vertices)
        self.meshpath = meshpath
        if meshpath is not None:
            vertices, triangles = load_mesh_data(meshpath)
            self.vertices, self.triangles = np.array(vertices), np.array(triangles)
        else:
            self.vertices, self.triangles = np.array(vertices), np.array(triangles)
        self.voltages = None


    def initalize_voltages(self, xml_dir):
        self.voltages = load_voltage_data_from_xml(xml_dir, self.meshpath)

    def plot(self, voltage_type='Bipolar'):
        if self.voltages is None:
            print('Plotting without voltages. If you have voltages, call .initialize_voltages first')
            plot_meshes_3d([Trimesh(vertices=self.vertices, faces=self.triangles)])
        if self.voltages is not None:
            print('Plotting with voltages.')
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
                except RuntimeError:
                    continue

        print(f"Set {count} voxels to 1 from {len(self.triangles)} triangles.")

        return image




def load_mesh_data(meshpath, with_vertex_ids=False):
    '''
    :param meshpath: string, where .mesh file is located
    :param with_vertex_ids: If true, adds vertex id as first element in row.
    :return: np arrays of vertices (with or without ids) and triangles.
    '''

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

def load_voltage_data_from_xml(xml_dir, meshpath):
    '''
    :param xml_dir: String, directory where corresponding "points_export.xml" can be found
    :param meshpath: String, where mesh file is
    :return: dict: int -> (int, int); point id -> (unipolar voltage, bipolar voltage)
    '''
    voltage_map = {}

    # Get mesh file prefix (remove .mesh, add _Points_Export.xml)
    mesh_prefix = Path(meshpath).stem
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

    # Load voltages for each point
    for point_id, filename in point_map.items():
        file_path = os.path.join(xml_dir, filename)
        if os.path.exists(file_path):
            tree = ET.parse(file_path)
            root = tree.getroot()
            voltages = root.find(".//Voltages")
            if voltages is not None:
                unipolar = float(voltages.attrib.get("Unipolar", 0))
                bipolar = float(voltages.attrib.get("Bipolar", 0))
                voltage_map[point_id] = (unipolar, bipolar)

    return voltage_map


if __name__ == '__main__':
    meshpath_1 = 'C:/Users/steph/Documents/UNC Cardiac Imaging/EAM data/ExportData28_02_25 16_19_56/Patient 2025_02_28/AF/Export_AF-02_28_2025-16-01-43/6-1-sinus.mesh'
    xml_folder_1 = 'C:/Users/steph/Documents/UNC Cardiac Imaging/EAM data/ExportData28_02_25 16_10_53/Patient 2025_02_28/AF/Export_AF-02_28_2025-16-01-43/'
    meshpath_2 = 'C:/Users/steph/Documents/UNC Cardiac Imaging/EAM data/ExportData28_02_25 16_10_53/Patient 2025_02_28/AF/Export_AF-02_28_2025-16-01-43/6-LA fam.mesh'
    xml_folder_2 = 'C:/Users/steph/Documents/UNC Cardiac Imaging/EAM data/ExportData28_02_25 16_10_53/Patient 2025_02_28/AF/Export_AF-02_28_2025-16-01-43/'
    meshpath_3 = "C:/Users/steph/Downloads/Atrium_L.nii.gz"

    mesh = Mesh(meshpath_1)
    mesh.initalize_voltages(xml_folder_1)
    mesh.plot(voltage_type='Unipolar')

    #plot_3d()
    from register import sitk_binary_shell
    #vertices, triangles = load_mesh_data(meshpath_3, True)


    #plot_voltages_3d(vertices, np.array([]), meshpath_2, xml_folder_1, "Bipolar")
    #sitk_image_1 = mesh_to_sitk(vertices, triangles)
    #vertices, triangles = load_mesh_data(meshpath_2, True)
    #sitk_image_2 = mesh_to_sitk(vertices, triangles)
    #show_multiple_sitk_images_3d([sitk_image_1, sitk_image_2], colors=['red', 'blue'])
    #plot_voltages_3d_color_adjust(meshpath, xml_folder, 'Bipolar')