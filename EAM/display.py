import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import re
import xml.etree.ElementTree as ET
import os
from pathlib import Path

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
    xml_folder_2 = 'C:/users/steph/Documents/UNC Cardiac Imaging/ExportData28_02_25 16_10_53/Patient 2025_02_28/AF/Export_AF-02_28_2025-16-01-43/'



    #plot_3d()
    vertices, triangles = load_mesh_data(meshpath_1, True)
    from graph import plot_voltages_3d
    plot_voltages_3d(meshpath_1, xml_folder_1, "Bipolar")
    sitk_image_1 = mesh_to_sitk(vertices, triangles)
    vertices, triangles = load_mesh_data(meshpath_2, True)
    sitk_image_2 = mesh_to_sitk(vertices, triangles)
    #show_multiple_sitk_images_3d([sitk_image_1, sitk_image_2], colors=['red', 'blue'])
    #plot_voltages_3d_color_adjust(meshpath, xml_folder, 'Bipolar')