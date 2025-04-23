import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import re
import plotly.graph_objects as go
import xml.etree.ElementTree as ET
import os
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt
import trimesh

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


def plot_2d(meshpath):
    vertices, triangles = load_mesh_data(meshpath)

    print('Read', len(vertices), 'vertices in', meshpath)
    print('Read', len(triangles), 'triangles in', meshpath)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.add_collection3d(Poly3DCollection(vertices[triangles], alpha=0.3, edgecolor='k'))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])

    plt.show()


def plot_3d(meshpath):
    vertices, triangles = load_mesh_data(meshpath)

    fig = go.Figure(
        data=[go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            color='lightblue',
            opacity=0.5
        )]
    )

    fig.update_layout(
        title="Interactive 3D Mesh",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z")
    )

    fig.show()


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


def plot_voltages_3d(meshpath, xml_dir, voltage_type='Bipolar'):
    vertices, triangles = load_mesh_data(meshpath, with_vertex_ids=True)
    vertex_voltage = load_voltage_data_from_xml(xml_dir, meshpath)

    # Extract voltage values (index 0 for unipolar, 1 for bipolar)
    voltages = np.array([
        vertex_voltage.get(v[0], (0, 0))[1 if voltage_type == "Bipolar" else 0]
        for v in vertices
    ])
    colors = (voltages - np.min(voltages)) / (np.max(voltages) - np.min(voltages))

    fig = go.Figure(
        data=[go.Mesh3d(
            x=vertices[:, 1],
            y=vertices[:, 2],
            z=vertices[:, 3],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            intensity=voltages,
            colorscale='jet',
            showscale=True,
            colorbar=dict(title=f"{voltage_type} Voltage (mV)"),
            opacity=1
        )]
    )

    fig.update_layout(
        title=f"3D Mesh Colored by {voltage_type} Voltage",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z")
    )

    fig.show()


def plot_voltages_3d_color_adjust(meshpath, xml_dir, voltage_type='Bipolar'):
    vertices, triangles = load_mesh_data(meshpath, with_vertex_ids=True)
    vertex_voltage = load_voltage_data_from_xml(xml_dir, meshpath)

    # Build voltage array using vertex IDs
    voltages_raw = []
    for v in vertices:
        v_id = int(v[0])
        voltage_tuple = vertex_voltage.get(v_id, (np.nan, np.nan))
        value = voltage_tuple[1 if voltage_type == 'Bipolar' else 0]
        voltages_raw.append(value)

    voltages = np.array(voltages_raw)

    # Normalize color intensity only for 0–2 mV range
    clip_min, clip_max = 0.0, 2.0
    clipped = np.clip(voltages, clip_min, clip_max)
    norm_intensity = (clipped - clip_min) / (clip_max - clip_min)

    # Create a masked array for color application
    nan_mask = np.isnan(voltages)

    # Custom color scale emphasizing detail in 0–2 mV
    colorscale = [
        [0.00, "rgb(169,169,169)"],     # Gray for missing
        [0.001, "rgb(0, 0, 255)"],      # Blue at 0 mV
        [0.25,  "rgb(0, 255, 255)"],    # Cyan
        [0.5,   "rgb(0, 255, 0)"],      # Green
        [0.75,  "rgb(255, 255, 0)"],    # Yellow
        [1.0,   "rgb(255, 0, 0)"],      # Red at 2 mV
    ]

    # Fill in NaNs with gray voltage so colorbar remains consistent
    voltages_filled = voltages.copy()
    voltages_filled[nan_mask] = -1  # Put gray below visible range

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 1],
                y=vertices[:, 2],
                z=vertices[:, 3],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                intensity=voltages_filled,
                colorscale=colorscale,
                showscale=True,
                cmin=0,
                cmax=2,
                colorbar=dict(title=f"{voltage_type} Voltage (mV)"),
                opacity=1
            )
        ]
    )

    fig.update_layout(
        title=f"3D Mesh Colored by {voltage_type} Voltage (Detail 0–2 mV)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        )
    )

    fig.show()


def mesh_to_sitk(vertices: np.ndarray, triangles: np.ndarray, spacing=(1.0, 1.0, 1.0), padding=5) -> sitk.Image:
    """
    Convert a triangle mesh to a SimpleITK binary image.
    """
    # Remove vertex ID column if present
    if vertices.shape[1] == 4:
        coords = vertices[:, 1:]
    else:
        coords = vertices

    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=coords, faces=triangles, process=False)

    # Create a voxel grid using Trimesh's built-in voxelization
    voxelized = mesh.voxelized(pitch=spacing[0])  # assume isotropic spacing
    filled = voxelized.fill()  # fill interior

    # Convert to dense numpy volume
    volume = filled.matrix.astype(np.uint8)  # (z, y, x)

    # Convert to sitk.Image
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((spacing[2], spacing[1], spacing[0]))  # z, y, x

    # Correct origin
    origin = filled.transform[:3, 3][::-1]  # x,y,z → z,y,x
    image.SetOrigin(origin)

    return image


if __name__ == '__main__':
    meshpath_1 = 'C:/Users/steph/Documents/UNC Cardiac Imaging/EAM data/ExportData28_02_25 16_19_56/Patient 2025_02_28/AF/Export_AF-02_28_2025-16-01-43/6-1-sinus.mesh'
    xml_folder_1 = 'C:/Users/steph/Documents/UNC Cardiac Imaging/EAM data/ExportData28_02_25 16_10_53/Patient 2025_02_28/AF/Export_AF-02_28_2025-16-01-43/'
    meshpath_2 = 'C:/Users/steph/Documents/UNC Cardiac Imaging/EAM data/ExportData28_02_25 16_10_53/Patient 2025_02_28/AF/Export_AF-02_28_2025-16-01-43/6-LA fam.mesh'
    xml_folder_2 = 'C:/users/steph/Documents/UNC Cardiac Imaging/ExportData28_02_25 16_10_53/Patient 2025_02_28/AF/Export_AF-02_28_2025-16-01-43/'



    #plot_3d()
    vertices, triangles = load_mesh_data(meshpath_1, True)
    plot_voltages_3d(meshpath_1, xml_folder_1, "Bipolar")
    sitk_image_1 = mesh_to_sitk(vertices, triangles)
    vertices, triangles = load_mesh_data(meshpath_2, True)
    sitk_image_2 = mesh_to_sitk(vertices, triangles)
    #show_multiple_sitk_images_3d([sitk_image_1, sitk_image_2], colors=['red', 'blue'])
    #plot_voltages_3d_color_adjust(meshpath, xml_folder, 'Bipolar')
