import numpy as np
from skimage import measure
import trimesh
import SimpleITK as sitk
import plotly.graph_objects as go


def plot_clusters_plotly(points: np.ndarray, labels: np.ndarray):
    """
    Plot 3D clustered point cloud using Plotly for interactive visualization.

    Parameters:
        points (np.ndarray): Nx3 array of 3D coordinates.
        labels (np.ndarray): Array of cluster labels (length N).
    """
    fig = go.Figure()

    unique_labels = np.unique(labels)
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_points = points[mask]
        fig.add_trace(go.Scatter3d(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            z=cluster_points[:, 2],
            mode='markers',
            marker=dict(size=3),
            name=f'Cluster {cluster_id}'
        ))

    fig.update_layout(
        title="KMeans Clustering of Point Cloud",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        legend_title="Clusters",
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()


def plot_shell_point_cloud(points, point_size=2):
    """
    Plots a 3D point cloud using Plotly.

    Args:
        points (numpy.ndarray): Nx3 array of 3D coordinates.
        point_size (int): Size of each point in the plot.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=point_size,
                color='blue',  # color by depth (optional)
                opacity=0.8
            )
        )
    ])

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title='Shell Point Cloud',
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()


def plot_long_axes(vertices, image_vertices, pc_center=None, img_center=None, scale=30):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    def get_long_axis(points: np.ndarray):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca.fit(points - points.mean(axis=0))
        return pca.components_[0]

    pc_axis = get_long_axis(vertices)
    img_axis = get_long_axis(image_vertices)

    if pc_center is None:
        pc_center = np.mean(vertices, axis=0)
    if img_center is None:
        img_center = np.mean(image_vertices, axis=0)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot point clouds
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='b', s=1, alpha=0.3, label='Point Cloud')
    ax.scatter(image_vertices[:, 0], image_vertices[:, 1], image_vertices[:, 2], color='g', s=1, alpha=0.3, label='Image Mesh')

    # Plot long axes as lines (both directions from center)
    pc_line = np.vstack([pc_center - scale * pc_axis, pc_center + scale * pc_axis])
    img_line = np.vstack([img_center - scale * img_axis, img_center + scale * img_axis])
    ax.plot(pc_line[:, 0], pc_line[:, 1], pc_line[:, 2], color='r', linewidth=2, label='PC Axis')
    ax.plot(img_line[:, 0], img_line[:, 1], img_line[:, 2], color='m', linewidth=2, label='Image Axis')

    # Optional: add arrowheads for direction
    ax.quiver(*pc_center, *pc_axis, length=scale, color='r', arrow_length_ratio=0.1)
    ax.quiver(*img_center, *img_axis, length=scale, color='m', arrow_length_ratio=0.1)

    ax.legend()
    ax.set_title("Point Cloud and Image Long Axes")
    ax.set_box_aspect([1, 1, 1])  # equal aspect ratio
    plt.tight_layout()
    plt.show()


def plot_sitk_image_3d(image: sitk.Image, level: float = 0.5, name: str = "Mask", color: str = "blue", opacity: float = 0.5):
    """
    Converts a sitk.Image into a surface mesh and plots it in 3D using Plotly.

    Parameters:
        image (sitk.Image): The binary or labeled image to visualize.
        level (float): The threshold level used for surface extraction.
        name (str): Legend label.
        color (str): Color of the mesh.
        opacity (float): Mesh opacity.
    """
    volume = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()  # (x, y, z) in physical units
    origin = image.GetOrigin()
    direction = np.array(image.GetDirection()).reshape(3, 3)
    from skimage.measure import marching_cubes
    # Marching cubes returns vertices in voxel index space (z, y, x) with spacing applied
    verts, faces, _, _ = marching_cubes(volume, level=level, spacing=spacing[::-1])  # spacing is (z, y, x)

    # Convert to physical coordinates: (z, y, x) → (x, y, z)
    verts = verts[:, [2, 1, 0]]  # to (x, y, z)
    verts = (verts @ direction.T) + origin  # apply direction and origin

    x, y, z = verts.T
    i, j, k = faces.T

    fig = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=color,
            name=name,
            opacity=opacity,
            flatshading=True
        )
    ])

    fig.update_layout(
        title=f"3D View: {name}",
        scene=dict(aspectmode="data"),
        showlegend=True
    )
    fig.show()


def show_multiple_sitk_images_3d(images, labels=None, colors=None, spacing_override=None, opacity=0.4):
    """
    Display multiple sitk.Image volumes as overlaid 3D surfaces using plotly.

    Args:
        images (list): List of SimpleITK images.
        labels (list): Optional list of labels for the plot legend.
        colors (list): Optional list of colors (e.g., ['red', 'blue']).
        spacing_override (float): If specified, overrides pitch for marching cubes.
        opacity (float): Opacity of each mesh.
    """
    fig = go.Figure()

    for idx, image in enumerate(images):
        label = labels[idx] if labels else f"Image {idx+1}"
        color = colors[idx] if colors else None

        # Convert image to numpy
        volume = sitk.GetArrayFromImage(image)  # (z, y, x)
        spacing = image.GetSpacing()[::-1] if spacing_override is None else (spacing_override,) * 3
        origin = image.GetOrigin()[::-1]

        # Generate mesh with marching cubes via trimesh
        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(volume, pitch=spacing[0])
        mesh.apply_translation(origin)

        verts = mesh.vertices
        faces = mesh.faces

        fig.add_trace(go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name=label,
            opacity=opacity,
            color=color,
            flatshading=True
        ))

    fig.update_layout(
        title="3D Overlay of Multiple Volumes",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        showlegend=True
    )

    fig.show()


def plot_sitk_images(struct1, struct2, name1="Structure 1", name2="Structure 2",
                                     color1="red", color2="blue", opacity1=0.5, opacity2=0.5, level=0.5):
    def sitk_to_mesh(sitk_image, level):
        """Convert a SimpleITK image to a mesh using marching cubes."""
        array = sitk.GetArrayFromImage(sitk_image)  # shape: (z, y, x)
        spacing = sitk_image.GetSpacing()
        origin = sitk_image.GetOrigin()
        direction = np.array(sitk_image.GetDirection()).reshape(3, 3)

        verts, faces, _, _ = measure.marching_cubes(array, level=level, spacing=spacing)
        verts = (verts @ direction.T) + origin
        return verts, faces

    verts1, faces1 = sitk_to_mesh(struct1, level)
    verts2, faces2 = sitk_to_mesh(struct2, level)

    mesh1 = go.Mesh3d(
        x=verts1[:, 0], y=verts1[:, 1], z=verts1[:, 2],
        i=faces1[:, 0], j=faces1[:, 1], k=faces1[:, 2],
        color=color1, opacity=opacity1, name=name1, showscale=False
    )

    mesh2 = go.Mesh3d(
        x=verts2[:, 0], y=verts2[:, 1], z=verts2[:, 2],
        i=faces2[:, 0], j=faces2[:, 1], k=faces2[:, 2],
        color=color2, opacity=opacity2, name=name2, showscale=False
    )

    fig = go.Figure(data=[mesh1, mesh2])
    fig.update_layout(
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data"
        ),
        title="3D Visualization of Two Structures"
    )
    fig.show()


def plot_meshes_3d(meshes, names=None, colors=None, opacity=0.7):
    '''
    Plots a graph of trimesh objects passed in
    :param meshes: List of trimesh objects
    :param names: list of strings of names to graph
    :param colors: list of strings to colors
    :param opacity: self explanatory
    :return: None
    '''
    fig = go.Figure()
    for i, mesh in enumerate(meshes):
        name = names[i] if names else f"Mesh {i+1}"
        color = colors[i] if colors else None
        fig.add_trace(go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            name=name,
            color=color,
            opacity=opacity,
            flatshading=True
        ))
    fig.update_layout(
        title="ICP Registration of CT Segmentation and Mesh",
        scene=dict(aspectmode='data'),
        showlegend=True
    )
    fig.show()


def plot_voltages_3d(vertices, triangles, in_voltages, voltage_type='Bipolar'):

    # Extract voltage values (index 0 for unipolar, 1 for bipolar)
    voltages = np.array([
        in_voltages.get(v[0], (0, 0))[1 if voltage_type == "Bipolar" else 0]
        for v in vertices
    ])
    colors = (voltages - np.min(voltages)) / (np.max(voltages) - np.min(voltages))

    fig = go.Figure(
        data=[go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
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


def plot_voltages_3d_color_adjust(vertices, triangles, in_voltages, voltage_type='Bipolar'):

    # Build voltage array using vertex IDs
    voltages_raw = []
    for v in vertices:
        v_id = int(v[0])
        voltage_tuple = in_voltages.get(v_id, (np.nan, np.nan))
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
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
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