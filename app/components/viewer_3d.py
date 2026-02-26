"""
viewer_3d.py — 3D Plotly figure builder for the alignment viewer.

Builds a go.Figure containing two traces:
  1. CT shell — rendered as a grey semi-transparent point cloud (Scatter3d).
  2. EAM mesh — rendered as a solid Mesh3d. If the Mesh object has voltages,
     the mesh is coloured by bipolar voltage intensity using the jet colorscale.

This module is purely presentational: it takes numpy arrays and returns a
go.Figure. It has no side effects and no Dash dependencies.

Architecture role
-----------------
  app/callbacks.py calls build_alignment_figure() whenever the pipeline store
  is updated (after each step) and writes the result to the 3D viewer component.

To add a new trace type
------------------------
1. Add a new branch in build_alignment_figure() checking for the new data key
   in the store dict.
2. Append a new go.Trace to `traces` before returning.
"""

import numpy as np
import plotly.graph_objects as go


# ── Color + opacity constants ──────────────────────────────────────────────────
_CT_COLOR = '#888888'
_CT_OPACITY = 0.25
_CT_POINT_SIZE = 1.5

_MESH_COLOR = '#e74c3c'      # red when no voltages
_MESH_OPACITY = 0.85

_VOLTAGE_COLORSCALE = 'jet'


def build_alignment_figure(
    ct_vertices: np.ndarray | None = None,
    mesh_vertices: np.ndarray | None = None,
    mesh_triangles: np.ndarray | None = None,
    voltages: list | None = None,
    title: str = "EAM–CT Alignment",
) -> go.Figure:
    """
    Build the 3D alignment viewer figure.

    Parameters
    ----------
    ct_vertices : np.ndarray or None
        CT shell point cloud (Nx3, mm). If None, no CT trace is added.
    mesh_vertices : np.ndarray or None
        EAM mesh vertices (Mx3, mm). If None, no mesh trace is added.
    mesh_triangles : np.ndarray or None
        EAM mesh triangle indices (Tx3). Required if mesh_vertices is provided.
    voltages : list or None
        Per-vertex bipolar voltage values (length M). If provided, the mesh is
        coloured by voltage intensity using the jet colorscale.
    title : str
        Figure title displayed above the 3D scene.

    Returns
    -------
    go.Figure
        Plotly figure with up to two traces (CT shell + EAM mesh).

    Notes
    -----
    - CT shell is rendered as Scatter3d (points) for performance: marching cubes
      on 137K-point shells is slow in the browser. Switch to Mesh3d here if
      visual fidelity is more important than interactivity speed.
    - The figure uses aspectmode='data' so the anatomical proportions are preserved.
    """
    traces = []

    # ── CT shell trace ─────────────────────────────────────────────────────────
    if ct_vertices is not None and len(ct_vertices) > 0:
        ct_arr = np.asarray(ct_vertices)
        traces.append(go.Scatter3d(
            x=ct_arr[:, 0],
            y=ct_arr[:, 1],
            z=ct_arr[:, 2],
            mode='markers',
            marker=dict(size=_CT_POINT_SIZE, color=_CT_COLOR, opacity=_CT_OPACITY),
            name='CT Shell',
            hoverinfo='skip',
        ))

    # ── EAM mesh trace ─────────────────────────────────────────────────────────
    if mesh_vertices is not None and mesh_triangles is not None:
        mv = np.asarray(mesh_vertices)
        mt = np.asarray(mesh_triangles)

        if voltages is not None and len(voltages) == len(mv):
            traces.append(go.Mesh3d(
                x=mv[:, 0], y=mv[:, 1], z=mv[:, 2],
                i=mt[:, 0], j=mt[:, 1], k=mt[:, 2],
                intensity=voltages,
                colorscale=_VOLTAGE_COLORSCALE,
                showscale=True,
                colorbar=dict(title='Bipolar (mV)', thickness=15, len=0.6),
                opacity=_MESH_OPACITY,
                name='EAM Mesh',
                flatshading=True,
            ))
        else:
            traces.append(go.Mesh3d(
                x=mv[:, 0], y=mv[:, 1], z=mv[:, 2],
                i=mt[:, 0], j=mt[:, 1], k=mt[:, 2],
                color=_MESH_COLOR,
                opacity=_MESH_OPACITY,
                name='EAM Mesh',
                flatshading=True,
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            bgcolor='#1a1a2e',
            xaxis=dict(showbackground=False, gridcolor='#333'),
            yaxis=dict(showbackground=False, gridcolor='#333'),
            zaxis=dict(showbackground=False, gridcolor='#333'),
        ),
        paper_bgcolor='#0d0d1a',
        plot_bgcolor='#0d0d1a',
        font=dict(color='#cccccc'),
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.4)', font=dict(size=11)),
        margin=dict(l=0, r=0, t=30, b=0),
        uirevision='alignment-viewer',  # preserves camera angle across figure updates
    )
    return fig


def build_empty_figure(message: str = "Load files and click 'Run Next Step' to begin") -> go.Figure:
    """
    Return a placeholder figure shown before any data is loaded.

    Parameters
    ----------
    message : str
        Text shown in the center of the empty plot area.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref='paper', yref='paper',
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color='#888888'),
    )
    fig.update_layout(
        paper_bgcolor='#0d0d1a',
        plot_bgcolor='#0d0d1a',
        scene=dict(bgcolor='#1a1a2e'),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
