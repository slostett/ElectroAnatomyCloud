"""
ct_slicer.py — CT cross-section slice viewer component.

Renders a single axial, coronal, or sagittal slice of the raw CT volume as a
Plotly Heatmap figure. A Dash Slider controls which slice is shown.

The CT array is stored server-side in a diskcache.Cache (managed by callbacks.py)
and individual slices are fetched on demand via pipeline.get_ct_slice().
Only the slice index and axis selection are stored in the browser (dcc.Store).

Architecture role
-----------------
  app/app.py imports build_ct_slicer_panel() and places it below the 3D viewer.
  app/callbacks.py wires render_ct_slice() to update the slice figure whenever
  the axis radio or slice slider changes. The panel is disabled (greyed out)
  until step 6 (alignment complete).

To add a new viewing orientation
----------------------------------
1. Add the option to the axis RadioItems below.
2. The render_ct_slice() callback in callbacks.py already delegates to
   pipeline.get_ct_slice() which supports 'axial', 'coronal', 'sagittal'.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


# Plotly colorscale for CT HU values (bone/grey is standard)
_CT_COLORSCALE = 'gray'
_CT_ZMIN = -1000   # HU floor (air)
_CT_ZMAX = 1000    # HU ceiling (bone)


def build_ct_slicer_panel() -> html.Div:
    """
    Build the CT slice viewer panel.

    Returns
    -------
    html.Div
        Collapsible panel with axis selector, slice slider, and Plotly Heatmap figure.
        The panel is enabled/disabled by the 'ct-slicer-collapse' Collapse component
        which is toggled by callbacks.py when step reaches 6.

    Component IDs (used in callbacks.py)
    --------------------------------------
    - 'btn-toggle-slicer'   : Button to show/hide the slicer panel
    - 'ct-slicer-collapse'  : dbc.Collapse wrapping the slicer content
    - 'ct-axis-radio'       : RadioItems — 'axial', 'coronal', 'sagittal'
    - 'ct-slice-slider'     : Slider for slice index
    - 'ct-slice-figure'     : dcc.Graph showing the 2D Heatmap slice
    """
    return html.Div([
        dbc.Button(
            "CT Slice Viewer",
            id='btn-toggle-slicer',
            color='secondary',
            outline=True,
            size='sm',
            className='mt-3 mb-1',
            disabled=True,   # enabled by callback at step 6
        ),
        dbc.Collapse(
            html.Div([
                dbc.Row([
                    dbc.Col(
                        dbc.RadioItems(
                            id='ct-axis-radio',
                            options=[
                                {'label': 'Axial', 'value': 'axial'},
                                {'label': 'Coronal', 'value': 'coronal'},
                                {'label': 'Sagittal', 'value': 'sagittal'},
                            ],
                            value='axial',
                            inline=True,
                            className='small',
                            inputStyle={'marginRight': '4px', 'marginLeft': '8px'},
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        html.Span(id='ct-slice-label', className='text-muted small'),
                        width=6,
                        className='d-flex align-items-center justify-content-end',
                    ),
                ], className='mb-2 mt-2'),

                dcc.Slider(
                    id='ct-slice-slider',
                    min=0, max=100, step=1, value=50,
                    marks={},
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    className='mb-2',
                ),

                dcc.Graph(
                    id='ct-slice-figure',
                    figure=_empty_slice_figure(),
                    config={'displayModeBar': False},
                    style={'height': '320px'},
                ),
            ], className='p-2 border rounded bg-dark'),
            id='ct-slicer-collapse',
            is_open=False,
        ),
    ])


def _empty_slice_figure() -> go.Figure:
    """Placeholder figure shown before CT data is loaded."""
    fig = go.Figure()
    fig.add_annotation(
        text="CT slice viewer — available after alignment is complete",
        xref='paper', yref='paper',
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=12, color='#888888'),
    )
    fig.update_layout(
        paper_bgcolor='#111',
        plot_bgcolor='#111',
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=300,
    )
    return fig


def build_slice_figure(slice_array, axis: str, index: int) -> go.Figure:
    """
    Build a Plotly Heatmap figure for a single CT slice.

    Parameters
    ----------
    slice_array : np.ndarray
        2D array of HU values for the slice.
    axis : str
        Orientation label ('axial', 'coronal', or 'sagittal'). Used in the title.
    index : int
        Slice index along the chosen axis. Used in the title.

    Returns
    -------
    go.Figure
        Plotly figure with a single Heatmap trace.

    Notes
    -----
    zmin/zmax are fixed at -1000/+1000 HU for consistent window/level across slices.
    For cardiac CT the soft-tissue window (-200 to +300 HU) can be toggled in future
    by exposing zmin/zmax as parameters.
    """
    import numpy as np
    arr = np.asarray(slice_array)

    fig = go.Figure(go.Heatmap(
        z=arr,
        colorscale=_CT_COLORSCALE,
        zmin=_CT_ZMIN,
        zmax=_CT_ZMAX,
        showscale=False,
        hovertemplate='HU: %{z}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(
            text=f"{axis.capitalize()} slice {index}",
            font=dict(size=12, color='#aaaaaa'),
        ),
        paper_bgcolor='#111',
        plot_bgcolor='#111',
        margin=dict(l=0, r=0, t=28, b=0),
        xaxis=dict(visible=False, scaleanchor='y'),
        yaxis=dict(visible=False, autorange='reversed'),
        height=300,
    )
    return fig
