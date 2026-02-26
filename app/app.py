"""
app.py — Dash application entry point for the ElectroAnatomyCloud alignment tool.

This module defines the Dash app object and the page layout. It contains NO
business logic — all callback logic lives in callbacks.py and all computation
lives in pipeline.py.

Running locally (bare Python, no Docker)
-----------------------------------------
    cd ElectroAnatomyCloud
    python app/app.py
    # Open http://localhost:8050 in your browser

Running in Docker
-----------------
    docker-compose up --build
    # Open http://localhost:8050 in your browser

Architecture
------------
    app.py        — Layout + server
    callbacks.py  — All @app.callback definitions
    pipeline.py   — Stateless backend wrappers over EAM/display.py and EAM/register.py
    components/   — Individual UI panel modules

See DEVELOPER.md for a full architecture diagram and extension guide.
"""

import sys
import os

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

# ── Import UI components ───────────────────────────────────────────────────────
_APP_DIR = os.path.dirname(__file__)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from components.file_inputs import build_file_inputs_panel
from components.param_controls import build_param_controls_panel
from components.viewer_3d import build_empty_figure
from components.ct_slicer import build_ct_slicer_panel

# ── App object ─────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],   # dark Bootstrap theme
    title='ElectroAnatomyCloud — EAM Alignment',
    suppress_callback_exceptions=True,
)
server = app.server   # expose Flask server for production WSGI deployment

# ── Layout ─────────────────────────────────────────────────────────────────────
#
# Page structure:
#   Header (title + step breadcrumb)
#   ├── Sidebar (file inputs + parameter controls)
#   └── Main area
#       ├── Control row (Run Next Step button + status)
#       ├── 3D viewer (dcc.Graph)
#       ├── Metrics card (shown at step 6)
#       └── CT slicer panel (collapsible, enabled at step 6)

def _header() -> dbc.Navbar:
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand(
                "ElectroAnatomyCloud — EAM / CT Alignment",
                className='fw-bold text-white',
            ),
            html.Div(
                id='step-breadcrumb',
                className='text-muted small ms-4',
                children="Step 0 of 6 — Load files to begin",
            ),
        ], fluid=True),
        color='dark',
        dark=True,
        className='mb-3',
    )


def _sidebar() -> html.Div:
    return html.Div([
        build_file_inputs_panel(),
        build_param_controls_panel(),
    ], style={'width': '320px', 'minWidth': '280px', 'flexShrink': '0'})


def _main_area() -> html.Div:
    return html.Div([
        # Control row
        dbc.Row([
            dbc.Col(
                dbc.Button(
                    "Run Next Step",
                    id='btn-next-step',
                    color='primary',
                    size='lg',
                    className='me-3',
                    disabled=True,      # enabled by callback when files are valid
                ),
                width='auto',
            ),
            dbc.Col(
                html.Div(id='step-status', className='text-muted align-self-center'),
                width='auto',
            ),
        ], className='mb-3 align-items-center'),

        # 3D viewer
        dcc.Loading(
            id='loading-viewer',
            type='circle',
            color='#3498db',
            children=dcc.Graph(
                id='viewer-3d',
                figure=build_empty_figure(),
                config={
                    'displayModeBar': True,
                    'modeBarButtonsToRemove': ['toImage'],
                    'displaylogo': False,
                },
                style={'height': '560px'},
            ),
        ),

        # Metrics card (hidden until step 6)
        html.Div(id='metrics-card', className='mt-3'),

        # CT slicer (collapsed until step 6)
        build_ct_slicer_panel(),

    ], style={'flex': '1', 'minWidth': '0'})


app.layout = html.Div([
    # Session state store — JSON-serializable dict, persists across callbacks in a session
    dcc.Store(id='pipeline-store', storage_type='session', data={
        'step': 0,
        'mesh_path': None,
        'seg_path': None,
        'ct_path': None,
        'params': {},
        'mesh_vertices': None,
        'mesh_triangles': None,
        'ct_vertices': None,
        'metrics': None,
        'ct_meta': None,
        'voltages': None,
    }),

    # Polling interval for background region-grow progress (fires every 30s while active)
    dcc.Interval(id='grow-poll-interval', interval=30000, n_intervals=0, disabled=True),

    _header(),

    dbc.Container([
        html.Div([
            _sidebar(),
            html.Div(style={'width': '24px', 'flexShrink': '0'}),  # gutter
            _main_area(),
        ], style={'display': 'flex', 'alignItems': 'flex-start'}),
    ], fluid=True),
], style={'backgroundColor': '#0d0d1a', 'minHeight': '100vh'})

# ── Import and register callbacks ──────────────────────────────────────────────
# Importing callbacks.py has the side-effect of registering all @app.callback
# decorators. This must happen after `app` is defined.
import callbacks   # noqa: E402 (import after definition is intentional)

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=False)
