"""
file_inputs.py — File path input panel component.

Renders three text inputs (mesh path, segmentation path, CT path) and a status
badge per field (green = file found, red = not found). Also provides a Reset button
that clears the pipeline store.

Architecture role
-----------------
  app/app.py imports build_file_inputs_panel() and places it in the sidebar.
  app/callbacks.py wires validate_and_store_files() to update the status badges
  and store validated paths in dcc.Store.

To add a new input field
-------------------------
1. Add an html.Div with dcc.Input + status badge below, following the existing pattern.
2. Add the new Input to the validate_and_store_files() callback in callbacks.py.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc


def _input_row(label: str, input_id: str, placeholder: str, status_id: str) -> dbc.Row:
    """Helper: one labelled text input with a status badge on the right."""
    return dbc.Row([
        dbc.Col(dbc.Label(label, className="fw-semibold"), width=12),
        dbc.Col(
            dcc.Input(
                id=input_id,
                type='text',
                placeholder=placeholder,
                debounce=True,          # validate on blur / Enter, not every keystroke
                className='form-control form-control-sm font-monospace',
                style={'fontSize': '0.78rem'},
            ),
            width=10,
        ),
        dbc.Col(
            html.Span('?', id=status_id, className='badge bg-secondary ms-1'),
            width=2,
            className='d-flex align-items-end pb-1',
        ),
    ], className='mb-2')


def build_file_inputs_panel() -> html.Div:
    """
    Build the file path input panel.

    Returns
    -------
    html.Div
        A Dash layout div containing three path inputs (mesh, seg, CT) and a Reset button.
        Status badges update via the validate_and_store_files callback in callbacks.py.

    Input IDs (used in callbacks.py)
    ----------------------------------
    - 'input-mesh-path'   : EAM .mesh file path
    - 'input-seg-path'    : Segmentation .nii.gz path
    - 'input-ct-path'     : Raw CT .nii.gz path
    - 'status-mesh'       : Status badge for mesh
    - 'status-seg'        : Status badge for seg
    - 'status-ct'         : Status badge for CT
    - 'btn-reset'         : Reset pipeline button
    """
    return html.Div([
        html.H6("Input Files", className='text-uppercase text-muted mb-3 mt-1 fw-bold'),

        _input_row(
            label="EAM Mesh (.mesh)",
            input_id='input-mesh-path',
            placeholder='/data/mesh/patient.mesh',
            status_id='status-mesh',
        ),
        _input_row(
            label="Segmentation (.nii.gz)",
            input_id='input-seg-path',
            placeholder='/data/seg/label_map.nii.gz',
            status_id='status-seg',
        ),
        _input_row(
            label="Raw CT (.nii.gz)",
            input_id='input-ct-path',
            placeholder='/data/ct/raw_ct.nii.gz',
            status_id='status-ct',
        ),

        dbc.Row(
            dbc.Col(
                dbc.Button(
                    "Reset Pipeline",
                    id='btn-reset',
                    color='danger',
                    outline=True,
                    size='sm',
                    className='mt-2 w-100',
                ),
            ),
        ),
    ], className='p-3 border rounded bg-light mb-3')
