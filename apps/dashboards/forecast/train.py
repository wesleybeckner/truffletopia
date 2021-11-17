import plotly.express as px
from dash import dcc, html, Dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import os
import dash

### FIGURES ###
from .plotlyfunctions import *

# Build App
def init_dashboard(server):

    dash_app = Dash(
        server=server,
        routes_pathname_prefix='/train/',
        external_stylesheets=external_stylesheets,
        title = "🍫 Truffletopia 🍭",
    )

    ### LAYOUT ###
    CONTROLS = html.Div([
        html.H2("Visualization Controls"),
        html.P("Choose Data File", style={'margin-top': '15px'}),
        dcc.Dropdown(
        multi=False,
        id="file",
        style={'color': colors['text']},
        options=[{'label': i, 'value': i} for i in f],
        value=f[0],
        ),
        html.P("Primary Groupby", style={'margin-top': '15px'}),
        dcc.Dropdown(
        multi=False,
        id="groupby1",
        style={'color': colors['text']},
        ),
        html.P("Secondary Groupby", style={'margin-top': '15px'}),
        dcc.Dropdown(
        multi=False,
        id="groupby2",
        style={'color': colors['text']},
        ),
        html.Div([
            dbc.Button('Refresh', id='reset', n_clicks=0)
        ], style={'margin-top': '15px',
                  'margin-bottom': '15px'}),
        html.Div(id='table'),
    ])

    dash_app.layout = html.Div([
        NAVBAR,
        html.Div([
            html.Div([
                CONTROLS,   
            ],
            className = "four columns",
            ),
            html.Div([
                dcc.Store(id='df'),
                html.Div(id='figures'),
                html.Div(id='dummydiv'),
                ],
                className = "eight columns",
                ),
            ], 
            className = "twelve columns",
            style = {'padding': '10px'}
            ),
    ],
    style={'backgroundColor': colors['background']}
    )

    init_callbacks(dash_app)

    return dash_app.server

### MAIN CALLBACKS ###
def init_callbacks(app):

    @app.callback(
        Output('table', 'children'),
        Output('figures', 'children'),
        Output('df', 'data'),
        Input('file', 'value'))
    def view_data(name):
        if 'csv' in name:
            df = pd.read_csv('data/forecast/{}'.format(name))
        elif 'xlsx' in name:
            df = pd.read_excel('data/forecast/{}'.format(name), engine='openpyxl')
        # infer dates
        for col in df.columns:
            if df[col].dtype == np.object:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass

        return create_table(df),\
            html.Div([
                html.H1(["Coming Soon!"], style={'textAlign': 'center'})
            ]), df.to_json()