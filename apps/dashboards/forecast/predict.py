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
from .mlfunctions import *

# Build App
def init_dashboard(server):

    dash_app = Dash(
        server=server,
        routes_pathname_prefix='/forecast/',
        external_stylesheets=external_stylesheets,
        title = "🍫 Truffletopia 🍭",
    )

    ### LAYOUT ###
    CONTROLS = html.Div([
        html.H2("Prediction Controls"),
        html.P("Data File", style={'margin-top': '15px'}),
        dcc.Dropdown(
        multi=False,
        id="file",
        # style={'color': darkcolors['text'],
        #        'textColor': 'white',
        #        'backgroundColor': darkcolors['background']},
        options=[{'label': i, 'value': i} for i in f],
        value=f[0],
        ),
        html.P("Model", style={'margin-top': '15px'}),
        dcc.Dropdown(
        multi=False,
        id="model",
        # style={'color': darkcolors['text'],
        #        'text-color': 'white',
        #        'backgroundColor': darkcolors['background']},
        options=[{'label': i, 'value': i} for i in m],
        value=m[0],
        ),
        html.P("Backward Projection", style={'margin-top': '15px'}),
        dcc.Slider(
            id='backward',
            min=1,
            max=36,
            step=1,
            value=5,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.P("Foward Projection", style={'margin-top': '15px'}),
        dcc.Slider(
            id='forward',
            min=1,
            max=36,
            step=1,
            value=5,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div(id='table', style={'margin-top': '15px',
                                    'height': '300px'}),
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
        Input('file', 'value'),
        Input('model', 'value'))
    def view_data(data_file, model_file):
        if 'csv' in data_file:
            df = pd.read_csv('data/forecast/{}'.format(data_file))
        elif 'xlsx' in data_file:
            df = pd.read_excel('data/forecast/{}'.format(data_file), engine='openpyxl')

        # process data
        df_pivot = process_data(df)
        X_train, X_test, qty, enc = train_test_split(df_pivot, groupby=None, verbiose=True)


        # model
        model = load_model(model_file)

        dff = make_forcast(model, df_pivot, window=36, qty=qty, 
                           projection=5, groupby=None, item='Doughnut',
                           time_delta_previous=5)
        fig1 = bar(dff, x='Month', y='KG', color='Source')
        
        ### FOR SECOND PLOT, SOME REDUNDANCY
        X, y, labels = sweep_window(X_train, qty, window=36, verbiose=True)
        pred = model.predict(X)
        dff2 = pd.DataFrame([pred,y]).T
        dff2.columns = ['Prediction', 'Actual']

        fig2 = parity(dff2, x='Actual', y='Prediction') 
        fig2.update_layout(title='R2: {:.3f}'.format(r2_score(y, pred)))

        return create_table(df, export=False),\
            html.Div([
            dcc.Graph(
                        id='fig-1',
                        figure=fig1,
                        style={'height': style['height'],
                                'margin': '10px',
                                'padding': '0px'}
                        ),
            dcc.Graph(
                        id='fig-2',
                        figure=fig2,
                        style={'height': style['height'],
                                'margin': '10px'}
                        ),
        ]), df.to_json()

    @app.callback(
        Output('fig-1', 'figure'),
        Output('fig-2', 'figure'),
        State('df', 'data'),
        Input('data', "derived_virtual_data"),
        Input('data', "derived_virtual_selected_rows"),
        Input('model', 'value'),
        Input('forward', 'value'),
        Input('backward', 'value'))
    def update_fig(data, rows, derived_virtual_selected_rows, model_file,
                   forward, backward):
        if derived_virtual_selected_rows is None:
            derived_virtual_selected_rows = []

        df = pd.read_json(data) if rows is None else pd.DataFrame(rows)
        df_pivot = process_data(df)
        X_train, X_test, qty, enc = train_test_split(df_pivot, groupby=None, verbiose=True)


        # model
        model = load_model(model_file)

        dff = make_forcast(model, df_pivot, window=36, qty=qty, 
                           projection=forward, groupby=None, item='Doughnut',
                           time_delta_previous=backward)
        fig1 = bar(dff, x='Month', y='KG', color='Source')
        
        ### FOR SECOND PLOT, SOME REDUNDANCY
        X, y, labels = sweep_window(X_train, qty, window=36, verbiose=True)
        pred = model.predict(X)
        dff2 = pd.DataFrame([pred,y]).T
        dff2.columns = ['Prediction', 'Actual']

        fig2 = parity(dff2, x='Actual', y='Prediction') 
        fig2.update_layout(title='R2: {:.3f}'.format(r2_score(y, pred)))

        return fig1, fig2