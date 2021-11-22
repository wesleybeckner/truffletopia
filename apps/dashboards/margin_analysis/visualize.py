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
        routes_pathname_prefix='/visualize/',
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
        Output('groupby1', 'options'),
        Output('groupby2', 'options'),
        Output('groupby1', 'value'),
        Output('groupby2', 'value'),
        Output('df', 'data'),
        Input('file', 'value'))
    def view_data(name):
        if 'csv' in name:
            df = pd.read_csv('data/{}'.format(name))
        elif 'xlsx' in name:
            df = pd.read_excel('data/{}'.format(name), engine='openpyxl')
        # infer dates
        for col in df.columns:
            if df[col].dtype == np.object:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        groupby = list(df.columns[df.dtypes != np.float64].values)

        return create_table(df, dark_mode=False),\
            html.Div([
            dcc.Graph(
                        id='fig-1',
                        # figure=fig1,
                        style={'height': style['height'],
                                'margin': '10px',
                                'padding': '0px'}
                        ),
            dcc.Graph(
                        id='fig-2',
                        # figure=fig2,
                        style={'height': style['height'],
                                'margin': '10px'}
                        ),
        ]), [{'label': i, 'value': i} for i in groupby],\
            [{'label': i, 'value': i} for i in groupby],\
            groupby[0],\
            groupby[1],\
            df.to_json()

    @app.callback(
        Output('fig-1', 'figure'),
        Output('fig-2', 'figure'),
        Output('file', 'options'),
        Input('groupby1', 'value'),
        Input('groupby2', 'value'),
        State('df', 'data'),
        Input('fig-1', 'clickData'),
        Input('fig-2', 'clickData'),
        Input('reset', 'n_clicks'),
        Input('data', "derived_virtual_data"),
        Input('data', "derived_virtual_selected_rows"))
    def update_fig(groupby1, groupby2, data, click1, click2, reset,
                   rows, derived_virtual_selected_rows):
        # UPDATE FILE LIST
        # establish local data files
        dir_name = 'data/'
        # Get list of all files only in the given directory
        f = filter( lambda x: os.path.isfile(os.path.join(dir_name, x)),
                                os.listdir(dir_name) )
        # Sort list of files based on last modification time in ascending order
        f = sorted( f,
                                key = lambda x: os.path.getmtime(os.path.join(dir_name, x))
                                )
        f = f[::-1]

        # UPDATE PLOTS
        ctx = dash.callback_context
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        df = pd.read_json(data)
        if derived_virtual_selected_rows is None:
            derived_virtual_selected_rows = []

        df = df if rows is None else pd.DataFrame(rows)
        target = list(df.columns[df.dtypes == np.float64].values)[0]

        # INFER DATES
        for col in df.columns:
            if df[col].dtype == np.object:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass

        ### CHART INTERACTIVITY FOR NON FLOATING POINT SELECTION
        if (trigger == 'fig-1') and (df[groupby1].dtype != np.float64):
            grp = click1['points'][0]['y']
            df = df.loc[df[groupby1] == grp].reset_index(drop=True)
        elif (trigger == 'fig-2') and (df[groupby2].dtype != np.float64):
            grp = click2['points'][0]['y']
            df = df.loc[df[groupby2] == grp].reset_index(drop=True)
        elif trigger == 'reset':
            pass

        legend_order = None
        
        # FIGURE 1
        # GROUPBY SORT FOR CATEGORICAL
        if (df[groupby1].dtype == np.object):
            df = df.iloc[df.groupby([groupby2, groupby1])[target].transform('median')\
                .sort_values().index].reset_index(drop=True)
            legend_order = df[groupby1].unique()
            fig1 = box(df, y=target, x=groupby1, color=groupby1)
            
        # FIGURE 2
        # GROUPBY SORT FOR CATEGORICAL
        if (df[groupby2].dtype == np.object):
            df3 = df.iloc[df.groupby([groupby2, groupby1])[target].transform('median')\
                .sort_values().index].reset_index(drop=True)
            if legend_order is None:
                legend_order = df[groupby1].unique()
            fig2 = box(df3, y=target, x=groupby2, color=groupby1)

        # FIGURE 1
        # RESAMPLE FOR DATE
        elif df[groupby1].dtype == '<M8[ns]':
            df_select = df.groupby(groupby1).resample('M', on=groupby2).median()
            df_select = df_select.reindex(legend_order, level=0)
            df_select = df_select.reset_index()
            fig1 = line(df_select, y=target, x=groupby1, color=groupby1)

        # FIGURE 2
        # RESAMPLE FOR DATE
        elif df[groupby2].dtype == '<M8[ns]':
            df_select = df.groupby(groupby1).resample('M', on=groupby2).median()
            df_select = df_select.reindex(legend_order, level=0)
            df_select = df_select.reset_index()
            fig2 = line(df_select, y=target, x=groupby2, color=groupby1)
        return fig1, fig2, [{'label': i, 'value': i} for i in f]