from dash import dcc, html, callback_context, Dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd

### FIGURES ###
from .plotlyfunctions import *

# Build App
def init_dashboard(server):

    dash_app = Dash(
        server=server,
        routes_pathname_prefix='/analyze/',
        external_stylesheets=external_stylesheets,
        title = "🍫 Truffletopia 🍭",
    )

    ### LAYOUT ###
    CONTROLS = html.Div([
        html.H2("Analysis Controls"),
        html.Button('Reset', id='reset', n_clicks=0),
        html.P(),
        html.P("Run Moods Median Analysis For Products of:\n\n", id='msg-analysis'),
        html.Div([
            html.Button('High-Value', id='high', n_clicks=0),
            html.Button('Low-Value', id='low', n_clicks=0),
        ]),
        html.P(),
        html.P("Select Product Descriptor in the Distributions Below"),
        html.P(),
        dcc.Graph(
            id='fig-3',
            figure=violin(df, moodsdf), # YOUR FIGURE HERE
            ),
    ])
    
    dash_app.layout = html.Div([
        NAVBAR,  
        html.Div([   
            html.Div([
                CONTROLS,
            ], 
            className='four columns'
            ),

            html.Div([
                html.H3("Portfolio, All Products", id='main'),
                        dcc.Graph(
                            id='fig-1',
                            figure=scatter(df),
                            ),
                        dcc.Graph(
                            id='fig-2',
                            figure=sun(df),
                            ),
            ], 
            className='eight columns'
            ),
        ], 
        className='twelve columns',
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
        [Output('fig-1', 'figure'),
         Output('fig-2', 'figure')],
        [Input('fig-3', 'clickData'),
         Input('reset', 'n_clicks'),
         Input('high', 'n_clicks'),
         Input('low', 'n_clicks'),])
    def filter_data(data, reset, high, low):
        changed_id = [p['prop_id'] for p in callback_context.triggered][0]
        if 'reset' in changed_id:
            return scatter(df), sun(df)
        elif 'high' in changed_id:
            return scatter(df), sun(df)
        elif 'low' in changed_id:
            return scatter(df), sun(df)
        elif data:
            desc = pd.DataFrame(data['points'])['x'][0].split(": ")[0]
            val = pd.DataFrame(data['points'])['x'][0].split(": ")[1]
            return scatter(df, desc, val), sun(df, desc, val)
        else:
            return scatter(df), sun(df)

    @app.callback(
        Output('main', 'children'),
        [Input('fig-3', 'clickData'),
         Input('reset', 'n_clicks'),
         Input('high', 'n_clicks'),
         Input('low', 'n_clicks'),])
    def filter_data(data, reset, high, low):
        changed_id = [p['prop_id'] for p in callback_context.triggered][0]
        if ('reset' in changed_id) or ('low' in changed_id) or ('high' in changed_id):
            return "Portfolio, All Products"
        elif data:
            desc = pd.DataFrame(data['points'])['x'][0].split(": ")[0]
            val = pd.DataFrame(data['points'])['x'][0].split(": ")[1]
            return "Portfolio from Selection: {}".format(val)
        else:
            return "Portfolio, All Products"

    ### SIDE CALLBACKS ###
    @app.callback(
        Output('fig-3', 'figure'),
        [Input('high', 'n_clicks'),
         Input('low', 'n_clicks'),])
    def filter_data(high, low):
        changed_id = [p['prop_id'] for p in callback_context.triggered][0]
        if 'high' in changed_id:
            return violin(df, moodsdf, ascending=False)
        elif 'low' in changed_id:
            return violin(df, moodsdf, ascending=True)
        else:
            return violin(df, moodsdf, ascending=False)