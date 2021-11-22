import plotly.express as px
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash import dash_table, html
from dash.dash_table.Format import Format, Scheme, Trim
import plotly.express as px
import pandas as pd
import os

# invisible = rgba(0,0,0,0)
darkcolors = {'background': 'rgb(30, 30, 30)',
          'text': 'white'}
colors = {'background': '#FFFFFF',
          'text': '#111111'}
style = {'height': 300}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                        dbc.themes.VAPOR # SLATE,VAPOR
                        ]
plot_layouts = {
    "template": "plotly_dark",
    # "plot_bgcolor": colors['background'],
    # "paper_bgcolor": colors['background'],
    "height": style['height'],
    "margin": dict(
        l=0,
        r=0,
        b=10,
        t=30,
        pad=4
    )}

small_plot_layouts = plot_layouts.copy()
small_plot_layouts['height'] = 100
small_plot_layouts['font'] = dict(size=6)

df = pd.read_csv('data/oee_1000.csv')
moodsdf = pd.read_csv('data/moods_short.csv')

# establish local data files
dir_name = 'data/forecast/'
# Get list of all files only in the given directory
f = filter(lambda x: os.path.isfile(os.path.join(dir_name, x)),
           os.listdir(dir_name))
# Sort list of files based on last modification time in ascending order
f = sorted(f,
           key=lambda x: os.path.getmtime(os.path.join(dir_name, x))
           )

DROPDOWN = dbc.DropdownMenu(
    label="Menu",
    children=[
        dbc.DropdownMenuItem(
            "Forecast", href="/forecast/", external_link=True),
        dbc.DropdownMenuItem("Train", href="/train/", external_link=True),
    ],
)

NAVBAR = dbc.Navbar(
    dbc.Row([
            dbc.Col(
                html.A(
                    html.H2("🍫 Truffletopia 🍭",
                            style={'margin': '10px'}),
                    href='/',
                    style={'color': colors['text'],
                           'text-decoration': 'none'},
                ),
                width=4),

            dbc.Col(
                    html.H2("Customer Forecasts",
                            style={'margin': '10px',
                                   'text-align': 'center',
                                   'color': colors['text'], }
                            ),
                width=4, align="center"),

            dbc.Col(dbc.Row(
                [DROPDOWN],
                className="row",
                style={
                    'float': 'right',
                },
            ),
                width=4),

            ],
            justify="between",
            align="center",
            className="twelve columns",
            ),
)


def scatter(df, desc=None, val=None):
    x = df.index
    y = 'ebit_per_hr'
    if desc:
        df = df.loc[df[desc] == val]
        x = df.index
    fig = px.scatter(df,
                     y=y,
                     x=x,
                     color='base_cake')

    fig.update_layout(plot_layouts)
    fig.update_layout({"title": "EBIT per Hr by Base Cake"})
    return fig


def sun(df, desc=None, val=None):
    path = ['base_cake', 'truffle_type']
    if desc:
        df = df.loc[df[desc] == val]
        path = ['base_cake', 'truffle_type', 'primary_flavor',
                'secondary_flavor', 'color_group']
    fig = px.sunburst(df, path=path, color='ebit_per_hr')
    fig.update_layout(plot_layouts)
    return fig


def violin(df, moodsdf, ascending=False):
    fig = go.Figure()
    moodsdf = moodsdf.sort_values(
        'group_median', ascending=ascending).reset_index(drop=True)
    for index in range(5):
        x = df.loc[(df[moodsdf.iloc[index]['descriptor']] ==
                    moodsdf.iloc[index]['group'])]['ebit_per_hr']
        y = moodsdf.iloc[index]['descriptor'] + ': ' + df.loc[(df[moodsdf
                                                                  .iloc[index]['descriptor']] == moodsdf.iloc[index]['group'])][moodsdf.iloc[index]['descriptor']]
        name = '${:.2f}'.format(x.median())
        fig.add_trace(go.Violin(x=y,
                                y=x,
                                name=name,
                                box_visible=True,
                                meanline_visible=True))
    fig.update_layout(plot_layouts)
    fig.update_layout({"yaxis.title": "EBIT/HR"})
    return fig


def box(df, y, x, color):
    fig = px.box(df, y, x, color, points='all')
    fig.update_layout(plot_layouts)
    fig.update_layout(showlegend=False)
    return fig


def line(df, y, x, color):
    fig = px.line(df, y, x, color)
    fig.update_layout(plot_layouts)
    fig.update_layout(showlegend=False)
    return fig

def bar(df, x, y, color):
    fig = px.bar(df, x, y, color, barmode='group')
    fig.update_layout(plot_layouts)
    return fig

def parity(df, x, y):
    fig = px.scatter(df, x, y) 
    fig.update_layout(plot_layouts)
    return fig

def seasonal(df, customer):
    fig1 = px.line(df, x='month', y='kg', color='year')
    fig1.update_traces(line=dict(color = 'rgba(50,50,50,0.2)'))
    fig2 = px.scatter(df, x='month', y='kg', color='year', color_continuous_scale='viridis')
    fig3 = go.Figure(data=fig1.data + fig2.data)
    fig3.update_layout(small_plot_layouts)
    fig3.update_layout({'title': customer,
                        'showlegend': False})

    return fig3

def table_type(df_column):
    # Note - this only works with Pandas >= 1.0.0
    
    if (isinstance(df_column.dtype, pd.DatetimeTZDtype) or
       (df_column.dtype == '<M8[ns]')):
        
        return 'datetime'
    elif (isinstance(df_column.dtype, pd.StringDtype) or
            isinstance(df_column.dtype, pd.BooleanDtype) or
            isinstance(df_column.dtype, pd.CategoricalDtype) or
            isinstance(df_column.dtype, pd.PeriodDtype)):
        return 'text'
    elif (isinstance(df_column.dtype, pd.SparseDtype) or
            isinstance(df_column.dtype, pd.IntervalDtype) or
            isinstance(df_column.dtype, pd.Int8Dtype) or
            isinstance(df_column.dtype, pd.Int16Dtype) or
            isinstance(df_column.dtype, pd.Int32Dtype) or
            isinstance(df_column.dtype, pd.Int64Dtype) or
             (df_column.dtype == np.float64)):
        return 'numeric'
    else:
        return 'any'

def create_table(df, id='data', filter_action='native', export='xlsx',
                 fixed_header=True, height='260', scheme=Scheme.fixed,
                 trim=Trim.yes, dark_mode=True):
    if scheme and trim:
        table = dash_table.DataTable(
            id=id,
            data=df.to_dict('records'),
            columns=[
                {'name': i, 'id': i, 'type': table_type(df[i]),
                'format': Format(precision=3, scheme=scheme,
                    trim=trim)} for i in df.columns
            ])
    else:
        table = dash_table.DataTable(
            id=id,
            data=df.to_dict('records'),
            columns=[
                {'name': i, 'id': i, 'type': table_type(df[i]),
                'format': Format(precision=3)} for i in df.columns
            ])
    table.style_table={'overflowX': 'auto',
                        'height': f'{height}px', 'overflowY': 'auto',
                        "margin": dict(
                                        l=0,
                                        r=0,
                                        b=0,
                                        t=30,
                                        pad=4
                                    )
                        }
    table.style_cell={'minWidth': 95}
    if filter_action:
        table.filter_action = filter_action
    if export:
        table.export = export
    if fixed_header:
        table.fixed_rows={'headers': True}
    if dark_mode:
        table.style_data={
                'color': 'white',
                'backgroundColor': 'rgb(50, 50, 50)',}
        table.style_filter={
                'color': 'white',
                'backgroundColor': 'rgb(100, 100, 100)',}
        table.style_header={
                'color': 'white',
                'backgroundColor': 'rgb(30, 30, 30)',}
        table.style_table['backgroundColor'] = 'black'
    
    return table
