from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc

import plotly.express as px
import pandas as pd

import bot_2d_problem
import bot_2d_rep
import glob
import os
import dill

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css])
# server = app.server

def get_latest_timestamp(path):
    files = glob.glob(path)
    if len(files) == 0:
        return None
    latest_file = max(files, key=os.path.getctime)
    timestamp = latest_file.split('_')[-1].split('.pkl')[0]
    return timestamp

timestamp = get_latest_timestamp('./_output/df_opt_*.pkl')

unopt_df = pd.read_pickle(f'./_output/df_unopt_{timestamp}.pkl')
opt_df = pd.read_pickle(f'./_output/df_opt_{timestamp}.pkl')
with open(f'./_output/problem_{timestamp}.pkl', 'rb') as file:
    problem = dill.load(file)

combined_df = pd.concat([unopt_df, opt_df])

app.layout = html.Div([
    html.H1("Generation and Selection of Sensor Packages for Mobile Robots"),
    html.H2("Problem Statement"),
    html.P("The goal of this project was to generate, select, and optimize sensor packages for mobile robots."),
    html.H2("Approach"),
    html.P("The goal of this project was to generate, select, and optimize sensor packages for mobile robots."),
    html.H2("Results"),
    dcc.Graph(id='tradespace',figure=bot_2d_problem.plot_tradespace(combined_df, unopt_df.shape[0])),
    html.Div([
        html.Img(id='unopt_bot', style={'width': '400px'}),
        html.Img(id='opt_bot', style={'width': '400px'})
    ])
])

@app.callback(
    Output(component_id='unopt_bot', component_property='src'),
    Output(component_id='opt_bot', component_property='src'),
    Input('tradespace', 'hoverData')
)
def update_bots(hoverData):
    
    matplotlib.pyplot.close()

    if hoverData is None:
        return {}
    if len(hoverData['points']) < 2:
        return {}

    ub_idx = hoverData['points'][0]['pointIndex']
    ub = problem.convert_1D_to_bot(combined_df.iloc[ub_idx]['X'])
    ub_buf = BytesIO()
    ub_fig = ub.plot_bot(title="Pre-Optimization")
    ub_fig.savefig(ub_buf, format="png")
    # Embed the result in the html output.
    ub_fig_data = base64.b64encode(ub_buf.getbuffer()).decode("ascii")
    ub_fig_bar_matplotlib = f'data:image/png;base64,{ub_fig_data}'

    ob_idx = hoverData['points'][1]['pointIndex']
    ob = problem.convert_1D_to_bot(combined_df.iloc[ob_idx]['X'])
    ob_buf = BytesIO()
    ob_fig = ob.plot_bot(title="Optimized")
    ob_fig.savefig(ob_buf, format="png")
    # Embed the result in the html output.
    ob_fig_data = base64.b64encode(ob_buf.getbuffer()).decode("ascii")
    ob_fig_bar_matplotlib = f'data:image/png;base64,{ob_fig_data}'

    return ub_fig_bar_matplotlib, ob_fig_bar_matplotlib


if __name__ == '__main__':
    app.run(debug=True)