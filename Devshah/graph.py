import dash
import dash as dcc
import dash as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

df = pd.read_csv('/Users/asadtariq/Downloads/IDS/python_40/Work/Devshah/danielle_ed_actively.csv')
#df.head()

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id="ticker",
        options=[{"label": x, "value": x} 
                 for x in df.columns[1:]],
        value=df.columns[1],
        clearable=False,
    ),
    dcc.Graph(id="time-series-chart"),
])

@app.callback(
    Output("time-series-chart", "figure"), 
    [Input("ticker", "value")])
def display_time_series(ticker):
    fig = px.line(df, x='date', y=ticker)
    return fig

app.run_server(debug=True)