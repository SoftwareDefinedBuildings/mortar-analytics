import dash
import dash_core_components as dcc
import dash_html_components as html

import data

summary = data.site_summary[data.site_summary['site'] != 'hayward-station-8']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#summary = summary.set_index(summary.pop('site'))

def generate_table(dataframe, max_rows=50):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

app.layout = html.Div(children=[
    html.H1(children='Mortar/Dash Test'),
    html.Div(children='''
        Mortar Testbed working with Meter data
    '''),
    html.Div(children=[
        html.H3(children='Mean Daily Consumption by Site'),
        dcc.Graph(
            id='example-graph',
            figure={
              'data': [
                {'x': summary['site'], 'y': summary['mean_daily'], 'type': 'bar', 'name':'mean'},
                {'x': summary['site'], 'y': summary['max_daily'], 'type': 'bar', 'name':'max'}
              ],
              'layout': {
                'title': "Energy Consumption Viz",
              }
            },
        ),
    ]),
    html.Div(children=[
        html.H3(children='Summary of Min/Max/Mean Daily Consumption'),
        generate_table(summary),
    ])
])

if __name__ == '__main__':
  app.run_server(host='0.0.0.0', debug=False)
