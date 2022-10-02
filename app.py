'''
 # @ Create Time: 2022-10-02 08:40:20.795480
 Kamil Górny
 Kamil Bukalski
'''

from dash import Dash, html, dcc, Input, Output, State
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from joblib import load
import xgboost as xgb

app = Dash(__name__, title = 'Startup prediction project')
server = app.server
app.layout = html.Div(children=[
    html.Div(['Startup success prediction'], style={'color': 'LightGreen', 'font-size': 40, 'textAlign': 'center', }),
    html.Br(),
    html.Div([
        html.Div(children=[
            html.Label('Age'),
            dcc.Slider(
                id='age-slider',
                min=0,
                max=40,
                step=1,
                marks={i: f'Label {i}' if i == 1 else str(i) for i in range(0, 42, 2)},
                value=5,
            ),

            html.Br(),
            html.Label('Relationships'),
            dcc.Slider(
                id='relationships-slider',
                min=0,
                max=60,
                step=1,
                marks={i: f'Label {i}' if i == 1 else str(i) for i in range(0, 62, 2)},
                value=5,
            ),

            html.Br(),
            html.Label('Milestones'),
            dcc.Slider(
                id='milestones-slider',
                min=0,
                max=30,
                step=1,
                marks={i: f'Label {i}' if i == 1 else str(i) for i in range(0, 32, 2)},
                value=5,
            ),

            html.Br(),
            html.Label('Average participants'),
            dcc.Slider(
                id='participants-slider',
                min=0,
                max=30,
                step=1,
                marks={i: f'Label {i}' if i == 1 else str(i) for i in range(0, 32, 2)},
                value=5,
            ),

            html.Br(),
            html.Label('Funding rounds'),
            dcc.Slider(
                id='funding-rounds-slider',
                min=0,
                max=20,
                step=1,
                marks={i: f'Label {i}' if i == 1 else str(i) for i in range(0, 22, 2)},
                value=5,
            ),

            html.Br(),
            html.Label('Age first funding year'),
            dcc.Slider(
                id='first-funding-slider',
                min=0,
                max=30,
                step=1,
                marks={i: f'Label {i}' if i == 1 else str(i) for i in range(0, 32, 2)},
                value=5,
            ),

            html.Br(),
            html.Label('Age last funding year'),
            dcc.Slider(
                id='last-funding-slider',
                min=0,
                max=30,
                step=1,
                marks={i: f'Label {i}' if i == 1 else str(i) for i in range(0, 32, 2)},
                value=5,
            ),

            html.Br(),
            html.Label('Age first milestone year'),
            dcc.Slider(
                id='first-milestone-slider',
                min=0,
                max=30,
                step=1,
                marks={i: f'Label {i}' if i == 1 else str(i) for i in range(0, 32, 2)},
                value=5,
            ),

            html.Br(),
            html.Label('Age last milestone year'),
            dcc.Slider(
                id='last-milestone-slider',
                min=0,
                max=30,
                step=1,
                marks={i: f'Label {i}' if i == 1 else str(i) for i in range(0, 32, 2)},
                value=5,
            ),

        ], style={'padding': 10, 'flex': 1}),

        html.Div(children=[
            html.Label('Type'),
            dcc.RadioItems(
                ['Software', 'Web', 'Mobile', 'Enterprise', 'Advertising',
                 'Gamesvideo', 'Ecommerce', 'Biotech', 'Consulting', 'Othercategory'],
                'Enterprise', id='type-radio'
            ),

            html.Br(),
            html.Label('Investors'),
            dcc.Checklist(['VC', 'Angel'], id='investors'
                          ),

            html.Br(),
            html.Label('Series Funding'),
            dcc.Checklist(['A', 'B', 'C', 'D'], id='series-funding'
                          ),

            html.Br(),
            html.Label('Top 500'),
            dcc.RadioItems(
                ['Yes', 'No'],
                'No', id='500-radio'
            ),

            html.Br(),
            html.Label('Total funding USD'),
            dcc.Input(id="total-funding", type="text", placeholder="", value="50000", style={'marginRight': '10px'}),

            html.Br(),
            html.Br(),
            html.Label('Model'),
            dcc.RadioItems(
                ['RandomForest', 'XGBoost'],
                'XGBoost', id='model-radio'
            ),

            html.Br(),
            html.Br(),
            html.Button('Predict', id='predict-val', n_clicks=0),

            html.Br(),
            html.Br(),
            html.Div(id='prediction-output'),
        ], style={'padding': 10, 'flex': 1})
    ], style={'display': 'flex', 'flex-direction': 'row'})
    ])


@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-val', 'n_clicks'),
    State('age-slider', 'value'),
    State('relationships-slider', 'value'),
    State('milestones-slider', 'value'),
    State('participants-slider', 'value'),
    State('funding-rounds-slider', 'value'),
    State('first-funding-slider', 'value'),
    State('last-funding-slider', 'value'),
    State('first-milestone-slider', 'value'),
    State('last-milestone-slider', 'value'),
    State('type-radio', 'value'),
    State('investors', 'value'),
    State('model-radio', 'value'),
    State('500-radio', 'value'),
    State('total-funding', 'value'),
    State('series-funding', 'value'),

)
def update_output(n_clicks, age, relationships, milestones, participants,
                  funding_rounds, first_funding, last_funding, first_milestone,
                  last_milestone, type_radio, investors, model, is_500,
                  total_funding, series_funding):

    # dump information to that file
    rfc = load('saved_rfc.pkl')
    gbc = xgb.XGBClassifier()
    gbc.load_model('saved_gbc.pkl')

    buisnes_type = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if type_radio == 'Software':
        buisnes_type[0] = 1
    elif type_radio == 'Web':
        buisnes_type[1] = 1
    elif type_radio == 'Mobile':
        buisnes_type[2] = 1
    elif type_radio == 'Enterprise':
        buisnes_type[3] = 1
    elif type_radio == 'Advertising':
        buisnes_type[4] = 1
    elif type_radio == 'Gamesvideo':
        buisnes_type[5] = 1
    elif type_radio == 'Ecommerce':
        buisnes_type[6] = 1
    elif type_radio == 'Biotech':
        buisnes_type[7] = 1
    elif type_radio == 'Consulting':
        buisnes_type[8] = 1
    elif type_radio == 'Othercategory':
        buisnes_type[9] = 1

    VC = 0
    angel = 0
    if investors != None:
        if 'VC' in investors:
            VC = 1
        if 'Angel' in investors:
            angel = 1

    A = 0
    B = 0
    C = 0
    D = 0
    if series_funding != None:
        if 'A' in series_funding:
            A = 1
        if 'B' in series_funding:
            B = 1
        if 'C' in series_funding:
            C = 1
        if 'D' in series_funding:
            D = 1

    if total_funding.isnumeric():
        total_funding = float(total_funding)
    else:
        total_funding = 0

    if is_500 == 'Yes':
        is_500 = 1
    else:
        is_500 = 0

    pred_data = {'age': [age],
                 'age_first_funding_year': [first_funding],
                 'age_last_funding_year': [last_funding],
                 'age_first_milestone_year': [first_milestone],
                 'age_last_milestone_year': [last_milestone],
                 'relationships': [relationships],
                 'funding_rounds': [funding_rounds],
                 'funding_total_usd': [total_funding],
                 'milestones': [milestones],
                 'is_software': [buisnes_type[0]],
                 'is_web': [buisnes_type[1]],
                 'is_mobile': [buisnes_type[2]],
                 'is_enterprise': [buisnes_type[3]],
                 'is_advertising': [buisnes_type[4]],
                 'is_gamesvideo': [buisnes_type[5]],
                 'is_ecommerce': [buisnes_type[6]],
                 'is_biotech': [buisnes_type[7]],
                 'is_consulting': [buisnes_type[8]],
                 'is_othercategory': [buisnes_type[9]],
                 'has_VC': [VC],
                 'has_angel': [angel],
                 'has_roundA': [A],
                 'has_roundB': [B],
                 'has_roundC': [C],
                 'has_roundD': [D],
                 'avg_participants': [participants],
                 'is_top500': [is_500]}

    pred_data2 = pd.DataFrame.from_dict(pred_data)
    predicted = 0

    if model == 'RandomForest':
        predicted = rfc.predict_proba(pred_data2)
        predicted = predicted[0][1]
    else:
        predicted = gbc.predict_proba(pred_data2)
        predicted = predicted[0][1]

    if predicted<0.4:
        final_text = f'Your chance of success is {predicted*100}. Rethink your business plan'
    elif predicted>0.4 and predicted<0.6:
        final_text = f'Your chance of success is {predicted * 100}. You have a great chance however you must continue to work hard'
    else:
        final_text = f'Your chance of success is {predicted * 100}. It seems that you have a bright future ahead of you'
    return final_text


if __name__ == '__main__':
    app.run_server(debug=True)
