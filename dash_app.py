from dash import Dash, html, dcc, Input, Output, callback
import plotly.express as px
import pandas as pd
import numpy as np
import joblib

# Read loan data, one hot encode some fields and calculated income and debt fields
df = pd.read_csv("PeerLoanKart_fraud.csv")
df = pd.concat((df, pd.get_dummies(df["purpose"], dtype="int")), axis=1)
# Calculate income from base values and clip at three standard deviations
df["income"] = np.exp(df["log.annual.inc"])
df["income_capped"] = df["income"].clip(upper = 4*df["income"].std())
# Calculate income from base values and clip at three standard deviations
df["debt"] = df["dti"] * df["income"]
df["debt_capped"] = df["debt"].clip(upper = 4*df["debt"].std())
# Convert target variable to color for graph
df["target"] = (df["Loan Repayment Status"]=="Paid").map(lambda x: "loan paid" if x else "loan not paid")

# Create feature lists

loan_purpose_list = ['credit_card',
                     'debt_consolidation',
                     'educational',
                     'home_improvement',
                     'major_purchase',
                     'small_business',]

features = ['log.annual.inc',
            'fico',
            'delinq.2yrs',
            'pub.rec',
            'int.rate',
            'installment',
            'days.with.cr.line',
            'inq.last.6mths',
            'dti',
            'revol.util',
            'revol.bal',
          ]

# Set features to median value and all binary features to zero

df_mean = df[loan_purpose_list + features].mean()

for x in loan_purpose_list:
    df_mean[x] = 0

# Comput histograms to be displayed

# Normalize purpose of loan counts. First compute counts for purpose and target. Then set target values to columns. Sum over purpose for target = 0/1. Then return target and purpose to columns
df_count = df[["purpose", "target"]].value_counts().unstack(-1)
df_count = df_count/df_count.sum()
df_count = df_count.stack(-1).reset_index()

fig_income  = px.histogram(df, x="income_capped", color="target", nbins=50, histnorm='probability', barmode="overlay")
fig_debt    = px.histogram(df, x="debt_capped", color="target", nbins=50, histnorm='probability', barmode="overlay")
fig_fico    = px.histogram(df, x="fico", color="target", nbins=50, histnorm='probability', barmode="overlay")
fig_purpose = px.bar(df_count, x="purpose", y=0, color="target", barmode="overlay")

# start the dash app
app = Dash(__name__)

# Allow user to view histograms of certain input fields. Update credit score from inputs
app.layout = html.Div([
    html.Div(className="parent", children=[
        html.Div(className="row", children=[
            html.H3('Income'),
            html.Div(className="parent", children=[
                html.Div(children='Income in thousand of dollars:'),
                dcc.Input(id="income", type="number", value=1, min=20, max=1000000, step=1),
            ]),
            dcc.Graph(figure=fig_income, style={'width': '50vh', 'height': '50vh'}),
        ]),
        html.Div(className="row", children=[
            html.H3('Debt'),
            html.Div(className="parent", children=[
                html.Div(children='Debt in thousand of dollars:'),
                dcc.Input(id="debt", type="number", value=1, min=0, max=1000000, step=1),
            ]),
            dcc.Graph(figure=fig_debt, style={'width': '50vh', 'height': '50vh'}),
        ]),
        html.Div(className="row", children=[
            html.H3('Fico'),
            html.Div(className="parent", children=[
                html.Div(children='Fico score:'),
                dcc.Input(id="fico", type="number", value=700, min=300, max=850, step=1),
            ]),
            dcc.Graph(figure=fig_fico, style={'width': '50vh', 'height': '50vh'}),
        ]),
        html.Div(className="row", children=[
            html.H3('Purpose of Loan'),
            html.Div("This loan will be used towards:"),
            dcc.Dropdown(loan_purpose_list, value='credit_card', id="loan_purpose"),
            dcc.Graph(figure=fig_purpose, style={'width': '50vh', 'height': '50vh'}),
        ]),
    ]),
    dcc.Dropdown(["logistic_regression", "random_forest", "xgboost"], value='xgboost', id="model"),
    html.H3(children='Risk of Default'),
    html.Div(id="score"),
])



# Set callback to score is updated automatically with new inputs
@callback(
    Output("score", "children"),
    Input("income", "value"),
    Input("debt", "value"),
    Input("fico", "value"),
    Input("loan_purpose", "value"),
    Input("model", "value"),
)
def update_score(income, debt, fico, loan_purpose, model):

    # Copy mean values
    df_input = df_mean.copy(deep=True)

    # Fill in fields with user input when available    
    if loan_purpose is not None:
        df_input[loan_purpose] = 1
    if fico is not None:
        df_input["fico"] = fico
    if (income is not None) and (income>0):
        income = income * 1000
        df_input["log.annual.inc"] = np.log(income)
        if debt is not None:
            debt = debt * 1000
            df_input["dti"] = debt/income

    # Load classifier, compute probabilyt score and display it
    if model is not None:
        clf = joblib.load(model + "_credit.joblib")
        pred = clf.predict_proba(df_input.to_frame().transpose())
        score = pred[0, 1]
    else:
        score = 0.0
    return f'Your percentage risk is : {score}'

if __name__ == '__main__':
    app.run(debug=True)