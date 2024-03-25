from dash import Dash, html, dcc, Input, Output, callback
import plotly.express as px
import pandas as pd
import numpy as np
import joblib

# Read loan data, one hot encode some fields and calculated income and debt fields
df = pd.read_csv("PeerLoanKart_fraud.csv")
df = pd.concat((df, pd.get_dummies(df["purpose"], dtype="int")), axis=1)
df["income"] = np.exp(df["log.annual.inc"])
df["debt"] = df["dti"] * df["income"]

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

df_median = df[loan_purpose_list + features].median()

for x in loan_purpose_list:
    df_median[x] = 0

# Comput histograms to be displayed

fig_income = px.histogram(df, "income")
fig_debt   = px.histogram(df, "debt")

# start the dash app
app = Dash(__name__)

# Allow user to view histograms of certain input fields. Update credit score from inputs
app.layout = html.Div([
    html.Div(children='Income in thousand of dollars'),
    dcc.Graph(figure=fig_income),
    dcc.Input(id="income", type="number", value=1, min=20, max=1000000, step=1),
    html.Div(children='Debt in thousands of dollars'),
    dcc.Graph(figure=fig_debt),
    dcc.Input(id="debt", type="number", value=0, min=0, max=1000000, step=1),
    html.Div("Purpose of Loan"),
    dcc.Dropdown(loan_purpose_list, value='credit_card', id="loan_purpose"),
    html.Div(children='Score'),
    html.Div(id="score"),
])

# Set callback to score is updated automatically with new inputs
@callback(
    Output("score", "children"),
    Input("income", "value"),
    Input("debt", "value"),
    Input("loan_purpose", "value"),
)
def update_score(income, debt, loan_purpose):

    # Fill in fields with user input
    df_input = df_median.copy(deep=True)
    if loan_purpose is not None:
        df_input[loan_purpose] = 1
    if (income is not None) and (income>0):
        income = income * 1000
        df_input["log.annual.inc"] = np.log(income)
        if debt is not None:
            debt = debt * 1000
            df_input["dti"] = debt/income

    # Load classifier, compute probabilyt score and display it
    clf = joblib.load("logistic_regression_credit.joblib")
    pred = clf.predict_proba(df_input.to_frame().transpose())
    score = pred[0, 0]
    return f'Score : {score}'

if __name__ == '__main__':
    app.run(debug=True)