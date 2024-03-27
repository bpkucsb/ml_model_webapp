# Dash Webapp for Credit Check

## App Functionality

This app allows the user to input their income, debt, fico score and purpose of the loan they are trying to get. The model looks up mean values for remaining fields and feeds them to a classifier that score the input. This score corresponds roughly to the risk that the customer will default on their loan.

Additionally histograms are drawn for the user inputable fields so that the webapp user can understand where they fall along the risk continuum.

## Design

The app loads data from a pandas dataframe in order to draw histograms and determine mean values for inputs not provided by the user. The app sets up a webpage using dash module functionalities. The machine learning classifiers are loaded using joblib.

I use the dash module from plotly to create a simple web app that loads data and uses the ML models from creditworthiness_ml_model repo. The user can input his or her income, debt and purpose of the loan and get scored by the model. The histograms for these features are also displayed.

## Screenshot

![Screen Shot 2024-03-26 at 5 29 35 PM](https://github.com/bpkucsb/ml_model_webapp/assets/13769127/d31a4b7c-cb8b-4a71-a2f9-724ff817e2c4)

## TODO:

Logistic regression still returning just 1 or 0. Need to investigate
T-sne plot for each model showing tp, fp, tn and fn
Better way to input default (not user provided) values to models
