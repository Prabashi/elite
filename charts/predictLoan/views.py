from django.shortcuts import render

# Create your views here.
from django.views.generic import View 
   
from rest_framework.views import APIView 
from rest_framework.response import Response 

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def get_pred(age, job, marital, education, default, housing, loan, contact, month, day_of_week, campaign, pdays, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed):
    df = pd.read_csv('bank-additional-full.csv')

    X = df.iloc[:, :-1]
    X = X.drop(columns=['duration'])
    y = df.iloc[:, -1]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    # Create the preprocessing pipelines for both numeric and categorical data (nominal and ordinal).
    numeric_features = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

    numeric_transformer = StandardScaler()

    # categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    nominal_features = ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

    # categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    nominal_transformer = OneHotEncoder()

    ordinal_features = ['education']
    ordinal_transformer = OrdinalEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('nomi', nominal_transformer, nominal_features),
            ('ord', ordinal_transformer, ordinal_features)])

    # Append classifier to preprocessing pipeline to obtain full prediction pipeline.
    # clf = Pipeline(steps=[('preprocessor', preprocessor),
    #                       ('classifier', LogisticRegression(max_iter=1000))])

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split=2, random_state=42, max_depth=4, ccp_alpha=0.002))])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf.fit(X_train, y_train)
    
    row_df = pd.DataFrame(np.array([[age, job, marital, education, default, housing, loan, contact, month, day_of_week, campaign, pdays, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed]]), columns=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'])
    y_pred = clf.predict(row_df)
    
    return le.inverse_transform(y_pred)[0]
   
class HomeView(View): 
    def get(self, request, *args, **kwargs): 
        return render(request, 'predict/index.html') 

class PredictData(APIView): 
    authentication_classes = [] 
    permission_classes = [] 
   
    def get(self, request, format = None): 
        pred = get_pred(49, 'entrepreneur', 'married' ,'university.degree', 'unknown', 'yes', 'no', 'telephone', 'may', 'mon', 1, 999, 0, 'nonexistent', 1.1, 93.994, -36.4, 4.857, 5191)

        data = {
            "prediction": pred 
        };

        return Response(data) 