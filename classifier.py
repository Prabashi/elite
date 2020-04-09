#!/usr/bin/env python
# coding: utf-8

# In[55]:


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


# In[56]:


df = pd.read_csv('bank-additional-full.csv')
print('Shape of original data set', df.shape)


# In[57]:


X = df.iloc[:, :-1]
X = X.drop(columns=['duration'])
y = df.iloc[:, -1]
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)


# In[58]:


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

# a_row = pd.Series([49, 'entrepreneur', 'married' ,'university.degree', 'unknown', 'yes', 'no', 'telephone', 'may', 'mon', 1, 999, 0, 'nonexistent', 1.1, 93.994, -36.4, 4.857, 5191])
row_df = pd.DataFrame(np.array([[49, 'entrepreneur', 'married' ,'university.degree', 'unknown', 'yes', 'no', 'telephone', 'may', 'mon', 1, 999, 0, 'nonexistent', 1.1, 93.994, -36.4, 4.857, 5191]]), columns=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'])

row_df = pd.DataFrame(np.array([[49, 'entrepreneur', 'married' ,'university.degree', 'unknown', 'yes', 'no', 'telephone', 'may', 'mon', 1, 999, 0, 'nonexistent', 1.1, 93.994, -36.4, 4.857, 5191]]), columns=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'])

print("model score: %.3f" % clf.score(X_test, y_test))

# y_pred = clf.predict(row_df)

X_new = pd.read_csv('new-data.csv')
print('Shape of new data set', X_new.shape)

y_pred = clf.predict(X_new)

X_new.insert(19, column='pred_output', value=le.inverse_transform(y_pred))
X_new.to_csv('pred.csv')


# In[59]:


def get_pred(age, job, marital, education, default, housing, loan, contact, month, day_of_week, campaign, pdays, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed):
    row_df = pd.DataFrame(np.array([[age, job, marital, education, default, housing, loan, contact, month, day_of_week, campaign, pdays, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed]]), columns=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'])
    y_pred = clf.predict(row_df)
    
    return le.inverse_transform(y_pred)[0]


# In[60]:


pred = get_pred(49, 'entrepreneur', 'married' ,'university.degree', 'unknown', 'yes', 'no', 'telephone', 'may', 'mon', 1, 999, 0, 'nonexistent', 1.1, 93.994, -36.4, 4.857, 5191)
print (pred)

# In[ ]:




