import numpy as np
import pandas as pd
import csv

import matplotlib.pyplot as plt
import plotly.graph_objects as go

df = pd.read_csv('bank-additional-full.csv')

# sort the dataframe
df.sort_values(by='y', axis=0, inplace=True)
# set the index to be this and don't drop
df.set_index(keys=['y'], drop=False,inplace=True)
# get a list of labels
labels=df['y'].unique().tolist()

data = {}
data[labels[0]] = df.loc[df.y==labels[0]]
data[labels[1]] = df.loc[df.y==labels[1]]

job_data = {}
job_data[labels[0]] = data[labels[0]]['job'].value_counts()
job_data[labels[1]] = data[labels[1]]['job'].value_counts()

# create a figure and axis 
fig, ax = plt.subplots(figsize=(17, 5)) 

ax.bar(job_data[labels[0]].index, job_data[labels[0]].values, color='r', align='center', label=labels[0])
ax.bar(job_data[labels[1]].index, job_data[labels[1]].values, color='y', align='center', label=labels[1])
# set title and labels 
ax.set_title('Distribution') 
ax.set_xlabel('Jobs') 
ax.set_ylabel('Frequency')
ax.legend()

plt.show()

marital_data = {}
marital_data[labels[0]] = data[labels[0]]['marital'].value_counts()
marital_data[labels[1]] = data[labels[1]]['marital'].value_counts()

# create a figure and axis 
fig, ax = plt.subplots() 

ax.bar(marital_data[labels[0]].index, marital_data[labels[0]].values, color='r', align='center', label=labels[0])
ax.bar(marital_data[labels[1]].index, marital_data[labels[1]].values, color='y', align='center', label=labels[1])
# set title and labels 
ax.set_title('Distribution') 
ax.set_xlabel('Marital Status') 
ax.set_ylabel('Frequency')
ax.legend()

plt.show()

education_data = {}
education_data[labels[0]] = data[labels[0]]['education'].value_counts()
education_data[labels[1]] = data[labels[1]]['education'].value_counts()

# create a figure and axis 
fig, ax = plt.subplots(figsize=(17, 5)) 

ax.bar(education_data[labels[0]].index, education_data[labels[0]].values, color='r', align='center', label=labels[0])
ax.bar(education_data[labels[1]].index, education_data[labels[1]].values, color='y', align='center', label=labels[1])
# set title and labels 
ax.set_title('Distribution') 
ax.set_xlabel('Education Level') 
ax.set_ylabel('Frequency')
ax.legend()

plt.show()

fig = go.Figure(
    data=[go.Bar(y=education_data[labels[0]].values, x=education_data[labels[0]].index), go.Bar(y=education_data[labels[1]].values, x=education_data[labels[1]].index)],
    layout_title_text="A Figure Displayed with fig.show()"
)
fig.show()

fig = go.Figure(data=[
    go.Bar(name=labels[0], x=education_data[labels[0]].index, y=education_data[labels[0]].values),
    go.Bar(name=labels[1], x=education_data[labels[1]].index, y=education_data[labels[1]].values)
])
# Change the bar mode
fig.update_layout(barmode='stack')
fig.show()



