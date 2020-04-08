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
print ((((df.loc[df.y==labels[0]])['job']).value_counts()).values.tolist())
print (job_data[labels[0]].index)
print (job_data[labels[0]])
# set title and labels 