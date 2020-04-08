# from django.http import JsonResponse 
   
from django.shortcuts import render 
from django.views.generic import View 
   
from rest_framework.views import APIView 
from rest_framework.response import Response

import numpy as np
import pandas as pd
import csv

import matplotlib.pyplot as plt
import plotly.graph_objects as go
   
class HomeView(View): 
    def get(self, request, *args, **kwargs): 
        return render(request, 'chartjs/index.html') 
   
   
#################################################### 
   
## if you don't want to user rest_framework 
   
# def get_data(request, *args, **kwargs): 
# 
# data ={ 
#             "sales" : 100, 
#             "person": 10000, 
#     } 
# 
# return JsonResponse(data) # http response 
   
   
####################################################### 
   
## using rest_framework classes 
   
class ChartData(APIView): 
    authentication_classes = [] 
    permission_classes = [] 
   
    def get(self, request, format = None): 
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
        x = data[labels[0]]['job'].value_counts().values.tolist()
        x1 = data[labels[1]]['job'].value_counts().values.tolist()

        job_data = {}
        job_data[labels[0]] = data[labels[0]]['job'].value_counts()
        job_data[labels[1]] = data[labels[1]]['job'].value_counts()

        labels=job_data[labels[0]].index
        chartLabel = df['y'].unique().tolist()[0]
        chartdata = x
        chart2Label = df['y'].unique().tolist()[1]
        chart2data = x1
        data =  { 
                 "labels":labels, 
                 "chartLabel":chartLabel, 
                 "chartdata":chartdata, 
                 "chart2Label":chart2Label, 
                 "chart2data":chart2data, 
                } 
        return Response(data) 