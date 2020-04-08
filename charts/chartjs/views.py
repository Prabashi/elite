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

class AboutUsView(View): 
    def get(self, request, *args, **kwargs): 
        return render(request, 'aboutus/index.html') 
   
   
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

        job = data[labels[0]]['job'].value_counts().values.tolist()
        job1 = data[labels[1]]['job'].value_counts().values.tolist()
        marital = data[labels[0]]['marital'].value_counts().values.tolist()
        marital1 = data[labels[1]]['marital'].value_counts().values.tolist()
        education = data[labels[0]]['education'].value_counts().values.tolist()
        education1 = data[labels[1]]['education'].value_counts().values.tolist()
        age = data[labels[0]]['age'].value_counts().values.tolist()
        age1 = data[labels[1]]['age'].value_counts().values.tolist()

        job_data = {}
        job_data[labels[0]] = data[labels[0]]['job'].value_counts()
        job_data[labels[1]] = data[labels[1]]['job'].value_counts()

        marital_data = {}
        marital_data[labels[0]] = data[labels[0]]['marital'].value_counts()
        marital_data[labels[1]] = data[labels[1]]['marital'].value_counts()

        education_data = {}
        education_data[labels[0]] = data[labels[0]]['education'].value_counts()
        education_data[labels[1]] = data[labels[1]]['education'].value_counts()

        age_data = {}
        age_data[labels[0]] = data[labels[0]]['age'].value_counts()
        age_data[labels[1]] = data[labels[1]]['age'].value_counts()

        labels_jobs=job_data[labels[0]].index
        labels_marital=marital_data[labels[0]].index
        labels_education=education_data[labels[0]].index
        labels_age=age_data[labels[0]].index

        chartLabel = df['y'].unique().tolist()[0]
        chart2Label = df['y'].unique().tolist()[1]

        data =  {
                    "job": { 
                     "labels":labels_jobs, 
                     "chartLabel":chartLabel, 
                     "chartdata":job, 
                     "chart2Label":chart2Label, 
                     "chart2data":job1, 
                    },
                    "marital": { 
                     "labels":labels_marital, 
                     "chartLabel":chartLabel, 
                     "chartdata":marital, 
                     "chart2Label":chart2Label, 
                     "chart2data":marital1, 
                    },
                    "education": { 
                     "labels":labels_education, 
                     "chartLabel":chartLabel, 
                     "chartdata":education, 
                     "chart2Label":chart2Label, 
                     "chart2data":education1, 
                    },
                    "age": { 
                     "labels":labels_age, 
                     "chartLabel":chartLabel, 
                     "chartdata":age, 
                     "chart2Label":chart2Label, 
                     "chart2data":age1, 
                    }
                } 
        return Response(data) 