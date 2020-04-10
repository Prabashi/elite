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

        durationcall = df['duration']
        dayOfWeek = df['day_of_week'].unique().tolist()
        #x = {}
        #x[dayOfWeek[0]] = df.loc[df.day_of_week==dayOfWeek[0]]
        #durationcall = x[dayOfWeek[0]]['duration']

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

        default = data[labels[0]]['default'].value_counts().values.tolist()
        housing = data[labels[0]]['housing'].value_counts().values.tolist()
        loan = data[labels[0]]['loan'].value_counts().values.tolist()
        default1 = data[labels[1]]['default'].value_counts().values.tolist()
        housing1 = data[labels[1]]['housing'].value_counts().values.tolist()
        loan1 = data[labels[1]]['loan'].value_counts().values.tolist()

        tseriesweek = data[labels[0]]['day_of_week'].value_counts().values.tolist()
        tseriesweek1 = data[labels[1]]['day_of_week'].value_counts().values.tolist()
        monthser = data[labels[0]]['month'].value_counts().values.tolist()
        monthser1 = data[labels[1]]['month'].value_counts().values.tolist()

        contact = data[labels[0]]['contact'].value_counts().values.tolist()
        contact1 = data[labels[1]]['contact'].value_counts().values.tolist()
        outcome = data[labels[0]]['poutcome'].value_counts().values.tolist()
        outcome1 = data[labels[1]]['poutcome'].value_counts().values.tolist()

        durationnorm = data[labels[0]]['duration'].value_counts().values.tolist()
        durationnorm1 = data[labels[1]]['duration'].value_counts().values.tolist()

        campaign = data[labels[0]]['campaign'].value_counts().values.tolist()
        campaign1 = data[labels[1]]['campaign'].value_counts().values.tolist()
        prevcontact = data[labels[0]]['previous'].value_counts().values.tolist()
        prevcontact1 = data[labels[1]]['previous'].value_counts().values.tolist()
        pdays = data[labels[0]]['pdays'].value_counts().values.tolist()
        pdays1 = data[labels[1]]['pdays'].value_counts().values.tolist()

        labels_jobs=data[labels[0]]['job'].value_counts().index
        labels_marital=data[labels[0]]['marital'].value_counts().index
        labels_education=data[labels[0]]['education'].value_counts().index
        labels_age=data[labels[0]]['age'].value_counts().index
        labels_default=data[labels[0]]['default'].value_counts().index
        labels_housing=data[labels[0]]['housing'].value_counts().index
        labels_loan=data[labels[0]]['loan'].value_counts().index
        labels_tseriesweek=data[labels[0]]['day_of_week'].value_counts().index
        labels_durationnorm=data[labels[0]]['duration'].value_counts().index
        labels_monthser=data[labels[0]]['month'].value_counts().index
        labels_contact = data[labels[0]]['contact'].value_counts().index
        labels_campaign = data[labels[0]]['campaign'].value_counts().index
        labels_prevcontact = data[labels[0]]['previous'].value_counts().index
        labels_pdays = data[labels[0]]['pdays'].value_counts().index
        labels_outcome = data[labels[0]]['poutcome'].value_counts().index

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
                    },
                    "default": { 
                     "labels":labels_default, 
                     "chartLabel":chartLabel, 
                     "chartdata":default, 
                     "chart2Label":chart2Label, 
                     "chart2data":default1, 
                    },
                    "housing": { 
                     "labels":labels_housing, 
                     "chartLabel":chartLabel, 
                     "chartdata":housing, 
                     "chart2Label":chart2Label, 
                     "chart2data":housing1, 
                    },
                    "loan": { 
                     "labels":labels_loan, 
                     "chartLabel":chartLabel, 
                     "chartdata":loan, 
                     "chart2Label":chart2Label, 
                     "chart2data":loan1, 
                    },
                    "tseriesweek": { 
                     "labels":labels_tseriesweek, 
                     "chartLabel":chartLabel, 
                     "chartdata":tseriesweek, 
                     "chart2Label":chart2Label, 
                     "chart2data":tseriesweek1, 
                    },
                    "durationcall": { 
                     "labels":'Duration Call', 
                     "chartLabel":'duration', 
                     "chartdata":durationcall, 
                     "chart2Label":'day of week', 
                     "chart2data":dayOfWeek, 
                    },
                    "durationnorm": { 
                     "labels":labels_durationnorm, 
                     "chartLabel":chartLabel, 
                     "chartdata":durationnorm, 
                     "chart2Label":chart2Label, 
                     "chart2data":durationnorm1, 
                    },
                    "monthser": {
                     "labels":labels_monthser, 
                     "chartLabel":chartLabel, 
                     "chartdata":monthser, 
                     "chart2Label":chart2Label, 
                     "chart2data":monthser1, 
                    },
                    "contact": {
                     "labels":labels_contact, 
                     "chartLabel":chartLabel, 
                     "chartdata":contact, 
                     "chart2Label":chart2Label, 
                     "chart2data":contact1, 
                    },
                    "campaign": {
                     "labels":labels_campaign, 
                     "chartLabel":chartLabel, 
                     "chartdata":campaign, 
                     "chart2Label":chart2Label, 
                     "chart2data":campaign1, 
                    },
                    "prevcontact": {
                     "labels":labels_prevcontact, 
                     "chartLabel":chartLabel, 
                     "chartdata":prevcontact, 
                     "chart2Label":chart2Label, 
                     "chart2data":prevcontact1, 
                    },
                    "pdays": {
                     "labels":labels_pdays, 
                     "chartLabel":chartLabel, 
                     "chartdata":pdays, 
                     "chart2Label":chart2Label, 
                     "chart2data":pdays1, 
                    },
                    "outcome": {
                     "labels":labels_outcome, 
                     "chartLabel":chartLabel, 
                     "chartdata":outcome, 
                     "chart2Label":chart2Label, 
                     "chart2data":outcome1, 
                    }
                } 
        return Response(data) 