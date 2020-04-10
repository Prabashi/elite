from django.shortcuts import render

# Create your views here.
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
        return render(request, 'diagnose/index.html') 

class DiagnoseData(APIView): 
    authentication_classes = [] 
    permission_classes = [] 
   
    def get(self, request, format = None): 
        df_sorted = pd.read_csv('bank-additional-full.csv')

        month_data = df_sorted[['month', 'y']]

        # month_data = pd.DataFrame(month_data)
        print (set(month_data['month']))
        print (set(month_data['y']))

        months = ['may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov']

        yes_score = []
        no_score = []
        yes_count = 0
        no_count = 0
        month = 'may'
        i = 0

        for index, row in month_data.iterrows():
            mnth = row['month']
            labl = row['y']
            month = months[i]
        #     print (month)
        #     print (mnth)
            
            if month == mnth:
                if labl == 'yes':
                    yes_count = yes_count + 1
                else:
                    no_count = no_count + 1
            else:
                i = i + 1
                yes_score.append(yes_count)
                no_score.append(no_count)
                month = months[i]
                
                while mnth != month:
                    yes_score.append(0)
                    no_score.append(0)
                    i = i + 1
                    month = months[i]
                    
                if labl == 'yes':
                    yes_count = 1
                    no_count = 0
                else:
                    yes_count = 0
                    no_count = 1
        print (yes_score)
        print (no_score)

        day_of_week_data = df_sorted[['day_of_week', 'y']]

        print (set(day_of_week_data['day_of_week']))
        print (set(day_of_week_data['y']))

        days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

        yes_scoredaily = []
        no_scoredaily = []
        yes_countdaily = 0
        no_countdaily = 0
        day = 'mon'
        i = 0
        max_i = len(days)
        print (max_i)

        for index, row in day_of_week_data.iterrows():
            dy = row['day_of_week']
            labl = row['y']
            day = days[i]
            
            if day == dy:
                if labl == 'yes':
                    yes_countdaily = yes_countdaily + 1
                else:
                    no_countdaily = no_countdaily + 1
            else:
                i = i + 1
                
                if i == max_i:
                    break
                
                yes_scoredaily.append(yes_countdaily)
                no_scoredaily.append(no_countdaily)
                day = days[i]
                
                while dy != day:
                    yes_scoredaily.append(0)
                    no_scoredaily.append(0)
                    i = i + 1
                    
                    if i == max_i:
                        break
                    
                    day = days[i]
                    
                if i == max_i:
                    break
                    
                if labl == 'yes':
                    yes_countdaily = 1
                    no_countdaily = 0
                else:
                    yes_countdaily = 0
                    no_countdaily = 1
        print (yes_scoredaily)
        print (no_scoredaily)

        data = {
                "tseriesobj": {
                 "labels":months, 
                 "chartLabel":'yes', 
                 "chartdata":yes_score, 
                 "chart2Label":'no', 
                 "chart2data":no_score, 
                },"tseriesobjdaily": {
                 "labels":days, 
                 "chartLabel":'yes', 
                 "chartdata":yes_scoredaily, 
                 "chart2Label":'no', 
                 "chart2data":no_scoredaily, 
                }};
        return Response(data) 