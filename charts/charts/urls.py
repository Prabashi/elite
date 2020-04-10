"""charts URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from chartjs import views as chartjsviews
from predictLoan import views as predictviews
from diagnose import views as diagnoseviews

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', chartjsviews.HomeView.as_view()),
    path('predict/', predictviews.HomeView.as_view()), 
    path('diagnose/', diagnoseviews.HomeView.as_view()), 
    path('aboutus/', chartjsviews.AboutUsView.as_view()), 
    # path('test-api', views.get_data), 
    path('api', chartjsviews.ChartData.as_view()),
    path('predictapi', predictviews.PredictData.as_view()),
    path('diagnoseapi', diagnoseviews.DiagnoseData.as_view())
]
