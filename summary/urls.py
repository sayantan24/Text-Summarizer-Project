from django.urls import path
from . import views

urlpatterns=[
    path('',views.home,name='home'),
    path('result/',views.textSum,name='result'),
    
]
