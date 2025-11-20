from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),                  # Home page
    path('dashboard/', views.dashboard, name='dashboard'), # Dashboard page
    path('simulator/', views.simulator, name='simulator'),
    path('index1/', views.index1, name='index1'),
    path('predict/', views.predict_anomaly, name='predict-anomaly'),
    path("process_npy/", views.process_npy, name="process_npy")
]
