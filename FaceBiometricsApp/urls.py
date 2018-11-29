from django.urls import path
from django.contrib import admin
from . import views
app_name = 'FaceRecognitionApp'

urlpatterns = [
    path('', views.index, name='index'),
    path('newCustomer/', views.newCustomer, name='newCustomer'),
    path('detail/<str:passport_number_or_residence_id>', views.detail, name='detail'),
    path('add_customer', views.add_customer, name='add_customer'),
    path('pcc_application', views.pcc_application, name='pcc_application'),
    path('new_crime', views.new_crime, name='new_crime'),
    path('MachineTrainUsingEigenFaces', views.MachineTrainUsingEigenFaces, name='MachineTrainUsingEigenFaces'),
    path('FaceRecognitionLogic', views.FaceRecognitionLogic,name='recognition'),
    path('customer_list', views.customer_list, name='customer_list'),
    path('<int:customer_id/results',views.results, name='results'),
    path('help', views.help, name='help'),
    path('error', views.error, name='error'),
   
]
admin.site.site_title = 'Ethiopian Police Clearnace Certificate Using FaceBiometrics'