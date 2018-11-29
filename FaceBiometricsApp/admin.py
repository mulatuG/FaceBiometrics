from django.contrib import admin
from django.contrib.admin import AdminSite
from django.http import HttpResponseRedirect
# Register your models here.

from .models import Customer, PCCApplication, Crime
# Define the admin class
class CustomerAdmin(admin.ModelAdmin):
    #change_form_template = "FaceRecognition/custom_change_template.html"

    list_display =  ('name','surname','father_name','date_of_birth','place_of_birth','nationality','sex','passport_number_or_residence_id')
    search_fields = ['name', 'surname', 'passport_number_or_residence_id']
    list_filter =  ('name','surname','father_name','place_of_birth','nationality','sex','passport_number_or_residence_id')
    fields = [('name','surname'),('father_name','sex'),'date_of_birth',('place_of_birth','nationality'),'passport_number_or_residence_id']
   
    #def __str__(self):
       # return 'Customers: {} {} {} {} {} {} {} {} {} {} {} {}'.format(self.name, self.surname,self.father_name,self.date_of_birth, self.place_of_birth,self.nationality, self.sex,self.reason_for_request, self.clearance_requested_by,self.reference_number, self.date,self.passport_number_or_residence_id)
# Register the admin class with the associated model
admin.site.register(Customer, CustomerAdmin)

class PCCApplicationAdmin(admin.ModelAdmin):
     """docstring for ClassName"""
     search_fields = ['customer', 'reason_for_request', 'clearance_requested_by','reference_number']
     list_filter =  ('customer', 'reason_for_request', 'clearance_requested_by','reference_number')
    


admin.site.register(PCCApplication, PCCApplicationAdmin)

class CrimeAdmin(admin.ModelAdmin):
    list_display=('customer','crime_title','date_of_crime_happened','crime_description')
    list_filter=('customer','crime_title','date_of_crime_happened','crime_description')
    search_fields=['crime_title','date_of_crime_happened','crime_description']
    """docstring for ClassName"""

admin.site.register(Crime, CrimeAdmin)     

AdminSite.site_title = 'Administrator'
AdminSite.site_header = 'Improved Ethiopian Police Clearnace Certificate Prototype Using Facebiometrics'

AdminSite.index_title = 'Admin : Data Base Administration'
AdminSite.site_url='/FaceBiometricsApp/'