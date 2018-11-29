from django.forms import ModelForm
from .models import Customer, PCCApplication, Crime

from django import forms


class CustomerForm2(ModelForm):
    class Meta:
        model = Customer
        fields = ['name','surname','father_name','date_of_birth','place_of_birth','nationality','sex','passport_number_or_residence_id']
       
    def __init__(self, *args, **kwargs):
        super(CustomerForm2, self).__init__(*args, **kwargs)
        self.fields['name'].widget = forms.TextInput(attrs={'class': 'form-control'})
        self.fields['surname'].widget = forms.TextInput(attrs={'class': 'form-control'})
        self.fields['father_name'].widget = forms.TextInput(attrs={'class': 'form-control'})
        self.fields['date_of_birth'].widget = forms.TextInput(attrs={'class': 'form-control'})
        self.fields['place_of_birth'].widget = forms.TextInput(attrs={'class': 'form-control'})
        self.fields['nationality'].widget = forms.TextInput(attrs={'class': 'form-control'})
        self.fields['passport_number_or_residence_id'].widget = forms.TextInput(attrs={'class': 'form-control'})
        
class PCCApplicationForm(ModelForm):
    class Meta:
        model=PCCApplication
        applicant = forms.ModelChoiceField(queryset=Customer.objects.all())
        #fields=['reason_for_request','clearance_requested_by','reference_number','date']
        fields = '__all__'
    def __str__(self, *args, **kwargs):
        super(PCCApplicationForm, self).__init__(*args, **kwargs)
        self.fields['reason_for_request'].widget = forms.Textarea(attrs={'class': 'form-control','rows':'2'})
        self.fields['clearance_requested_by'].widget = forms.Textarea(attrs={'class': 'form-control','rows':'2'})
        self.fields['reference_number'].widget = forms.TextInput(attrs={'class': 'form-control'})
        self.fields['date'].widget = forms.DateInput(attrs={'class': 'form-control'})
        
class CrimeForm(ModelForm):
    class Meta:
        model=Crime
        applicant = forms.ModelChoiceField(queryset=Customer.objects.all())
        #fields=['reason_for_request','clearance_requested_by','reference_number','date']
        fields = '__all__'
    def __str__(self, *args, **kwargs):
        super(PCCApplicationForm, self).__init__(*args, **kwargs)
        self.fields['crime_title'].widget = forms.Textarea(attrs={'class': 'form-control','rows':'2'})
        self.fields['crime_description'].widget = forms.Textarea(attrs={'class': 'form-control','rows':'2'})
        self.fields['date_of_crime_happened'].widget = forms.DateInput(attrs={'class': 'form-control'})
        
       
