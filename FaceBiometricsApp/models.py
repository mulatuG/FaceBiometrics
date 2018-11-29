from django.db import models

# Create your models here.
class Customer(models.Model):
    name = models.CharField(max_length=200)
    surname =models.CharField(max_length=200)
    father_name=models.CharField(max_length=200)
    date_of_birth = models.DateTimeField('Date of Birth')
    place_of_birth=models.CharField(max_length=1000)
    nationality=models.CharField(max_length=100)
    SEX = (
        ('M', 'Male'),
        ('F', 'Female'),
    )
    sex=models.CharField(max_length=5,choices=SEX)
    passport_number_or_residence_id=models.CharField(max_length=100,unique=True)
    verbose_name = 'Customer Summary'
    verbose_name_plural = 'Customers Summary'
    def __str__(self):
        return '%s %s %s' % (self.passport_number_or_residence_id+' :', self.name, self.father_name)

class PCCApplication(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    
    reason_for_request=models.TextField(max_length=2000)
    clearance_requested_by=models.TextField(max_length=1000)
    reference_number=models.CharField(max_length=1000)
    date = models.DateTimeField()
    date.help_text="Please use the following format: <em>MM/DD/YYYY</em>."
    verbose_name = 'PCC Application'
    verbose_name_plural = 'PCC Application summary'
    def __str__(self):
        return '%s %s' % (self.customer, '( '+self.clearance_requested_by+' )')

class Crime(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    
    crime_title=models.TextField(max_length=1000)
    crime_description=models.TextField()
    date_of_crime_happened = models.DateTimeField()
    verbose_name='List of Crime'
    verbose_name_plural='List of Crimes'
    def __str__(self):
        return  self.crime_title


