from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):

    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)

class Product_Details(models.Model):

    idno=models.CharField(max_length=300)
    ProductId=models.CharField(max_length=300)
    UserId=models.CharField(max_length=300)
    ProfileName=models.CharField(max_length=300)
    HelpfulnessNumerator=models.CharField(max_length=300)
    HelpfulnessDenominator=models.CharField(max_length=300)
    Score=models.CharField(max_length=300)
    Time=models.CharField(max_length=300)
    Summary=models.CharField(max_length=300)
    Review=models.CharField(max_length=300)

class Recommend_Prediction(models.Model):

    idno = models.CharField(max_length=300)
    ProductId = models.CharField(max_length=300)
    UserId = models.CharField(max_length=300)
    ProfileName = models.CharField(max_length=300)
    HelpfulnessNumerator = models.CharField(max_length=300)
    HelpfulnessDenominator = models.CharField(max_length=300)
    Score = models.CharField(max_length=300)
    Time = models.CharField(max_length=300)
    Summary = models.CharField(max_length=300)
    Review = models.CharField(max_length=300)
    Recommend_Prediction= models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



