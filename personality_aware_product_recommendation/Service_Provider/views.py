
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.ensemble import VotingClassifier

import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score




# Create your views here.
from Remote_User.models import ClientRegister_Model,Product_Details,Recommend_Prediction,detection_ratio,detection_accuracy


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            detection_accuracy.objects.all().delete()
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')

def Find_Recommendation_Prediction_Ratio(request):
    detection_ratio.objects.all().delete()
    ratio = ""
    kword = 'Recommend'
    print(kword)
    obj = Recommend_Prediction.objects.all().filter(Q(Recommend_Prediction=kword))
    obj1 = Recommend_Prediction.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = 'No Recommend'
    print(kword1)
    obj1 = Recommend_Prediction.objects.all().filter(Q(Recommend_Prediction=kword1))
    obj11 = Recommend_Prediction.objects.all()
    count1 = obj1.count();
    count11 = obj11.count();
    ratio1 = (count1 / count11) * 100
    if ratio1 != 0:
        detection_ratio.objects.create(names=kword1, ratio=ratio1)

    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/Find_Recommendation_Prediction_Ratio.html', {'objs': obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def ViewTrendings(request):
    topic = Recommend_Prediction.objects.values('topics').annotate(dcount=Count('topics')).order_by('-dcount')
    return  render(request,'SProvider/ViewTrendings.html',{'objects':topic})


def charts(request,chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def Predict_Recommendation_Details(request):

    obj =Recommend_Prediction.objects.all()
    return render(request, 'SProvider/Predict_Recommendation_Details.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})


def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="TrainedData.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = Recommend_Prediction.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1
        ws.write(row_num, 0, my_row.idno, font_style)
        ws.write(row_num, 1, my_row.ProductId, font_style)
        ws.write(row_num, 2, my_row.UserId, font_style)
        ws.write(row_num, 3, my_row.ProfileName, font_style)
        ws.write(row_num, 4, my_row.HelpfulnessNumerator, font_style)
        ws.write(row_num, 5, my_row.HelpfulnessDenominator, font_style)
        ws.write(row_num, 6, my_row.Score, font_style)
        ws.write(row_num, 7, my_row.Time, font_style)
        ws.write(row_num, 8, my_row.Summary, font_style)
        ws.write(row_num, 9, my_row.Review, font_style)
        ws.write(row_num, 10, my_row.Recommend_Prediction, font_style)

    wb.save(response)
    return response

def train_model(request):
    detection_accuracy.objects.all().delete()

    df = pd.read_csv('Reviews.csv')
    df
    df.columns
    df.rename(columns={'Score': 'Rating', 'Text': 'Review'}, inplace=True)

    def apply_recommend(Rating):
        if (Rating <= 2):
            return 0  # No Recommend
        else:
            return 1  # Recommend

    df['recommend'] = df['Rating'].apply(apply_recommend)
    df.drop(['Rating'], axis=1, inplace=True)
    recommend = df['recommend'].value_counts()
    df.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time',
             'Summary'], axis=1, inplace=True)

    cv = CountVectorizer()
    X = df['Review']
    y = df['recommend']
    X = cv.fit_transform(X)

    models = []
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    X_train.shape, X_test.shape, y_train.shape

    print("Naive Bayes")

    from sklearn.naive_bayes import MultinomialNB
    NB = MultinomialNB()
    NB.fit(X_train, y_train)
    predict_nb = NB.predict(X_test)
    naivebayes = accuracy_score(y_test, predict_nb) * 100
    print(naivebayes)
    print(confusion_matrix(y_test, predict_nb))
    print(classification_report(y_test, predict_nb))
    models.append(('naive_bayes', NB))

    detection_accuracy.objects.create(names="Naive Bayes", ratio=naivebayes)

    # SVM Model
    print("SVM")
    from sklearn import svm
    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train)
    predict_svm = lin_clf.predict(X_test)
    svm_acc = accuracy_score(y_test, predict_svm) * 100
    print(svm_acc)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, predict_svm))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predict_svm))
    models.append(('svm', lin_clf))

    detection_accuracy.objects.create(names="SVM", ratio=svm_acc)

    print("Logistic Regression")

    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, y_pred) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred))
    models.append(('logistic', reg))

    detection_accuracy.objects.create(names="Logistic Regression", ratio=accuracy_score(y_test, y_pred) * 100)

    classifier = VotingClassifier(models)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    predicts = 'predicts.csv'
    # df['predict_nb'] = predict_text
    df.to_csv(predicts, index=False)
    df.to_markdown

    se=''

    obj1 =Product_Details.objects.values('idno',
    'ProductId',
    'UserId',
    'ProfileName',
    'HelpfulnessNumerator',
    'HelpfulnessDenominator',
    'Score',
    'Time',
    'Summary',
    'Review')

    Recommend_Prediction.objects.all().delete()
    for t in obj1:

        idno= t['idno']
        ProductId= t['ProductId']
        UserId= t['UserId']
        ProfileName= t['ProfileName']
        HelpfulnessNumerator= t['HelpfulnessNumerator']
        HelpfulnessDenominator= t['HelpfulnessDenominator']
        Score= t['Score']
        Time= t['Time']
        Summary= t['Summary']
        Review= t['Review']

        review_data = [Review]
        vector1 = cv.transform(review_data).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if prediction == 0:
            predict = 'No Recommend'
        else:
            predict = 'Recommend'

        Recommend_Prediction.objects.create(idno=idno,
        ProductId=ProductId,
        UserId=UserId,
        ProfileName=ProfileName,
        HelpfulnessNumerator=HelpfulnessNumerator,
        HelpfulnessDenominator=HelpfulnessDenominator,
        Score=Score,
        Time=Time,
        Summary=Summary,
        Review=Review,
        Recommend_Prediction=predict
        )

    obj = detection_accuracy.objects.all()
    return render(request,'SProvider/train_model.html', {'objs': obj})














