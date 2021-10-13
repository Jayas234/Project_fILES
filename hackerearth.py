# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 13:02:06 2021

@author: 91986
"""
import pandas
#import numpy
#import matplotlib.pyplot as plt

#names=[i for i in 'abcdefghijklmno']

data=pandas.read_csv(r'C:\Users\91986\Downloads\New folder (4)\dataset\test.csv',names=names)
#print(data)


from sklearn.linear_model import LogisticRegression
modellr=LogisticRegression()
modellr.fit(xtrain,ytrain)
ypredlr=modellr.predict(xtest)
print(ypredlr)

print("Logistic-->",accuracy_score(ytest,ypredlr)*100)