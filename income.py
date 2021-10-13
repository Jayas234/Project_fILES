import pandas
#import numpy
#import matplotlib.pyplot as plt

names=[i for i in 'abcdefghijklmno']

data=pandas.read_csv(r'C:\Users\91986\Desktop\data.csv',names=names)
#print(data)
#print(data.shape)  #says about the number of rows and coloums in the data
#print(data.size) #says about the complete data
#print(data.head())

data['b']=data['b'].replace(to_replace=' ?',value=' Private')
data['g']=data['g'].replace(to_replace=' ?',value=' Sales')
data['n']=data['n'].replace(to_replace=' ?',value=' Mexico')

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for z in 'bdfghijn':
    data[z]=le.fit_transform(data[z])#.values converts data into matrix in numpy

x=data.iloc[:,:-1].values #iloc integer locations  except last col remaining save in x
y=data.iloc[:,-1].values    # last  position data will be stored in y and we need to predict that
#print(x)
#print(y)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2 ,random_state=9)#20 % of data is used for training
#print(xtrain)
#print(xtrain.shape)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
print(ypred)

from sklearn.naive_bayes import GaussianNB
modelnb=GaussianNB()
modelnb.fit(xtrain,ytrain)
yprednb=modelnb.predict(xtest)
print(yprednb)

from sklearn.tree import DecisionTreeClassifier
modeldt=DecisionTreeClassifier(criterion='entropy',random_state=9)#gini
modeldt.fit(xtrain,ytrain)
ypreddt=modeldt.predict(xtest)
print(ypreddt)

from sklearn.ensemble import RandomForestClassifier
modelrf=RandomForestClassifier(criterion='entropy',random_state=9)#gini
modelrf.fit(xtrain,ytrain)
ypredrf=modelrf.predict(xtest)
print(ypredrf)

from sklearn.linear_model import LogisticRegression
modellr=LogisticRegression()
modellr.fit(xtrain,ytrain)
ypredlr=modellr.predict(xtest)
print(ypredlr)

'''from sklearn.svm import SVC
modelsvm=SVC(kernal='linear')
modelsvm.fit(xtrain,ytrain)
ypredsvm=modelsvm.predict(xtest)
print(ypredsvm)'''




from sklearn.metrics import accuracy_score
print("KNN-->",accuracy_score(ytest,ypred)*100)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest,ypred))

#print(model.predict([[1.3,3.2,4.3,2.1]]))

print("Naive-->",accuracy_score(ytest,yprednb)*100)
print("Decision-->",accuracy_score(ytest,ypreddt)*100)
print("Random_forest-->",accuracy_score(ytest,ypredrf)*100)
print("Logistic-->",accuracy_score(ytest,ypredlr)*100)
#print("SVM-->",accuracy_score(ytest,ypredsvm)*100)

