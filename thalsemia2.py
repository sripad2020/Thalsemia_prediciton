import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('twoalphas.csv')
print(data.columns)
print(data.info())
print(data.isna().sum())
print(data.describe())
num=data.select_dtypes(include='number')
print(num.columns.values)
#for i in num.columns.values:
#    sn.boxplot(data[i])
#    plt.show()
#pcv
#neut
#lymph
#hba
#hbf
data[['hb', 'rbc', 'mch', 'mchc'
    , 'rdw' ,'wbc','plt','hba2']]=data[['hb', 'rbc',
                                        'mch','mchc', 'rdw' ,'wbc','plt'
    ,'hba2']].fillna(data[['hb', 'rbc', 'mch', 'mchc',
                 'rdw' ,'wbc','plt','hba2']].mean())
x=data[['hb', 'rbc', 'mch', 'mchc', 'rdw' ,'wbc','plt','hba2']]
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
data['label']=lab.fit_transform(data['phenotype'])
y=data['label']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
print('Logistic regression',lr.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier
tre=DecisionTreeClassifier()
tre.fit(x_train,y_train)
print('Decision tree',tre.score(x_test,y_test))
from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(x_train,y_train)
print('XGB',xgb.score(x_test,y_test))

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
print('Random forest',rf.score(x_test,y_test))

from lightgbm import LGBMClassifier
lgb=LGBMClassifier()
lgb.fit(x_train,y_train)
print('Light GBM',lgb.score(x_test,y_test))