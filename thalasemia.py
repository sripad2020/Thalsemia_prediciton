import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data=pd.read_csv('alphanorm.csv')
print(data)
print(data.columns)
print(data.describe())
print(data.info())
num=data.select_dtypes(include='number')
q1=data.mchc.quantile(0.25)
q3=data.mchc.quantile(0.75)
iqr=q3-q1
upp=q3+1.5*iqr
low=q1-1.5*iqr
df=data[(data.mchc < upp)&(data.mchc >low)]

qu1=df.rdw.quantile(0.25)
qu3=df.rdw.quantile(0.75)
iqr_=qu3-qu1
uppe=qu3+1.5*iqr_
lowe=qu1-1.5*iqr_
df=df[(df.rdw < uppe)&(df.rdw >lowe)]


for i in num.columns.values:
    sn.boxplot(df[i])
    plt.show()

from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
df['Sex']=lab.fit_transform(df['sex'])
df['type']=lab.fit_transform(df['phenotype'])

df[['hb', 'pcv', 'rbc', 'mcv', 'mch', 'mchc',
    'rdw', 'wbc', 'plt',
    'hba']]=df[['hb', 'pcv', 'rbc', 'mcv', 'mch',
                'mchc', 'rdw', 'wbc', 'plt',
                'hba']].fillna(df[['hb', 'pcv', 'rbc', 'mcv',
                                   'mch', 'mchc', 'rdw', 'wbc',
                                   'plt', 'hba']].median())
for i in num.columns.values:
    for j in num.columns.values:
        plt.plot(df[i],maker='o',color='red',label=f'{i}')
        plt.plot(df[j],marker='x',color='blue',label=f'{j}')
        plt.title(f"{i} vs {j}")
        plt.legend()
        plt.show()
for i in num.columns.values:
    if len(df[i].value_counts()) <=5:
        sn.countplot(df[i])
        plt.show()
plt.figure(figsize=(17,6))
corr = df.corr(method='kendall')
my_m=np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()

x=df[['hb', 'pcv', 'rbc', 'mcv', 'mch', 'mchc', 'rdw', 'wbc', 'plt', 'hba']]
y=df['type']
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