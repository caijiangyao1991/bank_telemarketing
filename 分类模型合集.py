# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 23:17:31 2016

@author: CJY
"""

import os
import sys
import string
import numpy as np
import pandas as pd
from pandas import DataFrame ,Series
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn.cross_validation as cross_validation
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear_model
import sklearn.svm as svm
import sklearn.feature_selection as feature_selection
import sklearn.metrics as metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
model_data=pd.read_csv("E:/pythonWK/CDAII/课程-就业班/答辩/bank-additional-full.csv",sep=';')
model_data=model_data.drop_duplicates()
df_train, df_test=cross_validation.train_test_split(model_data,test_size=0.2,random_state=0)
#缺失值填补
df_train['marital'].replace('unknown','married',inplace=True)
df_train['job'].replace('unknown','technician',inplace=True)
#default ，education合并水平数
df_train['default'].replace('yes','unknown',inplace=True)
df_train['education'].replace('illiterate','unknown',inplace=True)
#去掉duration
df_train=df_train.drop('duration',axis='columns')
#生成一个新的变量“newpday”，之前联系过得客户取值为1，否则为0。而之前的999用除99外的最大值替代
df_train['pdays_new']=df_train['pdays']
df_train['pdays_new'][df_train['pdays_new']!=999]=1
df_train['pdays_new'][df_train['pdays_new']==999]=0

df_train['pdays'].replace(999,0).max()
df_train['pdays'].replace(999,28,inplace=True)
#将age进行离散化处理
bins=[16,25,35,60,100]
df_train.loc[:,'ageseg']=pd.cut(df_train.loc[:,'age'],bins)
#去掉原来的age
df_train.drop('age',axis='columns',inplace=True)
df_train=df_train.copy()
df_train.loc[:,('job')]=LabelEncoder().fit_transform(df_train['job'])
df_train.loc[:,('marital')]=LabelEncoder().fit_transform(df_train['marital'])
df_train.loc[:,('education')]=LabelEncoder().fit_transform(df_train['education'])
df_train.loc[:,('housing')]=LabelEncoder().fit_transform(df_train['housing'])
df_train.loc[:,('loan')]=LabelEncoder().fit_transform(df_train['loan'])
df_train.loc[:,('contact')]=LabelEncoder().fit_transform(df_train['contact'])
df_train.loc[:,('month')]=LabelEncoder().fit_transform(df_train['month'])
df_train.loc[:,('default')]=LabelEncoder().fit_transform(df_train['default'])
df_train.loc[:,('day_of_week')]=LabelEncoder().fit_transform(df_train['day_of_week'])
df_train.loc[:,('poutcome')]=LabelEncoder().fit_transform(df_train['poutcome'])
df_train.loc[:,('ageseg')]=LabelEncoder().fit_transform(df_train['ageseg'])
df_train.loc[:,('y')]=LabelEncoder().fit_transform(df_train['y'])
#生成一个euribor3和cons.confi的交互项放入模型
df_train['eur_conf']=df_train['euribor3m']*df_train['cons.conf.idx']
df_train=df_train.drop(['euribor3m','cons.conf.idx'],axis=1)
df_test['marital'].replace('unknown','married',inplace=True)
df_test['job'].replace('unknown','technician',inplace=True)
#default ，education合并水平数
df_test['default'].replace('yes','unknown',inplace=True)
df_test['education'].replace('illiterate','unknown',inplace=True)
#去掉duration
df_test=df_test.drop('duration',axis='columns')
df_test['pdays_new']=df_test['pdays']
df_test['pdays_new'][df_test['pdays_new']!=999]=1
df_test['pdays_new'][df_test['pdays_new']==999]=0
df_test['pdays'].replace(999,0).max()
df_test['pdays'].replace(999,28,inplace=True)
#将age进行离散化处理
bins=[16,25,35,60,100]
df_test.loc[:,'ageseg']=pd.cut(df_test.loc[:,'age'],bins)
#去掉原来的age
df_test.drop('age',axis='columns',inplace=True)
df_test=df_test.copy()
df_test.loc[:,('job')]=LabelEncoder().fit_transform(df_test['job'])
df_test.loc[:,('marital')]=LabelEncoder().fit_transform(df_test['marital'])
df_test.loc[:,('education')]=LabelEncoder().fit_transform(df_test['education'])
df_test.loc[:,('housing')]=LabelEncoder().fit_transform(df_test['housing'])
df_test.loc[:,('loan')]=LabelEncoder().fit_transform(df_test['loan'])
df_test.loc[:,('contact')]=LabelEncoder().fit_transform(df_test['contact'])
df_test.loc[:,('month')]=LabelEncoder().fit_transform(df_test['month'])
df_test.loc[:,('default')]=LabelEncoder().fit_transform(df_test['default'])
df_test.loc[:,('day_of_week')]=LabelEncoder().fit_transform(df_test['day_of_week'])
df_test.loc[:,('poutcome')]=LabelEncoder().fit_transform(df_test['poutcome'])
df_test.loc[:,('ageseg')]=LabelEncoder().fit_transform(df_test['ageseg'])
df_test.loc[:,('y')]=LabelEncoder().fit_transform(df_test['y'])
#生成一个euribor3和cons.confi的交互项放入模型
df_test['eur_conf']=df_test['euribor3m']*df_test['cons.conf.idx']
df_test=df_test.drop(['euribor3m','cons.conf.idx'],axis=1)
#变量筛选
# corr_matrix =df_train.corr(method='pearson')
# corr_matrix = corr_matrix.abs()
# sns.set(rc={"figure.figsize": (10, 10)})
# sns.heatmap(corr_matrix,square=True,cmap="Blues")
# corr = df_train.corr(method='pearson').ix["y"].abs()
# corr.sort(ascending=False)
# corr.plot(kind="bar",title="corr",figsize=[12,6])
# sampler = np.random.randint(0,len(model_data),size=50)#随机数列表
# clustertable=model_data[['emp.var.rate','cons.price.idx','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','pdays','previous','campaign']]#选取连续型变量
# sns.clustermap(clustertable.iloc[sampler].T, col_cluster=False, row_cluster=True)#抽样并进行变量聚类
# import numpy as np
# import pandas as pd
# from scipy import stats, integrate
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(color_codes=True)

# df = df_train[['emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','y']]

# sns.pairplot(df,hue='y')
# corr = df_train.corr(method='spearman').ix["y"].abs()
# corr.sort(ascending=False)
# corr.plot(kind="bar",title="corr",figsize=[12,6])
# y=df_train['y']
# x=df_train.drop('y',axis='columns')
# rfc = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=3, max_features=0.5, min_samples_split=5)
# rfc_model = rfc.fit(x, y)
# rfc_model.feature_importances_
# rfc_fi = pd.DataFrame()
# rfc_fi["features"] = list(x.columns)
# rfc_fi["importance"] = list(rfc_model.feature_importances_)
# rfc_fi=rfc_fi.set_index("features",drop=True)
# rfc_fi.sort_index(by="importance",ascending=False).plot(kind="bar",title="randomforest",figsize=[12,6])
# def IV_between(y, x): # y and x are pd.series type
#     all_i = y.groupby(x).count()
#     bad_i = y.groupby(x).sum() # Assume: 1 indicate bad, 0 indicate good
#     good_i = all_i - bad_i
#     p1 = bad_i / bad_i.sum()
#     p0 = good_i / good_i.sum()
#     woe = np.log(p1 / p0)
#     IV = (p1 - p0) * woe
#     return IV.sum()

# IV = pd.Series()
# for i in x.columns:    
#     if len(x[i].unique()) > 10 and x[i].dtype != np.object:
#         try:
#             tmp = pd.qcut(x[i], 5)
#         except:
#             tmp = pd.cut(x[i], 5)
#         IV = IV.append(pd.Series([i], index=[IV_between(y, tmp)]))
#     else:
#         IV = IV.append(pd.Series([i], index=[IV_between(y, x[i])]))
# IV.sort_index(ascending=False)

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2, f_classif

# f_selector = SelectKBest(f_classif, k=10)
# f_selector.fit(x, y)
# x.columns[f_selector.get_support()]


#分出x,y
y_train=df_train['y'].as_matrix()
x_train=df_train.drop('y',axis='columns').as_matrix()
y_test=df_test['y'].as_matrix()
x_test=df_test.drop('y',axis='columns').as_matrix()
#用决策树建模
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=5)
clf_fit = clf.fit(x_train, y_train)
test_est_tr = clf.predict(x_test)
train_est_tr = clf.predict(x_train)
test_est_p_tr = clf.predict_proba(x_test)[:,1]
train_est_p_tr = clf.predict_proba(x_train)[:,1]

fpr_test_tr, tpr_test_tr, th_test_tr=metrics.roc_curve(y_test,test_est_p_tr,pos_label=1)
fpr_train_tr, tpr_train_tr, th_train_tr=metrics.roc_curve(y_train,train_est_p_tr,pos_label=1)

#用神经网络建模
from sklearn.neural_network import MLPClassifier
ann=MLPClassifier(activation='relu',hidden_layer_sizes=(12, ),random_state=None)
ann_fit = ann.fit(x_train, y_train)
test_est_ann = ann.predict(x_test)
train_est_ann = ann.predict(x_train)
test_est_p_ann = ann.predict_proba(x_test)[:,1]
train_est_p_ann = ann.predict_proba(x_train)[:,1]

fpr_test_ann, tpr_test_ann, th_test_ann=metrics.roc_curve(y_test,test_est_p_ann,pos_label=1)
fpr_train_ann, tpr_train_ann, th_train_ann=metrics.roc_curve(y_train,train_est_p_ann,pos_label=1)

#用svm创建模型
from sklearn import svm
svm=svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=True, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
svm.fit(x_train,y_train)
test_est_svm = svm.predict(x_test)
train_est_svm = svm.predict(x_train)
test_est_p_svm = svm.predict_proba(x_test)[:,1]
train_est_p_svm = svm.predict_proba(x_train)[:,1]
fpr_test_svm, tpr_test_svm, th_test_svm=metrics.roc_curve(y_test,test_est_p_svm,pos_label=1)
fpr_train_svm, tpr_train_svm, th_train_svm=metrics.roc_curve(y_train,train_est_p_svm,pos_label=1)


#用随机森林建模
import json
from operator import itemgetter
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split,StratifiedShuffleSplit,StratifiedKFold
clf=RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=5, min_samples_split=1,
  min_samples_leaf=1, max_features='auto',    bootstrap=False, oob_score=False, n_jobs=1, random_state=0,
  verbose=0)
param_grid = dict( )
##创建分类pipeline(不同参数情况下的交叉验证)
pipeline=Pipeline([ ('clf',clf) ])
grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=3,scoring='accuracy',\
cv=StratifiedShuffleSplit(y_train, n_iter=10, test_size=0.2, train_size=None,  \
random_state=0)).fit(x_train, y_train)  
test_est_ra = grid_search.best_estimator_.predict(x_test)
train_est_ra = grid_search.best_estimator_.predict(x_train)
test_est_p_ra = grid_search.best_estimator_.predict_proba(x_test)[:,1]
train_est_p_ra = grid_search.best_estimator_.predict_proba(x_train)[:,1]
fpr_test_ra, tpr_test_ra, th_test_ra=metrics.roc_curve(y_test,test_est_p_ra,pos_label=1)
fpr_train_ra, tpr_train_ra, th_train_ra=metrics.roc_curve(y_train,train_est_p_ra,pos_label=1)

#逻辑回归建模
model_data=pd.read_csv("E:/pythonWK/CDAII/课程-就业班/答辩/bank-additional-full.csv",sep=';')
model_data=model_data.drop_duplicates()
a=model_data[model_data['y']=='yes']
b=model_data[model_data['y']=='no']
split_train,split_test=cross_validation.train_test_split(b,test_size=0.3,random_state=0)
df_train=pd.concat([a,split_train])
df_test=pd.concat([a,split_test])
#缺失值填补
df_train['marital'].replace('unknown','married',inplace=True)
df_train['job'].replace('unknown','technician',inplace=True)
#default ，education合并水平数
df_train['default'].replace('yes','unknown',inplace=True)
df_train['education'].replace('illiterate','unknown',inplace=True)
#去掉duration
df_train=df_train.drop('duration',axis='columns')
#生成一个新的变量“newpday”，之前联系过得客户取值为1，否则为0。而之前的999用除99外的最大值替代
df_train['pdays_new']=df_train['pdays']
df_train['pdays_new'][df_train['pdays_new']!=999]=1
df_train['pdays_new'][df_train['pdays_new']==999]=0

df_train['pdays'].replace(999,0).max()
df_train['pdays'].replace(999,28,inplace=True)
#将age进行离散化处理
bins=[16,25,35,60,100]
df_train.loc[:,'ageseg']=pd.cut(df_train.loc[:,'age'],bins)
#去掉原来的age
df_train.drop('age',axis='columns',inplace=True)
#变成虚拟变量
dummies_job=pd.get_dummies(df_train['job'],prefix='job')
dummies_marital=pd.get_dummies(df_train['marital'],prefix='marital')
dummies_education=pd.get_dummies(df_train['education'],prefix='education')
dummies_default=pd.get_dummies(df_train['default'],prefix='default')
dummies_housing=pd.get_dummies(df_train['housing'],prefix='housing')
dummies_loan=pd.get_dummies(df_train['loan'],prefix='loan')
dummies_contact=pd.get_dummies(df_train['contact'],prefix='contact')
dummies_month=pd.get_dummies(df_train['month'],prefix='month')
dummies_day_of_week=pd.get_dummies(df_train['day_of_week'],prefix='day_of_week')
dummies_poutcome=pd.get_dummies(df_train['poutcome'],prefix='poutcome')
dummies_ageseg=pd.get_dummies(df_train['ageseg'],prefix='ageseg')
dummies_y=pd.get_dummies(df_train['y'],prefix='y')
df_train=pd.concat([df_train,dummies_job,dummies_marital,dummies_education,dummies_default,dummies_housing,dummies_loan,
             dummies_contact,dummies_month,dummies_day_of_week,dummies_poutcome,dummies_ageseg,dummies_y],axis=1)
df_train.drop(['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','ageseg','y','y_no'],axis=1,inplace=True)
df_train.rename(columns={'y_yes':'y'},inplace=True)
from sklearn.preprocessing import MinMaxScaler
from sklearn_pandas import DataFrameMapper
mapper=DataFrameMapper([
        (['campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed'],MinMaxScaler())
        ])
mapper.fit(df_train)
df_train.loc[:,['campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']]=mapper.transform(df_train)
#生成一个euribor3和cons.confi的交互项放入模型
df_train['eur_conf']=df_train['euribor3m']*df_train['cons.conf.idx']
df_train=df_train.drop(['euribor3m','cons.conf.idx'],axis=1)
df_test['marital'].replace('unknown','married',inplace=True)
df_test['job'].replace('unknown','technician',inplace=True)
#default ，education合并水平数
df_test['default'].replace('yes','unknown',inplace=True)
df_test['education'].replace('illiterate','unknown',inplace=True)
#去掉duration
df_test=df_test.drop('duration',axis='columns')
df_test['pdays_new']=df_test['pdays']
df_test['pdays_new'][df_test['pdays_new']!=999]=1
df_test['pdays_new'][df_test['pdays_new']==999]=0

df_test['pdays'].replace(999,0).max()
df_test['pdays'].replace(999,28,inplace=True)
#将age进行离散化处理
bins=[16,25,35,60,100]
df_test.loc[:,'ageseg']=pd.cut(df_test.loc[:,'age'],bins)
#去掉原来的age
df_test.drop('age',axis='columns',inplace=True)
#变成虚拟变量
dummies_job=pd.get_dummies(df_test['job'],prefix='job')
dummies_marital=pd.get_dummies(df_test['marital'],prefix='marital')
dummies_education=pd.get_dummies(df_test['education'],prefix='education')
dummies_default=pd.get_dummies(df_test['default'],prefix='default')
dummies_housing=pd.get_dummies(df_test['housing'],prefix='housing')
dummies_loan=pd.get_dummies(df_test['loan'],prefix='loan')
dummies_contact=pd.get_dummies(df_test['contact'],prefix='contact')
dummies_month=pd.get_dummies(df_test['month'],prefix='month')
dummies_day_of_week=pd.get_dummies(df_test['day_of_week'],prefix='day_of_week')
dummies_poutcome=pd.get_dummies(df_test['poutcome'],prefix='poutcome')
dummies_ageseg=pd.get_dummies(df_test['ageseg'],prefix='ageseg')
dummies_y=pd.get_dummies(df_test['y'],prefix='y')
df_test=pd.concat([df_test,dummies_job,dummies_marital,dummies_education,dummies_default,dummies_housing,dummies_loan,
             dummies_contact,dummies_month,dummies_day_of_week,dummies_poutcome,dummies_ageseg,dummies_y],axis=1)
df_test.drop(['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','ageseg','y','y_no'],axis=1,inplace=True)
df_test.rename(columns={'y_yes':'y'},inplace=True)
df_test.loc[:,['campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']]=mapper.transform(df_test)
#生成一个euribor3和cons.confi的交互项放入模型
df_test['eur_conf']=df_test['euribor3m']*df_test['cons.conf.idx']
df_test=df_test.drop(['euribor3m','cons.conf.idx'],axis=1)
#生成逻辑回归模型
clf=linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
y_train=df_train.as_matrix()[:,-2]
x_train=df_train.drop('y',axis=1).as_matrix()
clf.fit(x_train,y_train)
x_test=df_test.drop('y',axis=1).as_matrix()
y_test=df_test.as_matrix()[:,-2]
train_est = clf.predict(x_train)
test_est = clf.predict(x_test)
train_est_p = clf.predict_proba(x_train)[:,1]
test_est_p = clf.predict_proba(x_test)[:,1]
fpr_test, tpr_test, th_test = metrics.roc_curve(y_test, test_est_p)
fpr_train, tpr_train, th_train = metrics.roc_curve(y_train, train_est_p)





plt.figure()
plt.plot(fpr_test_tr, tpr_test_tr,color='green',label='decisiontree roc curve',linestyle=':')
plt.plot(fpr_test_ra, tpr_test_ra,color='red',label='randomforest roc curve',linestyle=':')
plt.plot(fpr_test_ann, tpr_test_ann,color='aqua',label='neural_network roc curve',linestyle='-')
plt.plot(fpr_test_svm, tpr_test_svm,color='navy',label='svm roc curve',linestyle='-.')
plt.plot(fpr_test, tpr_test,color='deeppink',label='logistic roc curve')
plt.legend(loc='lower right')
plt.title('ROC curve')


metrics.roc_auc_score(y_test, test_est_p_ra)
metrics.roc_auc_score(y_test, test_est_p_svm)
metrics.roc_auc_score(y_test, test_est_p_ann)
metrics.roc_auc_score(y_test, test_est_p_tr)
metrics.roc_auc_score(y_test, test_est_p)

print(metrics.classification_report(y_test,test_est_ra))
print(metrics.classification_report(y_test,test_est_svm))
print(metrics.classification_report(y_test,test_est_ann))
print(metrics.classification_report(y_test,test_est_tr))
print(metrics.classification_report(y_test,test_est_tr))
