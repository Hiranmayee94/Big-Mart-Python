#importing the data
import pandas as pd
import numpy as np
import os
wkdir = os.chdir('C:\\Users\\hi\\Desktop\\Data Science\\Python\\Big mart')
train=pd.read_csv('Train_Data.csv')
test=pd.read_csv('Test_Data.csv')

#summarising the data
summary=train.describe()
train['Outlet_Size'].value_counts()

#finding the missing values
train.isnull().sum()
train.info()

#imputing the missing values
new_item_wt=np.where(train['Item_Weight'].isnull(),12.60,train['Item_Weight'])
#overriding the column

train['Item_Weight']=new_item_wt
new_item_os=np.where(train['Outlet_Size'].isnull(),'Medium',train['Outlet_Size'])
train['Outlet_Size']=new_item_os

#checking if missing values are imputed
train.info()

#importing sklearn packages
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

#converting the categorical variables into signals
train['Item_Type']=LE.fit_transform(train['Item_Type'])
train['Item_Type'].value_counts()
train['Outlet_Size']=LE.fit_transform(train['Outlet_Size'])
train['Outlet_Size'].value_counts()
train['Outlet_Type']=LE.fit_transform(train['Outlet_Type'])
train['Outlet_Type'].value_counts()


#handling inconsistent values
train['Item_Fat_Content'].value_counts()
train['Item_Fat_Content'].replace('LF','Low Fat',inplace=True)
train['Item_Fat_Content'].replace('low fat','Low Fat',inplace=True)
train['Item_Fat_Content'].replace('reg','Regular',inplace=True)
train['Item_Fat_Content'].value_counts()

#converting the categorical variables into signals
train['Item_Fat_Content']=LE.fit_transform(train['Item_Fat_Content'])
train['Item_Fat_Content'].value_counts()
train['Outlet_Location_Type']=LE.fit_transform(train['Outlet_Location_Type'])
train['Outlet_Location_Type'].value_counts()

#no of years in business
train['Outlet_Establishment_Year'].value_counts()
train['noofyears']=2018-train['Outlet_Establishment_Year']

#dividing the data into Dependant and Independant
train.info()
Y=train['Item_Outlet_Sales']
X=train[['Item_Weight','Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP','noofyears',
         'Outlet_Size','Outlet_Location_Type','Outlet_Type']]

#applying linear and logistic regression
import statsmodels.api as sm
model_lm=sm.OLS(Y,X).fit()
model_lm.summary()

from sklearn import linear_model
lm=linear_model.LinearRegression()
model=lm.fit(X,Y)
preds_LR=model.predict(X)

from sklearn.metrics import mean_squared_error
rmse_LR=np.sqrt(mean_squared_error(Y,preds_LR))

print(rmse_LR)

########### Applying  random forest    ################ use RandomForestClassifier for categorical 
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=500)
model_rf=rf.fit(X,Y)
preds_rf=model_rf.predict(X)
rmse_RF=np.sqrt(mean_squared_error(Y,preds_rf))
print(rmse_RF)

################### Applying supoort vector machine ################
from sklearn.svm import SVR
svr_r=SVR(kernel='rbf')
model_svr=svr_r.fit(X,Y)
preds_svr=model_svr.predict(X)
rmse_svr=np.sqrt(mean_squared_error(Y,preds_svr))
print(rmse_svr)

from sklearn.svm import SVR
svr_r=SVR(kernel='poly')
model_svr=svr_r.fit(X,Y)
preds_svr=model_svr.predict(X)
rmse_svr=np.sqrt(mean_squared_error(Y,preds_svr))
print(rmse_svr)

##################### Applying Neural Network######################
from sklearn.neural_network import MLPRegressor
MLP=MLPRegressor(activation='relu',max_iter=100,hidden_layer_sizes=(10,10,10))
MLP.fit(X,Y)

preds_mlp=MLP.predict(X)
from sklearn.metrics import mean_squared_error
rmse=np.sqrt(mean_squared_error(Y,preds_mlp))
print(rmse)