# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 21:18:15 2017

@author: Abhilash Srivastava
"""

import pandas as pd
import numpy as np
#import quandl
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing,cross_validation
from sklearn import svm
from sklearn.linear_model import LinearRegression

#df=quandl.get('WIKI/PRICES')
df=pd.read_csv('WIKI-PRICES.csv')
df=df[['adj_open','adj_high','adj_close','adj_volume']]

df.fillna(value=-99999,inplace=True)
forecast_out=int(math.ceil(0.005*len(df)))

df['label']=df['adj_close'].shift(-forecast_out)
df.dropna(inplace=True)
X=np.array(df.drop(['label'],1))
y=np.array(df['label'])

plt.scatter(df['label'],df['adj_volume'])
plt.show()
X=preprocessing.scale(X)
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)
#classifier
clf=svm.SVR()
#clf=LinearRegression()
#train
clf.fit(X_train,y_train)
#test
confidence=clf.score(X_test,y_test)
