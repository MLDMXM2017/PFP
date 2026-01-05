# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 19:06:58 2025
@author: Gao Jie
Article Title: Construction of a machine learning-based pulmonary function prediction model for the Chinese population
Description：Pulmonary function prediction
Method: Normalize data, Calculate cluster center, Sample ranking by Euclidean distance, Eliminate noise, Machine learning modeling
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from deepforest import CascadeForestRegressor
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso,LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import normalize
from scipy.spatial import distance
from sklearn.svm import SVR, LinearSVR

#parameter setting
dataname = "fuqing.csv"
n_splits =10
seed = 42

#method define
def load_data(dataname):
    data = pd.read_csv(dataname)
    data = data.values
    label1  = data.shape[1]-1
    label2  = data.shape[1]-2
    x = data[:,:label2]
    y1 = data[:,label1] #y1 corresponds to FEV1
    y2 = data[:,label2] #y2 corresponds to FVC
    return data,x,y1,y2

def delete_noise(data,rate):
    #Normalize data
    data_normalize = normalize(data, norm="l2")
    #Calculate cluster center
    clster_center = np.mean(data_normalize, axis=0) 
    #Calculate the distance between each sample and the cluster center
    distance_list=[]
    for item in data_normalize:
        distance_os = distance.euclidean(item,clster_center)
        distance_list.append(distance_os)   
    #Sample ranking by Euclidean distance
    distance_list = np.array(distance_list)
    sorted_index = np.argsort(distance_list)
    #Eliminate noise
    size = int(len(sorted_index)*rate)
    sorted_index_remained = sorted_index[:size]
    return sorted_index_remained
    
#Main experiment starts
data,x,y1,y2 = load_data(dataname)
sorted_index_remained = delete_noise(x, 0.99)
x  = x[sorted_index_remained]
y1 = y1[sorted_index_remained]
y2 = y2[sorted_index_remained]
#print("Shape of the original data:", x.shape)

r21,r22 = [],[]
rmse1,rmse2 = [],[]
mae1, mae2 = [],[]
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=5, random_state=seed)
for i, (train_index, test_index) in enumerate(rkf.split(x)):
    x_train = x[train_index]
    y_train1 = y1[train_index]
    y_train2 = y2[train_index]   
    x_test = x[test_index]
    y_test1 = y1[test_index]
    y_test2 = y2[test_index]
                
    #model1 = RandomForestRegressor(random_state=seed, n_jobs=-1).fit(x_train, y_train1)
    #model2 = RandomForestRegressor(random_state=seed, n_jobs=-1).fit(x_train, y_train2)
    #model1 = LinearRegression(n_jobs=-1 ).fit(x_train, y_train1)
    #model2 = LinearRegression(n_jobs=-1 ).fit(x_train, y_train2)
    #model1 = KNeighborsRegressor(n_jobs=-1).fit(x_train, y_train1)
    #model2 = KNeighborsRegressor(n_jobs=-1).fit(x_train, y_train2)
    #model1 = MLPRegressor(random_state=seed, max_iter=500).fit(x_train, y_train1)
    #model2 = MLPRegressor(random_state=seed, max_iter=500).fit(x_train, y_train2)
    model1=GradientBoostingRegressor(random_state=seed).fit(x_train, y_train1)
    model2=GradientBoostingRegressor(random_state=seed).fit(x_train, y_train2)
    #model1 = CascadeForestRegressor(verbose=0,n_jobs=-1)
    #model1.fit(x_train, y_train1)
    #model2 = CascadeForestRegressor(verbose=0,n_jobs=-1)
    #model2.fit(x_train, y_train2)
        
    y_pred1 = model1.predict(x_test)
    y_pred2 = model2.predict(x_test)

    r2_temp1 = r2_score(y_test1,y_pred1)
    r2_temp2 = r2_score(y_test2,y_pred2)
    r21.append(r2_temp1)
    r22.append(r2_temp2)
        
    mse1 = mean_squared_error(y_test1,y_pred1)
    mse2 = mean_squared_error(y_test2,y_pred2)      
    rmse_temp1 = sqrt(mse1)
    rmse_temp2 = sqrt(mse2)
    rmse1.append(rmse_temp1)
    rmse2.append(rmse_temp2)

    mae_temp1 = mean_absolute_error(y_test1,y_pred1)
    mae_temp2 = mean_absolute_error(y_test2,y_pred2)   
    mae1.append(mae_temp1)
    mae2.append(mae_temp2)   
    print(i+1,"-th modeling complete")

#Result output
print("******")  
print("R2 for FEV1:",   round(np.mean(r21),2))
print("R2 for FVC:",   round(np.mean(r22),2))

print("******") 
print("RMSE for FEV1:", round(np.mean(rmse1),2))
print("RMSE for FVC:", round(np.mean(rmse2),2))

print("******")
print("MAE for FEV1:",  round(np.mean(mae1),2)) 
print("MAE for FVC:",  round(np.mean(mae2),2)) 