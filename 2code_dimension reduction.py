# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 19:06:58 2025
@author: Gao Jie
Article Title: Construction of a machine learning-based pulmonary function prediction model for the Chinese population
Description：Dimension reduction
Method: 
show performance changes during dimension reduction
Show the difference between shap-based ranking and model-based ranking
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from deepforest import CascadeForestRegressor
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor, AdaBoostRegressor,BaggingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
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
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import normalize
from scipy.spatial import distance
from sklearn.svm import SVR, LinearSVR
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#parameter setting
seed = 42
dataname = "fuqing.csv"

#method define
def load_data(dataname):
    data = pd.read_csv(dataname)
    data = data.values
    
    label1  = data.shape[1]-1  
    label2  = data.shape[1]-2  
    
    x  = data[:,:label2]
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
    
#Experiment starts
data,x,y1,y2 = load_data(dataname)
sorted_index_remained = delete_noise(x, 0.99)
x = x[sorted_index_remained]
y1 = y1[sorted_index_remained]
y2 = y2[sorted_index_remained]
model = GradientBoostingRegressor(random_state=seed).fit(x, y2) #Modeling using the FVC indicator

#The column names corresponding to the original data
featureName = ["Gender", "Occupational exposure", "Dust exposure", "Chemical vapor exposure", "Smoke exposure", "Acid/base exposure", "Other exposure", "Marital status", "Chronic diseases", "T1DM", "T2DM", "Hypertension", "COPD", "Asthma", "Hyperthyroidism", "Hypothyroidism", "Medication use", "Antidiabetic agents", "Antihypertensive  agents", "Ventilation use", "Self-cooking", "Burn incense", "Burn paper money", "Burn mosquito incense", "Family numbers", "Family income", "Age", "Education level", "Smoking status", "Occupation", "Alcohol drinking status", "Tea drinking status", "Physical activity", "Family history of COPD", "Family history of CHD", "Family history of hypertension", "Family history of stroke", "Family history of DM", "Family history of cancer", "PSQI", "PSQI group", "Alcohol consumption", "Smoking quantity", "Hight", "Waist circumference", "Hip circumference", "Weight", "BFR", "MS", "PBM", "Energe", "Body age", "Water", "VFR", "SBP", "DBP", "HR", "Lefthand grip", "Righthand grip", "Average grip", "6MWS", "BMI"]
data = pd.DataFrame(x,columns=featureName) 
explainer = shap.Explainer(model,x)
shap_values = explainer(data.values) #.values

#Extract the feature importance ranking based on SHAP
shap_fiv = np.mean(np.abs(shap_values.values),axis=0)
sort_index_shap = np.argsort(shap_fiv)
#Extract the feature importance ranking based on GradientBoosting
sort_index_tree = np.argsort(-model.feature_importances_)
#A slight difference exist between shap-based ranking and GB-based ranking, either one can be chosen. 
rank_type = sort_index_shap
#Show model performance changes during dimension reduction
while True:
    x_train, x_test, y_train, y_test = train_test_split(x[:,rank_type], y2, test_size=0.2, random_state=seed)
    model=GradientBoostingRegressor(random_state=seed).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test,y_pred)
    print("feature size:",len(rank_type), " r2:",r2)
    rank_type = rank_type[:len(rank_type)-1]
    if len(rank_type)==0:break
#Show the difference between shap-based ranking and GB-based ranking
for i in range(len(sort_index_shap)):
    print(featureName[sort_index_shap[i]], featureName[sort_index_tree[i]])