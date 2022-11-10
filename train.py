#!/usr/bin/env python
# coding: utf-8

# # ML Zoomcamp Midterm Term Project
# 
# ####  Emre Ozturk


import pickle
import numpy as np
import pandas as pd
from IPython.display import display
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.utils.multiclass import type_of_target
from sklearn import metrics  
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer

import warnings
warnings.filterwarnings("ignore")


# #  Data preparation

df_orig = pd.read_csv('/Users/kadkoy/Desktop/midtermproject_2022/wp_dataset/WP_fulldataset.csv')
df = df_orig.copy()


def data_preprocessing(df):
    df.columns = df.columns.str.lower().str.replace(' ','_')
    #Dropped duplicate rows, keep last duplicate
    df = df.drop_duplicates(keep='last')
    
    ###### It seems extraction_type_other - mkulima/shinyanga is noise and being trouble.Lets drop!! Because its only have 1 instance and test_df not including another one.
    #df.drop("extraction_type_other_mkulima/shinyanga", axis=1, inplace=True) 
 
    ##########################################################################
    df['management_group'] = df['management_group'].replace('unknown', 'other')
      
    ###########################################################################
    df['payment_type'] = df['payment_type'].replace('unknown', 'other')
    
    #######################################################
    df.drop("quality_group", axis=1, inplace=True)
    
    ###################################################################
    df['source'] = df['source'].replace('unknown', 'other')
      
    ##### other- mikulima/shinyanga should be noise?!!!!!
    ###############################################################
    df['public_meeting'].fillna(value='VARIOUS', inplace=True)
       
    ##################################################################
    df['scheme_management'].fillna(value='VARIOUS', inplace=True)
       
    ###################################################
    df['permit'].fillna(value='VARIOUS', inplace=True)
       
    ################################################################################
    df['scheme_management'] = df['scheme_management'].replace('None', 'Other')
       
    ########################################################################
    # Eliminate the values where population is = 0 and population = 1 and created a new data frame
    df_pop_nozerone = df[df['population'] >1]
    
    #################################################
    #The values where the population is 1 and 0.
    df_pop_predict = df[df["population"]<2]
    
    ####################################################
    #Created a new data frame without where the value of construction year is 0. 
    df_nozero_const = df_pop_nozerone[df_pop_nozerone['construction_year'] > 1000]  
    
    ####################################################
    #construction year = 0
    df_const_predict = df_pop_nozerone[df_pop_nozerone['construction_year'] < 1000]
    
    ########################################################################################
    df_scheme_manag_novarious = df_nozero_const[df_nozero_const['scheme_management'] != 'VARIOUS']
    
    ############################################################################################
    df_scheme_manag_predict = df_nozero_const[df_nozero_const['scheme_management'] == 'VARIOUS']
    
    #########################################################
    df_pub_meet_novarious = df_scheme_manag_novarious[df_scheme_manag_novarious['public_meeting'] != 'VARIOUS'] 
    df_pub_meet_predict = df_scheme_manag_novarious[df_scheme_manag_novarious['public_meeting'] == 'VARIOUS']
    
    ###############################################################
    #To predict the data where permit value is 'VARIOUS' are taken.
    df_permit_predict = df_pub_meet_novarious[df_pub_meet_novarious['permit'] == 'VARIOUS']
    df_permit_novarious = df_pub_meet_novarious[df_pub_meet_novarious['permit'] != 'VARIOUS']
    
    #we have created a new df after cleaning
    df_clean = df_permit_novarious

    #Fill VARIOUS (NULL) values with clean df 's mode.
    df['permit'] = df['permit'].replace('VARIOUS', df_clean['permit'].mode()[0])
       
    df['public_meeting'] = df['public_meeting'].replace('VARIOUS', df_clean['public_meeting'].mode()[0])
      
    df['scheme_management'] = df['scheme_management'].replace('VARIOUS', df_clean['scheme_management'].mode()[0])
    
    #######################################################################
    df['population'] = df['population'].replace(0, df_clean['population'].mean())
    
    df['population'] = df['population'].replace(1, df_clean['population'].mean())
    
    ############################################################################################
    df['construction_year'] = df['construction_year'].replace(0, df_clean['construction_year'].median())
    
    #changing bools to str for uniform names
    df['public_meeting'] = df['public_meeting'].astype(str)
    
    df['permit'] = df['permit'].astype(str)
    
    #Selecting categoric columns
    categorical = df.select_dtypes(exclude=[np.number])
    
    #selecting numerical columns
    numerical = df.select_dtypes(include=[np.number])
    
    for c in categorical:
        df[c] = df[c].str.lower().str.replace(' ','_')
    for c in categorical:
        df[c] = df[c].str.lower().str.replace('_-_','_')
    
    for c in categorical:
        df[c] = df[c].str.lower().str.replace('_/_','_')
        
        
    df['status_group'] = df['status_group'].replace('non_functional',0)
    df['status_group'] = df['status_group'].replace('functional',1)
    df['status_group'] = df['status_group'].replace('functional_needs_repair',2)
    
    return df 



df_full = data_preprocessing(df)


# # Training the model

#target values
y = df_full['status_group'].values
del df_full[ 'status_group']


#Selecting categoric columns
categorical = df_full.select_dtypes(exclude=[np.number])
    
#selecting numerical columns
numerical = df_full.select_dtypes(include=[np.number])


# ##### One Hot Encoding

train_dicts = df_full.to_dict(orient='records')

dv = DictVectorizer(sparse = False)

X = dv.fit_transform(train_dicts)


# train-test split
seed = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)

print('Train data     :', X_train.shape)
print('Test data      :', X_test.shape)


#target type
print(type_of_target(y))



#Scaling data for outliers
scaler = preprocessing.StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)


# # RANDOM FOREST
# ### (Choosed because highest accuracy, data has so many categoric features and RF has low computation cost , easy the tune)



RFmodel = RF(criterion = 'gini', n_estimators = 100, max_depth = 17, min_samples_leaf = 2, 
         min_samples_split = 9, max_features = 11, random_state = seed)
RFmodel.fit(X_train, y_train)
predictions = RFmodel.predict(X_test)
print( 'RF Training accuracy       :',(accuracy_score(y_train, RFmodel.predict(X_train))) )
print( 'RF Classification accuracy :',(accuracy_score(y_test, predictions)) ,"\n")


# # Saving Model


output_file = f'RFmodel.bin'

output_file



with open(output_file, 'wb') as f_out: 
    pickle.dump((dv,RFmodel), f_out)

