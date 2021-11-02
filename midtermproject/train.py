#!/usr/bin/env python
# coding: utf-8

# # ML Zoomcamp Midterm Term Project
# 
# ####  Emre Ozturk

#  # Data Mining on Water Pumps
#  
# ## Can you predict the faulty water pumps?
#  ![alt text](pumping.jpg "Water Pump Challenge")
#  
# Using data (**`WP_fulldataset.csv`**) provided from Taarifa and the Tanzanian Ministry of Water, can you predict which pumps are **functional**, which are functional but need some repairs (**functional needs repair**), and which don't work at all (**non functional**)? Predict one of these three classes based on a number of variables about what kind of pump is operating, when it was installed, and how it is managed. A smart understanding of which waterpoints will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania.
# 
# ## Features (attributes) in the dataset:
# 
# Your goal is to predict the operating condition of a waterpoint for each record in the dataset. You are provided the following set of information about the waterpoints:
# 
#             
# |Feature|Explanation|
# |-------|-----------|
# |amount_tsh | Total static head (amount water available to waterpoint)|
# |gps_height | Altitude of the well|
# |longitude | GPS coordinate|
# |latitude | GPS coordinate|
# |basin | Geographic water basin|
# |population | Population around the well|
# |public_meeting | True/False|
# |scheme_management | Who operates the waterpoint|
# |permit | If the waterpoint is permitted|
# |construction_year | Year the waterpoint was constructed|
# |extraction_type | The kind of extraction the waterpoint uses|
# |management_group | How the waterpoint is managed|
# |payment | What the water costs|
# |payment_type | What the water costs|
# |water_quality | The quality of the water|
# |quantity | The quantity of water|
# |quantity_group | The quantity of water|
# |source | The source of the water|
# |source_class | The source of the water|
# |waterpoint_type | The kind of waterpoint|
# 
# 
# ## Distribution of labels
# The labels in this dataset are simple. There are three possible values:
#  
# |Label|Explanation|
# |-----|-----------|
# |`functional` | the waterpoint is operational and there are no repairs needed|
# |`functional needs repair` | the waterpoint is operational, but needs repairs|
# |`non functional` | the waterpoint is not operational|
#  
# 
# ## Data files
# **`WP_fulldataset.csv`** : This is the data file that contains all the data about the water pump problem. 
# 
# **`WP_testdataset.csv`** : Once you built your best and final classifier, you will run your model on this 'test' set to produce your predictions. 
# 
# ## Evaulation metric
# 
# Evaulation metric of the model will be accuracy

# In[355]:



import time
import numpy as np
import pandas as pd
#import pandas_profiling 
from IPython.display import display
#import pandas_profiling 
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
from sklearn.utils.multiclass import type_of_target
from sklearn import metrics  
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings("ignore")


# ###  Load And Copy Data

# In[356]:


df_orig = pd.read_csv('E:\Downloads\WP_fulldataset.csv')
test_df = pd.read_csv('E:\Downloads\WP_testdataset.csv')
df = df_orig.copy()
col_names = df.columns.tolist()
print("Column names:")
print(col_names,'\n')

print('Data dimensions:',df.shape,'\n')

print("Sample data:")
df.head()


# ###  Data Type , Any Missing Values ?

# In[357]:


df.info()


# Tells us which features are continuous and which ones are categorical. 

# ## Overview of Data With Profile Report. Any anomally, duplicate rows..?

# In[358]:


#pandas_profiling.ProfileReport(df)


# Shows us missing values, 0 values ,duplicate row counts and many information about data.

# In[359]:


#Dropped duplicate rows, keep last duplicate
df = df.drop_duplicates(keep='last')


# ## How many unique value for each feature?

# In[360]:


df.nunique()


# In[361]:


#Looking for numeric features specs
df.describe()


# #  Data Exploration
# Some basic stats on the target variable:

# In[362]:


df['status_group'].unique()


# In[363]:


print('# non functional   = {}'.format(len(df[df['status_group'] =='non functional'])))
print('# functional = {}'.format(len(df[df['status_group'] =='functional'])))  
print('# functional needs repair = {}'.format(len(df[df['status_group'] =='functional needs repair'])))
print('% non functional   = {}%'.format(round(float(len(df[df['status_group'] == "non functional"])) / len(df) * 100), 3))


# ### Distribution of target values

# In[364]:


plt.figure(figsize=(25,10))
order = df.status_group.value_counts().index
sns.countplot(x='status_group', data=df, order=order).tick_params(labelsize=20)
sns.countplot(x='status_group', data=df, order=order).set_xlabel('status_group',fontsize=25)
sns.despine()


# ### Distribution of future values

# In[365]:


plt.figure(figsize=(25,10))
order = df.basin.value_counts().index
sns.countplot(x='basin', data=df, order=order).tick_params(labelsize=15)
sns.countplot(x='basin', data=df, order=order).set_xlabel('basin',fontsize=30)
sns.despine()


# In[366]:


plt.figure(figsize=(25,10))
order = df.extraction_type.value_counts().index
sns.countplot(x='extraction_type', data=df, order=order).set_xlabel('extraction_type',fontsize=25)
sns.countplot(x='extraction_type', data=df, order=order).tick_params(labelsize=15)
sns.despine()


# In[367]:


plt.figure(figsize=(25,10))
order = df.extraction_type_class.value_counts().index
sns.countplot(x='extraction_type_class', data=df, order=order ).tick_params(labelsize=20)
sns.countplot(x='extraction_type_class', data=df, order=order ).set_xlabel('extraction_type_class',fontsize=25)
sns.despine()


# In[368]:


plt.figure(figsize=(25,10))
order = df.management_group.value_counts().index
sns.countplot(x='management_group', data=df, order=order).tick_params(labelsize=25)
sns.countplot(x='management_group', data=df, order=order).set_xlabel('management_group',fontsize=30)
sns.despine()


# #We have 'other' and 'unknown' variables. It is useles having to unknow class. We will add 'unknown' to "other".

# In[369]:


df['management_group'] = df['management_group'].replace('unknown', 'other')
test_df['management_group'] = test_df['management_group'].replace('unknown', 'other')


# In[370]:


plt.figure(figsize=(25,10))
order = df.management_group.value_counts().index
sns.countplot(x='management_group', data=df, order=order).tick_params(labelsize=20)
sns.countplot(x='management_group', data=df, order=order).set_xlabel('management_group',fontsize=25)
sns.despine()


# In[371]:


plt.figure(figsize=(25,10))
order = df.payment_type.value_counts().index
sns.countplot(x='payment_type', data=df, order=order).tick_params(labelsize=20)
sns.countplot(x='payment_type', data=df, order=order).set_xlabel('payment_type',fontsize=25)
sns.despine()


# In[372]:


df['payment_type'].unique()


# #we have 'other' and 'unknown'. We will add 'unknown' to "other".

# In[373]:


df['payment_type'] = df['payment_type'].replace('unknown', 'other')
test_df['payment_type'] = test_df['payment_type'].replace('unknown', 'other')


# In[374]:


plt.figure(figsize=(25,10))
order = df.payment_type.value_counts().index
sns.countplot(x='payment_type', data=df, order=order).tick_params(labelsize=20)
sns.countplot(x='payment_type', data=df, order=order).set_xlabel('payment_type',fontsize=25)
sns.despine()


# In[375]:


plt.figure(figsize=(25,10))
order = df.water_quality.value_counts().index
sns.countplot(x='water_quality', data=df, order=order).tick_params(labelsize=20)
sns.countplot(x='water_quality', data=df, order=order).set_xlabel('water_quality',fontsize=25)
sns.despine()


# In[376]:


df['water_quality'].unique()


# In[377]:


print('# soft  = {}'.format(len(df[df['water_quality'] =='soft'])))
print('# salty  = {}'.format(len(df[df['water_quality'] =='salty'])))
print('# unknown  = {}'.format(len(df[df['water_quality'] =='unknown'])))


# In[378]:


plt.figure(figsize=(25,10))
order = df.quality_group.value_counts().index
sns.countplot(x='quality_group', data=df, order=order).tick_params(labelsize=20)
sns.countplot(x='quality_group', data=df, order=order).set_xlabel('quality_group',fontsize=25)
sns.despine()


# In[379]:


print('# good  = {}'.format(len(df[df['quality_group'] =='good'])))
print('# salty  = {}'.format(len(df[df['quality_group'] =='salty'])))
print('# unknown  = {}'.format(len(df[df['quality_group'] =='unknown'])))


# # When we look the detailed counts It seems water_quality and quality_group features are same. We are gonna drop the quality_group which has less variables.

# In[380]:


df.drop("quality_group", axis=1, inplace=True)
test_df.drop("quality_group", axis=1, inplace=True)


# In[381]:


df.head(1)


# In[382]:


plt.figure(figsize=(25,10))
order = df.quantity.value_counts().index
sns.countplot(x='quantity', data=df, order=order).tick_params(labelsize=20)
sns.countplot(x='quantity', data=df, order=order).set_xlabel('quantity',fontsize=25)
sns.despine()


# In[383]:


plt.figure(figsize=(25,10))
order = df.source.value_counts().index
sns.countplot(x='source', data=df, order=order)
sns.despine()


# In[384]:


df['source'].unique()


# In[385]:


df['source'] = df['source'].replace('unknown', 'other')
test_df['source'] = test_df['source'].replace('unknown', 'other')


# In[386]:


plt.figure(figsize=(25,10))
order = df.source.value_counts().index
sns.countplot(x='source', data=df, order=order).tick_params(labelsize=20)
sns.countplot(x='source', data=df, order=order).set_xlabel('source',fontsize=25)
sns.despine()


# In[387]:


plt.figure(figsize=(25,10))
order = df.source_class.value_counts().index
sns.countplot(x='source_class', data=df, order=order).tick_params(labelsize=20)
sns.countplot(x='source_class', data=df, order=order).set_xlabel('source_class',fontsize=25)
sns.despine()


# In[388]:


plt.figure(figsize=(25,10))
order = df.waterpoint_type.value_counts().index
sns.countplot(x='waterpoint_type', data=df, order=order).tick_params(labelsize=20)
sns.countplot(x='waterpoint_type', data=df, order=order).set_xlabel('waterpoint_type',fontsize=25)

sns.despine()


# # UNIVARIATE ANALYSIS 

# In[389]:


df.dtypes


# ### basin - functionality distrubution

# In[390]:


table=pd.crosstab(df.basin, df.status_group)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', figsize=(25,10), stacked=True , fontsize=20 , color=('g','b','r'))

plt.show()


# ### extraction_type - functionality distrubution

# In[391]:


table=pd.crosstab(df.extraction_type, df.status_group)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', figsize=(25,10), stacked=True,fontsize=20 , color=('g','b','r'))

plt.show()


# ### other- mikulima/shinyanga should be noise?

#  ### extraction_type_class - functionality distrubution

# In[392]:


table=pd.crosstab(df.extraction_type_class, df.status_group)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', figsize=(25,10), stacked=True,fontsize=20 , color=('g','b','r'))

plt.show()


# ### management_group - functionality distrubution

# In[393]:


table=pd.crosstab(df.management_group, df.status_group)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', figsize=(25,10), stacked=True,fontsize=20 , color=('g','b','r'))

plt.show()


# ### payment_type- functionality distrubution

# In[394]:


table=pd.crosstab(df.payment_type, df.status_group)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', figsize=(25,10), stacked=True, fontsize=20 , color=('g','b','r'))

plt.show()


# ### water_quality - functionality distrubution 

# In[395]:


table=pd.crosstab(df.water_quality, df.status_group)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', figsize=(25,10), stacked=True,fontsize=20 , color=('g','b','r'))

plt.show()


# ### quantity - functionality distrubution

# In[396]:


table=pd.crosstab(df.quantity, df.status_group)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', figsize=(25,10), stacked=True,fontsize=20 , color=('g','b','r'))


plt.show()


# ### source- functionality distrubution

# In[397]:


table=pd.crosstab(df.source, df.status_group)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', figsize=(25,10), stacked=True,fontsize=20 , color=('g','b','r'))


plt.show()


# ### source_class - functionality distrubution

# In[398]:


table=pd.crosstab(df.source_class, df.status_group)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', figsize=(25,10), stacked=True,fontsize=20 , color=('g','b','r'))


plt.show()


# ### waterpoint_type - functionality distrubution

# In[399]:


table=pd.crosstab(df.waterpoint_type, df.status_group)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', figsize=(25,10), stacked=True,fontsize=20 , color=('g','b','r'))


plt.show()


# # BOX PLOT (Is there any outlier in numeric values?)

# In[400]:


sns.boxplot(x=df["status_group"], y=df["amount_tsh"])


# ### It seems there is outliers but when we consider domanin they may not outliers. There is too much zero. They are affecting distrubution.

# In[401]:


sns.boxplot(x=df["status_group"], y=df["population"])


# ### Same situation is here. The values 0 and 1 are affecting the distribution.

# # PAIR PLOT

# In[402]:



#cols = ['status_group', 'waterpoint_type', 'source_class', 'source', 'quantity',  'water_quality', 'payment_type','management_group','extraction_type_class','construction_year','permit','scheme_management','public_meeting','population','basin','latitude','longitude','gps_height','amount_tsh']
#sns.pairplot(df[cols], hue = 'status_group', diag_kind = 'kde', 
             #plot_kws = {'alpha': 0.7, 's': 80, 'edgecolor': 'k'}, size = 4.5 )
#plt.show()


# #Pairplot is not saying too much. Data needs preprocessing

# In[403]:


df.info()


# ### Construcrion year seems important.

# In[404]:


table=pd.crosstab(df.construction_year, df.status_group)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', figsize=(50,10), stacked=True)
plt.xlabel('construction_year')
plt.show()


# In[405]:


df["construction_year"].unique()


# # Handling with Null Values and meaningless data

# In[406]:


df.isnull().sum()


# #### We've filled the missing values with 'VARIOUS'
# 

# In[407]:



df['public_meeting'].fillna(value='VARIOUS', inplace=True)
test_df['public_meeting'].fillna(value='VARIOUS', inplace=True)


# In[408]:


df['public_meeting'].value_counts().head()


# In[409]:


df_orig['public_meeting'].unique()


# In[410]:


df['scheme_management'].value_counts()


# In[411]:


df['scheme_management'].unique()


# In[412]:



df['scheme_management'].fillna(value='VARIOUS', inplace=True)
test_df['scheme_management'].fillna(value='VARIOUS', inplace=True)


# In[413]:


df['scheme_management'].value_counts()


# In[414]:


df['permit'].value_counts()


# In[415]:


df['permit'].unique()


# In[416]:



df['permit'].fillna(value='VARIOUS', inplace=True)
test_df['permit'].fillna(value='VARIOUS', inplace=True)


# In[417]:


df['permit'].value_counts()


# In[418]:


plt.figure(figsize=(25,10))
order = df.public_meeting.value_counts().index
sns.countplot(x='public_meeting', data=df, order=order).tick_params(labelsize=20)
sns.countplot(x='public_meeting', data=df, order=order).set_xlabel('public_meeting',fontsize=25)
sns.despine()


# In[419]:


df['scheme_management'].unique()


# In[420]:


df['scheme_management'] = df['scheme_management'].replace('None', 'Other')
test_df['scheme_management'] = test_df['scheme_management'].replace('None', 'Other')


# In[421]:


plt.figure(figsize=(25,10))
order = df.scheme_management.value_counts().index
sns.countplot(x='scheme_management', data=df, order=order).tick_params(labelsize=17)
sns.countplot(x='scheme_management', data=df, order=order).set_xlabel('scheme_management',fontsize=25)
sns.despine()


# In[422]:


plt.figure(figsize=(25,10))
order = df.permit.value_counts().index
sns.countplot(x='permit', data=df, order=order).tick_params(labelsize=17)
sns.countplot(x='permit', data=df, order=order).set_xlabel('permit',fontsize=25)
sns.despine()


# In[423]:


table=pd.crosstab(df.public_meeting, df.status_group)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', figsize=(25,10), stacked=True,fontsize=20 , color=('g','b','r'))


plt.show()


# In[424]:


table=pd.crosstab(df.scheme_management, df.status_group)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', figsize=(25,10), stacked=True,fontsize=20 , color=('g','b','r'))


plt.show()


# In[425]:


table=pd.crosstab(df.permit, df.status_group)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', figsize=(25,10), stacked=True,fontsize=20 , color=('g','b','r'))


plt.show()


# ### To fill missing values and handle with meaningless data, we are cleaned the data from both of them.

# ### Counts of missing and meaningless data

# In[426]:


print('#permit VARIOUS COUNT   = {}'.format(len(df[df['permit'] == 'VARIOUS'])))


# In[427]:


print('#public_meeting VARIOUS COUNT   = {}'.format(len(df[df['public_meeting'] == 'VARIOUS'] )))


# In[428]:


print('#scheme_management VARIOUS COUNT   = {}'.format(len(df[df['scheme_management'] == 'VARIOUS'])))


# In[429]:


print('# 0 in construction_year   COUNT   = {}'.format(len(df[df['construction_year'] == 0])))


# In[430]:


print('# 1 in population   COUNT   = {}'.format(len(df[df['population'] == 1])))


# In[431]:


print('# 0 in population   COUNT   = {}'.format(len(df[df['population'] == 0])))


# ### Cleaning Data

# In[432]:



# Eliminate the values where population is = 0 and population = 1 and created a new data frame
df_pop_nozerone = df[df['population'] >1]


# In[433]:



#The values where the population is 1 and 0.
df_pop_predict = df[df["population"]<2]


# In[434]:



#Created a new data frame without where the value of construction year is 0. 
df_nozero_const = df_pop_nozerone[df_pop_nozerone['construction_year'] > 1000]


# In[435]:


#construction year = 0
df_const_predict = df_pop_nozerone[df_pop_nozerone['construction_year'] < 1000]


# In[436]:


df_nozero_const['construction_year'].unique()


# In[437]:



df_scheme_manag_novarious = df_nozero_const[df_nozero_const['scheme_management'] != 'VARIOUS']


# In[438]:



df_scheme_manag_predict = df_nozero_const[df_nozero_const['scheme_management'] == 'VARIOUS']


# In[439]:


df_scheme_manag_novarious['scheme_management'].unique()


# In[440]:



df_pub_meet_novarious = df_scheme_manag_novarious[df_scheme_manag_novarious['public_meeting'] != 'VARIOUS']


# In[441]:


df_pub_meet_predict = df_scheme_manag_novarious[df_scheme_manag_novarious['public_meeting'] == 'VARIOUS']


# In[442]:


df_pub_meet_novarious['public_meeting'].unique()


# In[443]:


#To prediction the data where permit value is 'VARIOUS' are taken.
df_permit_predict = df_pub_meet_novarious[df_pub_meet_novarious['permit'] == 'VARIOUS']


# In[444]:



df_permit_novarious = df_pub_meet_novarious[df_pub_meet_novarious['permit'] != 'VARIOUS']


# In[445]:


df_permit_novarious['permit'].unique()


# In[446]:


#Created a new df after cleaning
df_clean = df_permit_novarious


# In[447]:


#pandas_profiling.ProfileReport(df_clean)


# ### After cleaned the data we are looking to data's specs.

# In[448]:


df_clean.describe()


# In[449]:


#cols = ['status_group', 'waterpoint_type', 'source_class', 'source', 'quantity',  'water_quality', 'payment_type','management_group','extraction_type_class','construction_year','permit','scheme_management','public_meeting','population','basin','latitude','longitude','gps_height','amount_tsh']
#sns.pairplot(df_clean[cols], hue = 'status_group', diag_kind = 'kde', 
             #plot_kws = {'alpha': 0.7, 's': 80, 'edgecolor': 'k'}, size = 4.5 )
#plt.show()


# ### Pair plot is saying already not much. There are skewnesses and not correlated data. Outliers? Normal distribution?

# In[450]:


df['permit'].unique()


# ### We've filled null values with mode of clean data

# In[451]:


#Fill VARIOUS (NULL) values with mode.
df['permit'] = df['permit'].replace('VARIOUS', df_clean['permit'].mode()[0])
test_df['permit'] = test_df['permit'].replace('VARIOUS', test_df['permit'].mode()[0])


# In[452]:


df['permit'].unique()


# In[453]:


df['public_meeting'] = df['public_meeting'].replace('VARIOUS', df_clean['public_meeting'].mode()[0])
test_df['public_meeting'] = test_df['public_meeting'].replace('VARIOUS', test_df['public_meeting'].mode()[0])


# In[454]:


df['public_meeting'].unique()


# In[455]:


df['scheme_management'] = df['scheme_management'].replace('VARIOUS', df_clean['scheme_management'].mode()[0])
test_df['scheme_management'] = test_df['scheme_management'].replace('VARIOUS', test_df['scheme_management'].mode()[0])


# In[456]:


df['scheme_management'].unique()


# In[457]:


# Checked the missing values again.
display(df.isnull().sum().sort_index()/len(df))


# ### We've filled 0 values with mean of the clean data.

# In[458]:


sns.distplot( df_clean["population"])


# #Seems the values 0 and 1 dominate the population.

# In[459]:


df['population'] = df['population'].replace(0, df_clean['population'].median())
test_df['population'] = test_df['population'].replace(0, test_df['population'].mean())


# In[460]:


#df['population'] = df['population'].replace(1, df_clean['population'].mean())


# In[461]:


sns.distplot( df_clean["construction_year"])


# In[462]:


df['construction_year'] = df['construction_year'].replace(0, df_clean['construction_year'].mean())
test_df['construction_year'] = test_df['construction_year'].replace(0, test_df['construction_year'].mean())


# In[463]:


df.isnull().any()


# In[464]:


print(df.columns + '\n')
print(test_df.columns)


# In[465]:


cols = df.select_dtypes(exclude=[np.number])


# In[466]:


list(cols)


# In[467]:


dummy_df = pd.get_dummies(df, columns = [ 
'basin',
 'public_meeting',
 'scheme_management',
 'permit',
 'extraction_type',
 'extraction_type_class',
 'management_group',
 'payment_type',
 'water_quality',
 'quantity',
 'source',
 'source_class',
 'waterpoint_type'])


# In[468]:


dummy_df.head()


# In[469]:


dummy_df['status_group'].unique()


# In[470]:


test_df.columns


# In[471]:


t_cols = test_df.select_dtypes(exclude=[np.number])


# In[472]:


list(t_cols)


# In[473]:


dummy_test_df = pd.get_dummies(test_df, columns = [ 
'basin',
 'public_meeting',
 'scheme_management',
 'permit',
 'extraction_type',
 'extraction_type_class',
 'management_group',
 'payment_type',
 'water_quality',
 'quantity',
 'source',
 'source_class',
 'waterpoint_type'])


# In[474]:


dummy_test_df.shape


# In[475]:


for a in dummy_df.columns:
    if a not in dummy_test_df:
        print(a)


# #It seems extraction_type_other - mkulima/shinyanga is noise and being trouble.Lets drop!!

# In[476]:


dummy_df.drop("extraction_type_other - mkulima/shinyanga", axis=1, inplace=True) 


# In[477]:


wp_testdataset = dummy_test_df


# In[478]:


wp_testdataset.head(1)


# In[479]:


dummy_df


# # PREDICTIONS

# In[480]:


X = dummy_df.drop(columns = [ 'status_group']).values
y = dummy_df['status_group'].values
wp_testdataset = wp_testdataset.values


# In[481]:


seed = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)


# In[482]:


print('Train data:', X_train.shape)
print('Test data :', X_test.shape)
print('wp_testdataset :', wp_testdataset.shape )


# In[483]:


type_of_target(y)


# ###  We've  scaled the data to avoid feature dominancy and outliers.

# In[484]:


scaler = preprocessing.StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)
wp_testdataset = scaler.transform(wp_testdataset)




# # RANDOM FOREST
# ### (Choosed because highest accuracy, data has so many categoric features and RF has low computation cost , easy the tune)

# In[494]:


param_grid = {"criterion"        : ['entropy','gini'],
              "min_samples_split": [ 5,9],
              "max_features"     : [ 8,11],
              "max_depth"        : [ 17, 20],
              "min_samples_leaf" : [2,3],
              "n_estimators"     : [60,100]
              }

kfolds = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
#t_start = time.clock()
grid_search = GridSearchCV(estimator = RF(random_state = seed), param_grid = param_grid, 
                           cv = kfolds, n_jobs = -1)
grid_search.fit(X_train, y_train)
#t_end = time.clock()
#print('Time elapsed :', t_end-t_start, 'sec\n')
print('Best parameters:\n', grid_search.best_params_,'\n')
print('Average CV accuracy:', np.mean(grid_search.cv_results_['mean_test_score']))


# In[496]:


RFmodel = RF(criterion = 'gini', n_estimators = 100, max_depth = 17, min_samples_leaf = 2, 
         min_samples_split = 9, max_features = 11, random_state = seed)
RFmodel.fit(X_train, y_train)
predictions = RFmodel.predict(X_test)
print( 'RF Training accuracy       :',(accuracy_score(y_train, RFmodel.predict(X_train))) )
print( 'RF Classification accuracy :',(accuracy_score(y_test, predictions)) ,"\n")


# In[318]:


wp_test_predictions = RFmodel.predict(wp_testdataset)


# In[323]:


test_df['predictions'] = wp_test_predictions


# In[325]:


test_df['predictions'].head()


# In[327]:


def draw_cm( actual, predicted ):
    cm = confusion_matrix( actual, predicted, ['non functional', 'functional', 'functional needs repair'] )
    sns.heatmap(cm, annot=True,  fmt='.0f', xticklabels = ['non functional', 'functional', 'functional needs repair'] , 
                yticklabels = ['non functional', 'functional', 'functional needs repair'] )
    plt.ylabel('ACTUAL')
    plt.xlabel('PREDICTED')
    plt.show()
draw_cm( y_test, predictions )





# # Saving Model

# In[498]:


import pickle


# In[500]:


output_file = f'RFmodel.bin'





# In[504]:


with open(output_file, 'wb') as f_out: 
    pickle.dump((RFmodel), f_out)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




