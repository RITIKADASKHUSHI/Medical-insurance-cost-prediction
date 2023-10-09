#!/usr/bin/env python
# coding: utf-8

# ### importing the dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


# ### Data Collection & Analysis

# In[2]:


# loading the data from csv file to Pandas DataFrame


# In[3]:


ins_dataset = pd.read_csv("insurance.csv")


# In[4]:


ins_dataset.head()


# In[5]:


# number of rows and columns
ins_dataset.shape


# In[6]:


# getting some information about the dataset
ins_dataset.info()


# #### Categorical Features:
# - Sex: M/F
# - Smoker: Y/N
# - Region: SW,SE,NW, NE

# In[7]:


# checking for missing values
ins_dataset.isnull().sum()


# #### Data Analysis
# (statistical Measure of the dataset)

# In[8]:


ins_dataset.describe()


# In[9]:


# distribution of age value
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(ins_dataset['age'])
plt.title('Age Distribution')
plt.show()


# In[10]:


# Gender column
plt.figure(figsize=(6,6))
sns.countplot(x='sex',data=ins_dataset)
plt.title('Sex Distribution')
plt.show()


# In[11]:


# cross-checking
ins_dataset['sex'].value_counts()


# In[12]:


#bmi distribution
plt.figure(figsize=(6,6))
sns.distplot(ins_dataset['bmi'])
plt.title('BMI Distribution')
plt.show()


# In[13]:


# Normal BMI Range --> 18.5 to 24.9


# In[14]:


# children column
plt.figure(figsize=(6,6))
sns.countplot(x='children',data=ins_dataset)
plt.title('Child Distribution')
plt.show()


# In[15]:


# smoker column
plt.figure(figsize=(6,6))
sns.countplot(x='smoker',data=ins_dataset)
plt.title('Smoker Distribution')
plt.show()


# In[16]:


ins_dataset['smoker'].value_counts()


# In[17]:


# region column
plt.figure(figsize=(6,6))
sns.countplot(x='region',data=ins_dataset)
plt.title('Region Distribution')
plt.show()


# In[18]:


# charges column
plt.figure(figsize=(6,6))
sns.distplot(ins_dataset['charges'])
plt.title('Charges Distribution')
plt.show()


# ### Data Pre-processing

# In[19]:


#encoding Sex Column
ins_dataset.replace({'sex':{'male':0,'female':1}},inplace=True)

#encoding Smoker Column
ins_dataset.replace({'smoker':{'yes':0,'no':1}},inplace=True)

#encoding Region Column
ins_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace=True)


# In[20]:


ins_dataset


# Splitting the Features and Labels

# In[21]:


#Features
x = ins_dataset.drop(columns='charges', axis=1)
#Labels
Y = ins_dataset['charges']


# In[22]:


print(x)


# In[23]:


print(Y)


# Splitting the data into Training data & Testing Data

# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.2, random_state=2)


# In[26]:


print(x.shape, X_train.shape, X_test.shape)


# # Model Training
# #### We need to compare with different machine learning models, and needs to find out the best predicted model
# - Linear Regression Model
# - Random Forest
# - Extreme Gradient Boost Regression Model
# - Support Vector Machine
# 

# ## Linear Regression Model

# In[27]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[28]:


regressor.fit(X_train,Y_train)


# In[29]:


#prediction on training data
training_data_prediction = regressor.predict(X_train)


# In[30]:


r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('Pridicted value is: ',r2_train)


# In[31]:


#prediction on testing data
test_data_prediction = regressor.predict(X_test)


# In[32]:


r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('Pridicted value is: ',r2_test)


# # Random Forest Regression Model

# In[33]:


from sklearn.ensemble import RandomForestRegressor
model2 = RandomForestRegressor()


# In[34]:


model2.fit(X_train,Y_train)


# In[35]:


#prediction on training data
training_data_prediction2 = model2.predict(X_train)


# In[36]:


r2_train2 = metrics.r2_score(Y_train, training_data_prediction2)
print('Pridicted value is: ',r2_train2)


# In[37]:


#prediction on testing data
test_data_prediction2 = model2.predict(X_test)


# In[38]:


r2_test2 = metrics.r2_score(Y_test, test_data_prediction2)
print('Pridicted value is: ',r2_test2)


# ## Extreme Gradient Boost Regression Model

# In[39]:


from xgboost import XGBRegressor


# In[40]:


model3 = XGBRegressor()


# In[41]:


model3.fit(X_train,Y_train)


# In[42]:


#prediction on training data
training_data_prediction3 = model3.predict(X_train)


# In[43]:


r2_train3 = metrics.r2_score(Y_train, training_data_prediction3)
print('Pridicted value is: ',r2_train3)


# In[44]:


#prediction on testing data
test_data_prediction3 = model3.predict(X_test)


# In[45]:


r2_test3 = metrics.r2_score(Y_test, test_data_prediction3)
print('Pridicted value is: ',r2_test3)


# ### Building a Predictive System

# In[48]:


# from csv file pick any row
# 31,male,36.3,2,yes,southwest,38711
input_data = (31,1,36.3,2,0,1)
# changing input_data to a numpy array
input_data_as_np_arr = np.asarray(input_data)

# reshape the array
input_data_reshape = input_data_as_np_arr.reshape(1,-1)

prediction = model3.predict(input_data_reshape)

print(prediction)


# # Support Vector Machines

# In[49]:


from sklearn.svm import SVR


# In[50]:


model4 = SVR(kernel='rbf',C=1)


# In[51]:


model4.fit(X_train,Y_train)


# In[52]:


#prediction on training data
training_data_prediction4 = model4.predict(X_train)


# In[53]:


r2_train4 = metrics.r2_score(Y_train, training_data_prediction2)
print('Pridicted value is: ',r2_train4)


# In[54]:


#prediction on testing data
test_data_prediction4 = model4.predict(X_test)


# In[55]:


r2_test4 = metrics.r2_score(Y_test, test_data_prediction4)
print('Pridicted value is: ',r2_test4)


# In[ ]:




