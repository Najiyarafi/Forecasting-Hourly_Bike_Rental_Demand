#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importing libraries
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
import calendar
import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# loadind the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[4]:


# shape of training and testing data
train.shape, test.shape


# In[5]:


# printing first five rows
train.head()


# In[6]:


test.head()


# In[7]:


# columns in the dataset
train.columns


# In[8]:


test.columns


# In[9]:


# Data type of the columns
train.dtypes


# ## Univariate Analysis

# In[10]:


# distribution of count variable
sn.displot(train["count"])


# The distribution is skewed towards right and hence we can take log of the variable and see if the distribution becomes normal.

# In[11]:


sn.displot(np.log(train["count"]))


# In[12]:


sn.displot(train["registered"])


# ## Bivariate Analysis

# In[13]:


# looking at the correlation between numerical variables
corr = train[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")


# Some of the inferences from the above correlation map are:
# 
# 1. temp and humidity features has got positive and negative correlation with count respectively.Although the correlation between them are not very prominent still the count variable has got little dependency on "temp" and "humidity".
# 
# 2. windspeed will not be really useful numerical feature and it is visible from it correlation value with "count"
# 
# 3. Since "atemp" and "temp" has got strong correlation with each other, during model building any one of the variable has to be dropped since they will exhibit multicollinearity in the data.

# In[13]:


# looking for missing values in the datasaet
train.isnull().sum()


# In[14]:


test.isnull().sum()


# In[15]:


# extracting date, hour and month from the datetime
train["date"] = train.datetime.apply(lambda x : x.split()[0])
train["hour"] = train.datetime.apply(lambda x : x.split()[1].split(":")[0])
train["month"] = train.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)


# In[16]:


test["date"] = test.datetime.apply(lambda x : x.split()[0])
test["hour"] = test.datetime.apply(lambda x : x.split()[1].split(":")[0])
test["month"] = test.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)


# In[17]:


training = train[train['datetime']<='2012-03-30 0:00:00']
validation = train[train['datetime']>'2012-03-30 0:00:00']


# In[18]:


train = train.drop(['datetime','date', 'atemp'],axis=1)
test = test.drop(['datetime','date', 'atemp'], axis=1)
training = training.drop(['datetime','date', 'atemp'],axis=1)
validation = validation.drop(['datetime','date', 'atemp'],axis=1)


# ## Model Building
# ### Linear Regression Model

# In[14]:


from sklearn.linear_model import LinearRegression


# In[ ]:


# initialize the linear regression model
lModel = LinearRegression()


# In[ ]:


X_train = training.drop('count', 1)
y_train = np.log(training['count'])
X_val = validation.drop('count', 1)
y_val = np.log(validation['count'])


# In[ ]:


# checking the shape of X_train, y_train, X_val and y_val
X_train.shape, y_train.shape, X_val.shape, y_val.shape


# In[ ]:


# fitting the model on X_train and y_train
lModel.fit(X_train,y_train)


# In[ ]:


# making prediction on validation set
prediction = lModel.predict(X_val)


# In[ ]:


# defining a function which will return the rmsle score
def rmsle(y, y_):
    y = np.exp(y),   # taking the exponential as we took the log of target variable
    y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


# Let's now calculate the rmsle value of the predictions

# In[ ]:


rmsle(y_val,prediction)


# In[ ]:


# uncomment it to save the predictions from linear regression model and submit these predictions to generate score.
test_prediction = lModel.predict(test)


#  rmsle value of 0.8875 on the validation set.
# 

# ## Decision Tree

# In[15]:


from sklearn.tree import DecisionTreeRegressor


# In[16]:


# defining a decision tree model with a depth of 5. You can further tune the hyperparameters to improve the score
dt_reg = DecisionTreeRegressor(max_depth=5)


# Let's fit the decision tree model now.

# In[ ]:


dt_reg.fit(X_train, y_train)


# In[ ]:


predict = dt_reg.predict(X_val)


# In[ ]:


# calculating rmsle of the predicted values
rmsle(y_val, predict)


# The rmsle value has decreased to 0.171. 

# In[ ]:


test_prediction = dt_reg.predict(test)


# In[ ]:


final_prediction = np.exp(test_prediction)


# In[ ]:


submission = pd.DataFrame()


# In[ ]:


# creating a count column and saving the predictions in it
submission['count'] = final_prediction


# In[ ]:


submission.to_csv('submission.csv', header=True, index=False)

