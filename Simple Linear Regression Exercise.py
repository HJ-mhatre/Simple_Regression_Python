#!/usr/bin/env python
# coding: utf-8

# # Simple linear regression - exercise

# You are given a real estate dataset. 
# 
# Real estate is one of those examples that every regression course goes through as it is extremely easy to understand and there is a (almost always) certain causal relationship to be found.
# 
# The data is located in the file: 'real_estate_price_size.csv'. 
# 
# You are expected to create a simple linear regression (similar to the one in the lecture), using the new data.
# 
# In this exercise, the dependent variable is 'price', while the independent variables is 'size'.
# 
# Good luck!

# ## Import the relevant libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ## Load the data

# In[7]:


data1=pd.read_csv('real_estate_price_size.csv')


# In[9]:


data1
data1.describe()


# ## Create the regression

# ### Declare the dependent and the independent variables

# In[10]:


y= data1['price']
x1= data1['size']


# ### Explore the data

# In[12]:


plt.scatter(x1,y)
plt.xlabel('SIZE', fontsize=20)
plt.ylabel('PRICE', fontsize=20)
plt.show()


# ### Regression itself

# In[15]:


x=sm.add_constant(x1)
results=sm.OLS(y,x).fit()
results.summary()


# ### Plot the regression line on the initial scatter

# In[19]:


plt.scatter(x1,y)
yhat=223.1787*x1 + 1.019e+05
fig=plt.plot(x1,yhat,lw=4,c='orange', Label='Regression Line')
plt.xlabel('SIZE', fontsize=20)
plt.ylabel('PRICE', fontsize=20)
plt.show()

