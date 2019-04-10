#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import datasets, linear_model, metrics
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


df_train2 = pd.read_csv("PB2_train.csv",header = None)
df_test2 = pd.read_csv("PB2_test.csv",header = None)


# In[3]:


df_train2_features = df_train2.iloc[:,[0,1]]
df_train2_response = df_train2.iloc[:,-1]


# In[12]:


reg2 = linear_model.LinearRegression()
reg2.fit(df_train2_features,df_train2_response)
print("The Model parameters are:{}".format(reg2.coef_))
print("The first model parameter Theta0 is:{}".format(reg2.intercept_))


# In[5]:


df_test2_features = df_test2.iloc[:,[0,1]]
df_test2_response = df_test2.iloc[:,-1]


# In[6]:


predicted_values = reg2.predict(df_test2_features)
print("The predicted values for the test_csv is: {}".format(predicted_values))


# In[11]:


mean_sqr_error = metrics.mean_squared_error(df_test2_response,predicted_values)
print("The mean square error if the model for the given dataset are: {}".format(mean_sqr_error))


# In[8]:


X = np.array([19,76])
X.reshape(1,-1)
sample_predict = reg2.predict([X])
print("The response value for the given sample feature values are : {}".format(sample_predict))


# In[13]:


from mpl_toolkits.mplot3d import axes3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X= df_test2_features.iloc[:,0].tolist() 
Y= df_test2_features.iloc[:,1].tolist()
Z= df_test2_response[:].tolist()
x_surf = np.arange(df_test2_features.iloc[:,0].min(), df_test2_features.iloc[:,0].max(),20)                # generate a mesh
y_surf = np.arange(df_test2_features.iloc[:,0].min(), df_test2_features.iloc[:,0].max(),20)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
# print(x_surf,y_surf)
exog = pd.DataFrame({1: x_surf.ravel(), 2: y_surf.ravel()})
out = reg2.predict(exog)
ax.plot_surface(x_surf, y_surf,
                out.reshape(x_surf.shape),
                rstride=1,
                cstride=1,
                color='None',
                alpha = 0.4)

ax.scatter(X,Y,Z, c='blue', marker='o',alpha=1)
plt.show()


# In[ ]:




