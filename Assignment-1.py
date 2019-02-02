#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as py
import numpy as np


# In[2]:


data = pd.read_csv('overdoses.csv', sep =',')
##print("The shape of the data is :",data.shape)
data.head()


# In[3]:


'''
Creating OOD Column
'''

data['Population'] = data['Population'].str.replace(',', '')
data['Deaths'] = data['Deaths'].str.replace(',', '')
data[data.columns[1:3]] = data[data.columns[1:3]].astype(int)
data.dtypes


# In[4]:


'''
Plotting Pearson correlation coefficient btw Population and Deaths
'''
print("The value of pearson coefficient")
print(data[data.columns[1:3]].corr(method ='pearson'))


# In[5]:


'''
Plotting Bar graph for OOD
'''

data['OOD'] = data['Deaths']/data['Population']
py.figure(figsize=(20,20))
py.bar(np.arange(50), data['OOD'])
py.xticks(np.arange(50), data['Abbrev'])
py.show()


# In[6]:


'''
Creating 50x50 distance Matrix
'''

sim_data = pd.DataFrame(data['Abbrev'])
for i in range(50):
    sim_data[i] =  data['OOD'] - data.iloc[i, -1]
print("The shape of the matrix is : ", sim_data.shape)
sim_data.head()


# In[7]:


'''
Scaling and Using Cosine Funciton for Similarity Matrix 
'''

sim_data.iloc[:, 1:] = sim_data.iloc[:, 1:].abs()
sim_data.iloc[:, 1:] = sim_data.iloc[:, 1:]*100000
sim_data.iloc[:, 1:] = sim_data.iloc[:, 1:]/max(sim_data.iloc[:, 1:].max(axis = 1))
sim_data.iloc[:, 1:] = np.cos(sim_data.iloc[:, 1:])
sim_data.head()


# In[8]:


'''
Appending the Abbrevations to form 51x51 Matrix
'''

l = ['Abbrev']
for i in range(50):
    l.append(data['Abbrev'][i])    
final_sim = pd.DataFrame(l)
final_sim = final_sim.transpose()
final_sim.columns = sim_data.columns
final_sim = pd.concat([final_sim, sim_data], axis = 0)
print("The shape of the Similarity matrix is :", final_sim.shape)
print(final_sim)


# In[ ]:




