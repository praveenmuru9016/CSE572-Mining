#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as  np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import random
import math
from copy import deepcopy
import random
import matplotlib.pyplot as plt


# In[2]:


'''
Function to calculate the euclidean dist
'''

def distance(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


# In[3]:


'''
Function to calculate the k-means.

A. The function calculates the clusters for k from 2 to 15
B. For each cluster size we run the algorithm and take the min value of J
C. Return a list of Min J value for each K and the clustering for k=5

'''

def kmeans_function(data_frame):
    l = []
    for k in range(2,16):
        final = []
        for time in range(100):
            max_itr = 100
            clusters = {}
            centroids = np.zeros((k,data_frame.shape[1]))
            C_index = random.sample(range(data_frame.shape[0]), k)
            C = data_frame.values[C_index, :]
            _c = np.zeros(C.shape)
            itr = 0
            while(distance(C,_c, None) > 0.0001):
                for i in range(k):
                    clusters[i] = []

                if itr == max_itr:
                    break

                for i in range(data_frame.shape[0]):
                    assigned_cluster = np.argmin(distance(data_frame.values[i,:], C))
                    clusters[assigned_cluster].append(i)

                _c = deepcopy(C)

                for i in range(k):
                    C[i] = np.mean(data_frame.values[clusters[i]], axis = 0)
                itr += 1

            if (k==5):
                 cluster_5 = deepcopy(clusters)
            
            L = 0
            for i in range(k):
                temp_data = data_frame.values[clusters[i], :]
        #         print(temp_data.shape)
                for j in range(temp_data.shape[0]):
                    L += math.pow(distance(temp_data[j], C[i], ax=0),2)
            final.append(L)
        l.append(min(final))
    return l, cluster_5


# In[4]:


'''
Importing the data and creating the population and death subdata set
'''

data_frame = pd.read_csv('overdoses.csv', delimiter = ',')
csv_data=data_frame
data_frame['Population'] = data_frame['Population'].str.replace(',', '')
data_frame['Deaths'] = data_frame['Deaths'].str.replace(',', '')
data_frame[data_frame.columns[1:3]] = data_frame[data_frame.columns[1:3]].astype(float)

population_deaths_data = data_frame[['Population', 'Deaths']]
print(population_deaths_data)


# In[5]:


'''
Calling the k-means for the population dataset
'''

l_kmeans, cluster_5 = kmeans_function(population_deaths_data) 


# In[6]:


'''
Calling the k-means for the cosine-similarity dataset
'''

out = cosine_similarity(population_deaths_data,population_deaths_data)
out = pd.DataFrame(out)
l_cos, discard = kmeans_function(out)


# In[7]:


'''
Printing the table for k=5 
'''

table = np.zeros((50,2))
table[:,0] = np.arange(0,50,1)
table = pd.DataFrame(table, columns=['Row Num', 'Cluster'])
for i in range(5):
    table.values[cluster_5[i], -1] = i
print(table)


# In[8]:


'''
Plotting the Obj Function vs Number of Clusters
'''

plt.figure(figsize=(10, 10))
plt.plot(np.arange(2,16,1), l_kmeans)
plt.title('Obj Function vs Number of Clusters')
plt.xticks(np.arange(2,16,1))
plt.xlabel('No of Clusters (k)')
plt.ylabel('Objective Function value (J)')
plt.show()


# In[10]:


'''
Plotting the Obj Function vs Number of Clusters for cosine sim matrix
'''

plt.figure(figsize=(10, 10))
plt.plot(np.arange(2,16,1), l_cos)
plt.title('Obj Function vs Number of Clusters for cosine ')
plt.xticks(np.arange(2,16,1))
plt.xlabel('No of Clusters (k)')
plt.ylabel('Objective Function value (J)')
plt.show()


# In[ ]:




