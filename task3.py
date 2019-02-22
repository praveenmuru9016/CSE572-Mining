# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 19:16:28 2019

"""
import numpy as np

def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a=np.linalg.norm(a)
    norm_b=np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

from math import *
def euclidean(x,y):
     return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

D1=np.array([2000,3000,2500,2000])
D2=np.array([1500,2500,4000,3000])
D3=np.array([20,30,25,20])
D4=np.array([15,25,40,30])

print('Cosine Similarity between D1 & D2',cos_sim(D1,D2))
print('Cosine Similarity between D1 & D3',cos_sim(D1,D3))
print('Cosine Similarity between D1 & D4',cos_sim(D1,D4))
print('Cosine Similarity between D2 & D3',cos_sim(D2,D3))
print('Cosine Similarity between D2 & D4',cos_sim(D2,D4))
print('Cosine Similarity between D3 & D4',cos_sim(D3,D4))
print
print
print('Euclidean Similarity between D1 & D2',euclidean(D1,D2))
print('Euclidean Similarity between D1 & D3',euclidean(D1,D3))
print('Euclidean Similarity between D1 & D4',euclidean(D1,D4))
print('Euclidean Similarity between D2 & D3',euclidean(D2,D3))
print('Euclidean Similarity between D2 & D4',euclidean(D2,D4))
print('Euclidean Similarity between D3 & D4',euclidean(D3,D4))

