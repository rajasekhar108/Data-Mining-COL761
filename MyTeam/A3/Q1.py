# -*- coding: utf-8 -*-
"""Q1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16ykjjv_RFCum5sDUqryo92rJmMri5ieH
"""

import numpy as np
import time
from scipy.spatial.distance import cdist
import math
import matplotlib.pyplot as plt



n_points=1000000
dim=np.array([1,2,4,8,16,32,64])
seed_value = 42
np.random.seed(seed_value)
def gen_data_points(n_point,d):
  data = np.round(np.random.uniform(0, 1, size=(n_point, d)),5)
  return data

tot_data={}
for i in dim:
  data=gen_data_points(n_points,i)
  tot_data[i]=data

query_data={}
for i in dim:
  query_indices = np.random.choice(n_points, 100, replace=False)
  q_data=tot_data[i][query_indices]
  query_data[i]=q_data


l1_min_max={d: {'max': None, 'min': None} for d in dim}
l2_min_max={d: {'max': None, 'min': None} for d in dim}
linfi_min_max={d: {'max': None, 'min': None} for d in dim}
for d in dim:
  l1maxarray=np.zeros(100)
  l1minarray=np.zeros(100)
  l2maxarray=np.zeros(100)
  l2minarray=np.zeros(100)
  linfimaxarray=np.zeros(100)
  linfiminarray=np.zeros(100)
  ll = len(tot_data[1])
  for i,q in enumerate(query_data[d]):
    templ1 = []
    templ2 = []
    templinfi = []
    if (d==1):
        templ1 = cdist([q],tot_data[d],metric='minkowski')
        templ2 = templ1
        templinfi = templ1
    else:
        templ1 = cdist([q],tot_data[d],metric='minkowski')
        templ2 = cdist([q],tot_data[d],metric='euclidean')
        templinfi = cdist([q],tot_data[d],metric='chebyshev')



    l1maxarray[i]=np.max(templ1)
    l1minarray[i]=np.min(templ1[templ1 !=0])

    l2maxarray[i]=np.max(templ2)
    l2minarray[i]=np.min(templ2[templ1 !=0])

    linfimaxarray[i]=np.max(templinfi)
    linfiminarray[i]=np.min(templinfi[templ1 !=0])

  l1_min_max[d]['max']=l1maxarray
  l1_min_max[d]['min']=l1minarray

  l2_min_max[d]['max']=l2maxarray
  l2_min_max[d]['min']=l2minarray

  linfi_min_max[d]['max']=linfimaxarray
  linfi_min_max[d]['min']=linfiminarray

ratiol1=np.zeros(7)
ratiol2=np.zeros(7)
ratiolinfi=np.zeros(7)
for j,d in enumerate(dim):
  nearl1 = l1_min_max[d]['min']
  farl1 = l1_min_max[d]['max']
  nearl2 = l2_min_max[d]['min']
  farl2 = l2_min_max[d]['max']
  nearlinfi = linfi_min_max[d]['min']
  farlinfi = linfi_min_max[d]['max']
  tl1 = np.mean(farl1) / np.mean(nearl1)
  tl2 = np.mean(farl2) / np.mean(nearl2)
  tlinfi = np.mean(farlinfi) / np.mean(nearlinfi)
  ratiol1[j]= tl1
  ratiol2[j]=tl2
  ratiolinfi[j]=tlinfi

plt.plot(dim, ratiol1, label='l1 norm',c='r')
plt.plot(dim, ratiol2, label='l2 norm',c='b')
plt.plot(dim, ratiolinfi, label='linfi norm',c='g')

plt.xlabel('Dimension (d)')
plt.ylabel('Average Ratio of Farthest to Nearest Distances')
plt.legend(loc='upper right')
plt.title('Behavior of Uniform Distribution in High-Dimensional Spaces')


plt.savefig("Q1.png")
