#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:50:33 2020

@author: smoke
"""
import pickle
import numpy as np
PATH = "/home/smoke/Documents/Machine Learning/project/NeuACF/processed/"

print("loading results")
a_file = open(PATH+"test_resuts.pkl", "rb")
results_dict = pickle.load(a_file)
print("Done")

print("loading testing data")
a_file = open(PATH+"test_neg.pkl", "rb")
test_dict = pickle.load(a_file)
print("Done")

#### performing sorting according to pridiction probabilities in descending order 
for key in test_dict:
    sort_ind = np.argsort(results_dict[key])
    l1 = np.array(test_dict[key])
    l1 = l1[sort_ind]
    test_dict[key] = l1[::-1]
    l2 = np.array(results_dict[key])
    l2 = l2[sort_ind]
    results_dict[key] = l2[::-1]
  
##### Finally testing for different top k
positives = np.genfromtxt("/home/smoke/Documents/Machine Learning/project/NeuACF/dataset/amazon/amovie.test.rating",delimiter="\t",max_rows=3000)
positives = np.array(positives, dtype=int)
positives = np.delete(positives, [2,3], axis = 1)
c=list()
for i in range(len(positives)):
    if positives[i][1]>=1000:
        c.append(i)
positives = np.delete(positives,c,axis = 0)

for k in range(5,21,5):
    count = 0
    NDCC = 0
    for i in positives:
        l = test_dict[i[0]]
        if i[1] in l[:k]:
            count += 1
            NDCC += 1/np.log2(1 + np.where(l==i[1])[0] + 1)  #### added 1 because indexing starts at 0
    print("For top "+str(k)+": HR = "+str(count/len(positives))+" NDCC = "+str(NDCC/len(positives)))