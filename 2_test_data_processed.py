import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle

PATH = "/home/smoke/Documents/Machine Learning/project/NeuACF/processed1/"

rat_mat = np.genfromtxt(PATH+"UI_new.csv",delimiter=',')

positives = np.genfromtxt(PATH+"amovie.test.rating",delimiter="\t",max_rows=3000) ##taking only 3000 users
print(len(positives))
print(len(positives[0]))
print(positives[0])
positives = np.array(positives, dtype=int)
for i in range(len(positives)):
  if i!=positives[i][0]:
    print(i)

positives = np.delete(positives, [2,3], axis = 1)
print(len(positives), len(positives[0]))
print(positives[0])
c=list()
for i in range(len(positives)):  ### taking only 1000 items
  if positives[i][1]>=1000:
    c.append(i)
positives = np.delete(positives,c,axis = 0)
print(len(positives), len(positives[0]))

test_negatives = dict()
neg_ratio = 99
for i in positives:
  l =list()
  l.append(i[1])
  for neg in range(neg_ratio):
    j = np.random.randint(1000)
    while rat_mat[i[0]][j] != 0 or j in l:
       j = np.random.randint(1000)
    l.append(j)
    print(i[0])
  test_negatives[i[0]] = l

a_file = open(PATH+"test_neg.pkl", "wb")
pickle.dump(test_negatives, a_file)
a_file.close()



