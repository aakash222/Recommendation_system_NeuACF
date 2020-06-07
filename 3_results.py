#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:44:52 2020

@author: smoke
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle

PATH = "/home/smoke/Documents/Machine Learning/project/NeuACF/processed1/"

class NeuACF(nn.Module):
  def __init__(self, num_U, num_I):
    super(NeuACF, self).__init__()
    self.UIBIU = nn.Sequential(
        nn.Linear(num_U,600), nn.ReLU(),
        nn.Linear(600,600), nn.ReLU(),
        nn.Linear(600,600), nn.ReLU(),
        nn.Linear(600,64)
    )
    self.UIU = nn.Sequential(
        nn.Linear(num_U,600), nn.ReLU(),
        nn.Linear(600,600), nn.ReLU(),
        nn.Linear(600,600), nn.ReLU(),
        nn.Linear(600,64)
    )
    self.IBI = nn.Sequential(
        nn.Linear(num_I,600), nn.ReLU(),
        nn.Linear(600,600), nn.ReLU(),
        nn.Linear(600,600), nn.ReLU(),
        nn.Linear(600,64)
    )
    self.IUI = nn.Sequential(
        nn.Linear(num_I,600), nn.ReLU(),
        nn.Linear(600,600), nn.ReLU(),
        nn.Linear(600,600), nn.ReLU(),
        nn.Linear(600,64)
    )
    self.UIBIU_att = nn.Sequential(nn.Linear(64,16), nn.ReLU(), nn.Linear(16,1))
    self.UIU_att = nn.Sequential(nn.Linear(64,16), nn.ReLU(), nn.Linear(16,1))
    self.IBI_att = nn.Sequential(nn.Linear(64,16), nn.ReLU(), nn.Linear(16,1))
    self.IUI_att = nn.Sequential(nn.Linear(64,16), nn.ReLU(), nn.Linear(16,1))
    self.softmax_u = nn.Softmax(dim=1)
    self.softmax_i = nn.Softmax(dim=1)
    self.sigmoid = nn.Sigmoid()

  def forward(self,uibiu,uiu,ibi,iui):
    x_u_1 = self.UIBIU(uibiu).squeeze()
    x_u_2 = self.UIU(uiu).squeeze()
    myvar_u = torch.cat((self.UIBIU_att(x_u_1).squeeze().unsqueeze(1),self.UIU_att(x_u_2).squeeze().unsqueeze(1)),dim=1)
    scores_u = self.softmax_u(myvar_u)

    x_u_1 = x_u_1 * scores_u[:,0].view(x_u_1.shape[0],1)
    x_u_2 = x_u_2 * scores_u[:,1].view(x_u_1.shape[0],1)
    x_u = x_u_1 + x_u_2
    

    x_i_1 = self.IBI(ibi).squeeze()
    x_i_2 = self.IUI(iui).squeeze()
    myvar_i = torch.cat((self.IBI_att(x_i_1).squeeze().unsqueeze(1),self.IUI_att(x_i_2).squeeze().unsqueeze(1)),dim=1)
    scores_i = self.softmax_i(myvar_i)
    
    x_i_1 = x_i_1 * scores_i[:,0].view(x_i_1.shape[0],1)
    x_i_2 = x_i_2 * scores_i[:,1].view(x_i_1.shape[0],1)
    x_i = x_i_1 + x_i_2

    z = torch.bmm(x_u.view(x_u.shape[0], 1, 64), x_i.view(x_i.shape[0], 64, 1)).squeeze()
    z = self.sigmoid(z)
    return z

batch_size = 100
num_users = 3000
num_items = 1000

print("loading model parameters")
net = NeuACF(num_users,num_items)
net.load_state_dict(torch.load(PATH+'net_cat.pt', map_location={'cuda:0': 'cpu'}))
print("Done")
net.eval()

print("loading testing data")
a_file = open(PATH+"test_neg.pkl", "rb")
test_ratings_dict = pickle.load(a_file)
print("Done")


print("Loading similarity matrix")
uibiu = np.genfromtxt(PATH+"similarities/UICIU.csv",delimiter=',')
uiu = np.genfromtxt(PATH+"similarities/UIU.csv",delimiter=',')
iui = np.genfromtxt(PATH+"similarities/IUI.csv",delimiter=',')
ibi = np.genfromtxt(PATH+"similarities/ICI.csv",delimiter=',')
print("Done")

###### evaluation
test_out_dict = dict()
print("getting outputs from model...")
with torch.no_grad():
  count = 0
  for key in test_ratings_dict:
    
    u_id = [key]*batch_size
    i_id = test_ratings_dict[key]

    uibiu_data = list()
    for i in u_id:
      uibiu_data.append(uibiu[i])

    uiu_data = list()
    for i in u_id:
      uiu_data.append(uiu[i])

    iui_data = list()
    for i in i_id:
      iui_data.append(iui[i])

    ibi_data = list()
    for i in i_id:
      ibi_data.append(ibi[i])

    uibiu_data = torch.tensor(uibiu_data).view(batch_size,num_users).unsqueeze(1).unsqueeze(1)
    uiu_data = torch.tensor(uiu_data).view(batch_size,num_users).unsqueeze(1).unsqueeze(1)
    iui_data = torch.tensor(iui_data).view(batch_size,num_items).unsqueeze(1).unsqueeze(1)
    ibi_data = torch.tensor(ibi_data).view(batch_size,num_items).unsqueeze(1).unsqueeze(1)

    out = net(uibiu_data.float(),uiu_data.float(),ibi_data.float(),iui_data.float()).detach().numpy()
    test_out_dict[key]=out
    
    if count%100 == 99:
      print(str((count*100)//len(test_ratings_dict))+"% completed")
    count += 1

a_file = open(PATH+"test_resuts.pkl", "wb")
pickle.dump(test_out_dict, a_file)
a_file.close()
