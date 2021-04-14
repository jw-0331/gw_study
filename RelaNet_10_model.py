#!/usr/bin/env python
# coding: utf-8

# In[102]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torchvision.datasets as dset   #토치비전은 영상처리용 데이터셋, 모델, 이미지 변환기가 들어있는 패키지 dset: 데이터 읽어오는 역할
import torchvision.transforms as transforms  #불러온 이미지를 필요에 따라 변환해주는 역할
from torch.utils.data import DataLoader  #데이터를 배치 사이즈로 묶어서 전달하거나 정렬 또는 섞는 등의 방법으로 데이터 모델에 전달해줌
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from random import random, shuffle
from torch.autograd import Variable
from pylab import *
from sklearn.model_selection import train_test_split


# In[103]:


"""
Motor data load
"""

#data : (109, 9600, 2)
data = scipy.io.loadmat('../JAEWON_RelationNet_study/DATA/motor_dataset/Motor_Imagery.mat')['data']


#스케일링 위해 2D로 변환
#data_2D : (2092800, 1)
data_2D = data.reshape(-1,1)


#스케일링
SDscaler = StandardScaler()
SDscaler.fit(data_2D)
scaled_data = SDscaler.transform(data_2D)


#데이터 나눠줌
data_size=60      #160*2*60=sub1마다 data 총 길이
batch_size=160    #batch size: 160(1초)
sub=109           #총 sub 수: 109


#Motor_data : (109*60, 160, 2)
Motor_data = scaled_data.reshape(sub*data_size, batch_size, 2)



#각 sub으로 data 분리
#Sub_data[i] : (160, 2, 60)
Sub_data=[]
for i in range(sub):
    Sub_data.insert(i, Motor_data[i*data_size:(i+1)*data_size, :, :])
    Sub_data[i] = Sub_data[i].reshape(-1,1)
    Sub_data[i] = Sub_data[i].reshape(batch_size, 2, data_size)
    Sub_data[i]=torch.tensor(Sub_data[i]) 


# In[104]:


"""
Support, Query, Test 나누기
"""

Support=[]
Query=[]
Test=[]



#Support : (160, 2, 30) * 109
#Query : (160, 2, 10) * 109
#Test : (160, 2, 20) * 109
for i in range(sub):
    Support.insert(i, Sub_data[i][:, :, 0:30])
    Query.insert(i, Sub_data[i][:, :, 30:40])
    Test.insert(i, Sub_data[i][:, :, 40:60])


# In[105]:


"""
Embedding function
"""

class embedding_function(nn.Module):
    def __init__(self):
        super(embedding_function, self).__init__()
        
        self.layer1 = nn.Sequential(nn.Conv1d(2, 16, kernel_size=3, stride=1, padding=0),
                                    nn.BatchNorm1d(16, momentum=1, affine=True),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=3, padding=0),
                                    nn.BatchNorm1d(32, momentum=1, affine=True),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=3, padding=0),
                                    nn.BatchNorm1d(64, momentum=1, affine=True),
                                    nn.ReLU())
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        
        return out
  

    
"""
Relation function
"""

class relation_function(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(relation_function, self).__init__()
        
        self.layer1 = nn.Sequential(nn.Conv1d(128, 16, kernel_size=3, padding=1),
                                    nn.BatchNorm1d(16, momentum=1, affine=True),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm1d(32, momentum=1, affine=True),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm1d(64, momentum=1, affine=True),
                                    nn.ReLU())
#         self.layer4 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, padding=1),
#                                     nn.BatchNorm1d(64, momentum=1, affine=True),
#                                     nn.ReLU())
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(160, -1)     #(batch_size, -1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        
        return out
    
    
    
"""
가중치 초기화
"""

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
        
        
 


# In[106]:


"""
각 query마다 모델 만듦
"""

model_cnt = 10  #만들 모델 갯수

feature_encoder=[]
relation_network=[]
feature_encoder_optim=[]
relation_network_optim=[]

for i in range(model_cnt):
    #각 sub 모델 생성
    feature_encoder.insert(i, embedding_function())
    relation_network.insert(i, relation_function(64*4, 8))
    
    
    #모델 초기화
    feature_encoder[i].apply(weights_init)
    relation_network[i].apply(weights_init)
    
    
    #모델마다 최적화(학습률:0.0001)
    feature_encoder_optim.insert(i, torch.optim.Adam(feature_encoder[i].parameters(), lr=0.0001))
    relation_network_optim.insert(i, torch.optim.Adam(relation_network[i].parameters(), lr=0.0001))


# In[107]:


"""
각 모델마다 Train
"""

for i in range(model_cnt):
    relation=[]    
    for episode in range(1000):
        for j in range(model_cnt):
            
            support_feature1 = feature_encoder[i](Support[j][:, :, 0:10].float())
            support_feature2 = feature_encoder[i](Support[j][:, :, 10:20].float())
            support_feature3 = feature_encoder[i](Support[j][:, :, 20:30].float())

            feature = torch.add(support_feature1, support_feature2)
            support_feature = torch.add(feature, support_feature3)
        
            query_feature = feature_encoder[i](Query[i].float())


            #feature_map 합침
            feature_pair = torch.cat((support_feature, query_feature), dim=1)

    
            #relation funcion에 대입
            relation.insert(j, relation_network[i](feature_pair))

            mse = nn.MSELoss()
            
            if i == j:
                label = torch.tensor(1, dtype=torch.float32)
            else:
                label = torch.tensor(0, dtype=torch.float32)
                
            loss = mse(relation[j], label)   #정답이 1이 나와야함
    
            feature_encoder[i].zero_grad()
            relation_network[i].zero_grad()
    
            loss.backward()
    
            feature_encoder_optim[i].step()
            relation_network_optim[i].step()
        

            if (episode+1)%200 == 0:
                predict_label = torch.max(relation[j].data)
                print("\tepisode:", episode+1, "sub:", j, "최대 예측값:",predict_label, "loss", loss.item())
                
                if (j+1) == model_cnt:
                    print("\t------------------------------------------------------------------------")
                    
                if ((episode+1)==1000 and (j+1) == model_cnt):
                    print("\n")
 


# In[128]:


"""
Test
"""

model = 1
request = 3
for i in range(model_cnt):
    support1_ = feature_encoder[model](Support[i][:, :, 0:10].float())
    support2_ = feature_encoder[model](Support[i][:, :, 10:20].float())
    support3_ = feature_encoder[model](Support[i][:, :, 20:30].float())

    feature = torch.add(support1_, support2_)
    support_feature = torch.add(feature, support3_)

    test_feature = feature_encoder[model](Test[request][:,:,10:20].float())


    #feature_map 합침
    feature_pair = torch.cat((support_feature, test_feature), dim=1)


    #relation funcion에 대입
    relation = relation_network[model](feature_pair)

    predict_label = torch.max(relation.data)
    print(predict_label)
                                                                                                                  


# In[ ]:
