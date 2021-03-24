#!/usr/bin/env python
# coding: utf-8

# In[445]:


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
from torch.autograd import Variable
from pylab import *


# In[446]:


"""
Support Data Load
"""


support_data_BSW = scipy.io.loadmat('../JAEWON_RelationNet_study/Data/BSW_200629_fin.mat')['data']

support_data_BSW = torch.tensor(support_data_BSW)
print(support_data_BSW.shape)


#shape을 다시 잡기 위해 tensor를 모두 풀어줌
support_BSW = support_data_BSW.reshape(-1,1)
support_BSW = support_BSW[0:698400, :]

SDscaler = StandardScaler()
SDscaler.fit(support_BSW)
support_BSW = SDscaler.transform(support_BSW)
support_BSW = torch.tensor(support_BSW)
print(support_BSW)

support_BSW = support_BSW.reshape(1,2,349200)
support_BSW = support_BSW.reshape(1800,2,194)
print(support_BSW.shape)


# In[327]:


# """
# Query Data Load
# """


# query_data = scipy.io.loadmat('../JAEWON_RelationNet_study/Data/BSW_200701_fin.mat')['data']

# query_data = torch.tensor(query_data)
# print(query_data.shape)

# query_data = query_data.reshape(-1,1)
# query_data = query_data[0:698400, :]

# query_data = SDscaler.transform(query_data)

# query_data = query_data.reshape(1,2,349200)
# query_data = query_data.reshape(1800,2,194)
# print(query_data.shape)


# In[447]:


"""
Query Data Load
"""

query_data = []
BSW_1 = scipy.io.loadmat('../JAEWON_RelationNet_study/Data/BSW_200701_fin.mat')['data']
EHS_1 = scipy.io.loadmat('../JAEWON_RelationNet_study/Data/EHS_200701_fin.mat')['data']
HSW_1 = scipy.io.loadmat('../JAEWON_RelationNet_study/Data/HSW_200701_fin.mat')['data']
BSW_2 = scipy.io.loadmat('../JAEWON_RelationNet_study/Data/BSW_200703_fin.mat')['data']
HSW_2 = scipy.io.loadmat('../JAEWON_RelationNet_study/Data/HSW_200702_fin.mat')['data']
BSW_3 = scipy.io.loadmat('../JAEWON_RelationNet_study/Data/BSW_200713_fin.mat')['data']

query_data.insert(0,torch.tensor(BSW_1))
query_data.insert(1,torch.tensor(EHS_1))
query_data.insert(2,torch.tensor(HSW_1))
query_data.insert(3,torch.tensor(BSW_2))
query_data.insert(4,torch.tensor(HSW_2))
query_data.insert(5,torch.tensor(BSW_3))

for i in range(6):
    query_data[i] = query_data[i].reshape(-1,1)
    query_data[i] = query_data[i][0:698400, :]

    query_data[i] = SDscaler.transform(query_data[i])
    query_data[i] = torch.tensor(query_data[i])
    
    query_data[i] = query_data[i].reshape(1,2,349200)
    query_data[i] = query_data[i].reshape(1800,2,194)
    print(query_data[i].shape)


# In[448]:


"""
Embedding function
"""


class embedding_function(nn.Module):
    def __init__(self):
        super(embedding_function, self).__init__()
        
        self.layer1 = nn.Sequential(nn.Conv1d(2, 64, kernel_size=3, padding=0),
                                   nn.BatchNorm1d(64, momentum=1, affine=True),
                                    nn.ReLU(),
                                   nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, padding=0),
                                   nn.BatchNorm1d(64, momentum=1, affine=True),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2))
        self.layer3 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(64, momentum=1, affine=True),
                                   nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(64, momentum=1, affine=True),
                                   nn.ReLU())
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        return out
    


# In[449]:


"""
Relation function
"""


class relation_function(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(relation_function, self).__init__()
        
        self.layer1 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(64, momentum=1, affine=True),
                                    nn.ReLU(),
                                   nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(64, momentum=1, affine=True),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(1800, -1)     #(batch_size, -1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        
        return out
    


# In[467]:


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


# In[468]:


#function 생성
feature_encoder = embedding_function()
relation_network = relation_function(64*11, 8)   #(batchsize 제외한 크기, 8) -> 즉, 원래 데이터크기: (1800, 64, 11)


#fucntion 가중치 초기화
feature_encoder.apply(weights_init)
relation_network.apply(weights_init)


feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=0.001)
feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=10, gamma=0.5)
relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=0.001)
relation_network_scheduler = StepLR(relation_network_optim, step_size=10, gamma=0.5)

label = torch.tensor([1, 0, 0, 1, 0, 1], dtype=torch.float32)
print(label)


# In[469]:


for i in range(6):
    for episode in range(20):
        #train, test 데이터 embeddinf function 대입
        support_BSW_feature = feature_encoder(support_BSW.float())
        query_data_feature = feature_encoder(query_data[i].float())


        #feature_map 합침
        feature_pair = torch.cat((support_BSW_feature, query_data_feature), dim=1)

    
        #relation funcion에 대입
        relation = relation_network(feature_pair)
        
        mse = nn.MSELoss()
        loss = mse(relation, label[i])   #정답이 1이 나와야함
    
        feature_encoder.zero_grad()
        relation_network.zero_grad()
    
        loss.backward()
    
    
        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)
    
        feature_encoder_optim.step()
        relation_network_optim.step()
        
        if (i%1==0)&((episode+1)==10):
            print("\n",i, "query data")
        if (episode+1)%10 == 0:
            predict_label = torch.max(relation.data)
            print("\tepisode:", episode+1, "  label", label[i].item(), "최대 예측값:",predict_label, "loss", loss.item())
        


# In[333]:


"""
Train Data Load
"""


train_data = scipy.io.loadmat('../JAEWON_RelationNet_study/Data/BSW_200703_fin.mat')['data']

train_data = torch.tensor(train_data)
print(train_data.shape)


#shape을 다시 잡기 위해 tensor를 모두 풀어줌
train = train_data.reshape(-1,1)
train = train.reshape(1,2,350000)
print(train.shape)

"""
Test Data Load
"""


test_data = scipy.io.loadmat('../JAEWON_RelationNet_study/Data/EHS_200702_fin.mat')['data']

test_data = torch.tensor(test_data)
print(test_data.shape)

test = test_data.reshape(-1,1)
test = test.reshape(1,2,350000)
print(test.shape)


train_feature = feature_encoder(train.float())
test_feature = feature_encoder(test.float())


#feature_map 합침
feature_pair = torch.cat((train_feature, test_feature), dim=1)


#relation funcion에 대입
relation = relation_network(feature_pair)
print(relation.item())

