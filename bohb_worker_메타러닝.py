#!/usr/bin/env python
# coding: utf-8

# In[71]:


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except:
    raise ImportError("For this example you need to install pytorch")
    
try:
    import torchvision
    import torchvision.transforms as transforms
except:
    raise ImportError("For this example you need to install pytorch-vision.")

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import logging
logging.basicConfig(level=logging.DEBUG)
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

embedding_output_filter=0
input_length=0

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

Support=[]
Query=[]
Test=[]
 
"""
BOHB에서 나온 parameter를 넣은 결과 확인
"""
class PytorchWorker(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
            
            
        """
        Support, Query, Test 나누기
        """


        #Support : (160, 2, 30) * 109
        #Query : (160, 2, 10) * 109
        #Test : (160, 2, 20) * 109
        for i in range(sub):
            Support.insert(i, Sub_data[i][:, :, 0:30])
            Query.insert(i, Sub_data[i][:, :, 30:40])
            Test.insert(i, Sub_data[i][:, :, 40:60])
        
        
    def compute(self, config, budget, working_directory, *args, **kwargs):
        model_cnt = 10  #만들 모델 갯수

        feature_encoder=[]
        relation_network=[]
        feature_encoder_optim=[]
        relation_network_optim=[]
        
        for i in range(model_cnt):
            feature_encoder.insert(i, embedding_function(num_embedding_layers=config['num_embedding_layers'],
                                             num_filters_1=config['num_filters_1'],
                                             num_filters_2=config['num_filters_2'] if 'num_filters_2' in config else None,
                                             num_filters_3=config['num_filters_3'] if 'num_filters_3' in config else None,
                                             num_filters_4=config['num_filters_4'] if 'num_filters_4' in config else None,
                                             dropout_rate=config['dropout_rate'],
                                             kernel_size=3))
        
            relation_network.insert(i, relation_function(num_relation_layers=config['num_relation_layers'], 
                                             embedding_output_filter=embedding_output_filter*2,
                                             input_length = input_length,
                                             num_rela_filters_1=config['num_rela_filters_1'] if 'num_rela_filters_1' in config else None, 
                                             num_rela_filters_2=config['num_rela_filters_2'] if 'num_rela_filters_2' in config else None, 
                                             num_rela_filters_3=config['num_rela_filters_3'] if 'num_rela_filters_3' in config else None, 
                                             num_rela_filters_4=config['num_rela_filters_4'] if 'num_rela_filters_4' in config else None, 
                                             rela_dropout_rate=config['rela_dropout_rate'], 
                                             kernel_size=3, 
                                             num_fc_units=config['num_fc_units']))
        
            feature_encoder[i].apply(weights_init)
            relation_network[i].apply(weights_init)
        
                                        
        
            if config['optimizer'] == 'Adam':
                feature_encoder_optim.insert(i, torch.optim.Adam(feature_encoder[i].parameters(), lr=config['lr']))
                relation_network_optim.insert(i, torch.optim.Adam(relation_network[i].parameters(), lr=config['lr']))
            else:            
                feature_encoder_optim.insert(i, torch.optim.SGD(feature_encoder[i].parameters(), lr=config['lr'], momentum=config['sgd_momentum']))
                relation_network_optim.insert(i, torch.optim.SGD(relation_network[i].parameters(), lr=config['lr'], momentum=config['sgd_momentum']))            
          
        #첫번째 사람에 대한 모델 
        for i in range(model_cnt):
            relation=[]    
            for episode in range(int(budget)):
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
        
                   
                    if (episode+1)==budget:
                        predict_label = torch.max(relation[j].data)
                    
                        print("sub:", j, "최대 예측값:",predict_label, "loss", loss.item())
                    
                        if (j+1) == model_cnt:
                            print("\t---------------------------------------------------------")
                
                
    def get_configspace(self):
        cs = CS.ConfigurationSpace()
        
        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)
        
        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])
        
        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9, log=False)
        
        cs.add_hyperparameters([lr, optimizer, sgd_momentum])
        
        cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_condition(cond)
        

        num_embedding_layers =  CSH.UniformIntegerHyperparameter('num_embedding_layers', lower=1, upper=4, default_value=2)

        num_filters_1 = CSH.UniformIntegerHyperparameter('num_filters_1', lower=4, upper=64, default_value=16, log=True)
        num_filters_2 = CSH.UniformIntegerHyperparameter('num_filters_2', lower=4, upper=64, default_value=16, log=True)
        num_filters_3 = CSH.UniformIntegerHyperparameter('num_filters_3', lower=4, upper=64, default_value=16, log=True)        
        num_filters_4 = CSH.UniformIntegerHyperparameter('num_filters_4', lower=4, upper=64, default_value=16, log=True)        
        
        cs.add_hyperparameters([num_embedding_layers, num_filters_1, num_filters_2, num_filters_3, num_filters_4])
   
        cond = CS.GreaterThanCondition(num_filters_2, num_embedding_layers, 1)
        cs.add_condition(cond)

        cond = CS.GreaterThanCondition(num_filters_3, num_embedding_layers, 2)
        cs.add_condition(cond)
        
        cond = CS.GreaterThanCondition(num_filters_4, num_embedding_layers, 3)
        cs.add_condition(cond)    

        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.5, log=False)

        cs.add_hyperparameters([dropout_rate])
        
        num_relation_layers =  CSH.UniformIntegerHyperparameter('num_relation_layers', lower=1, upper=4, default_value=2)

        num_rela_filters_1 = CSH.UniformIntegerHyperparameter('num_rela_filters_1', lower=4, upper=64, default_value=16, log=True)
        num_rela_filters_2 = CSH.UniformIntegerHyperparameter('num_rela_filters_2', lower=4, upper=64, default_value=16, log=True)
        num_rela_filters_3 = CSH.UniformIntegerHyperparameter('num_rela_filters_3', lower=4, upper=64, default_value=16, log=True)        
        num_rela_filters_4 = CSH.UniformIntegerHyperparameter('num_rela_filters_4', lower=4, upper=64, default_value=16, log=True)        
                

        cs.add_hyperparameters([num_relation_layers, num_rela_filters_1, num_rela_filters_2, num_rela_filters_3, num_rela_filters_4])
   
        cond = CS.GreaterThanCondition(num_rela_filters_2, num_relation_layers, 1)
        cs.add_condition(cond)

        cond = CS.GreaterThanCondition(num_rela_filters_3, num_relation_layers, 2)
        cs.add_condition(cond)

        cond = CS.GreaterThanCondition(num_rela_filters_4, num_embedding_layers, 3)
        cs.add_condition(cond)
        
       
        rela_dropout_rate = CSH.UniformFloatHyperparameter('rela_dropout_rate', lower=0.0, upper=0.9, default_value=0.5, log=False)
        num_fc_units = CSH.UniformIntegerHyperparameter('num_fc_units', lower=8, upper=256, default_value=32, log=True)

        cs.add_hyperparameters([rela_dropout_rate, num_fc_units])   
        
        return cs
         
"""
BOHB
"""
class embedding_function(torch.nn.Module):
    def __init__(self, num_embedding_layers, num_filters_1, num_filters_2, num_filters_3, num_filters_4, dropout_rate, kernel_size):
        super(embedding_function, self).__init__()
        
        self.layer1 = nn.Sequential(nn.Conv1d(2, num_filters_1, kernel_size=kernel_size),
                                    nn.BatchNorm1d(num_filters_1, momentum=1, affine=True),
                                    nn.ReLU())
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        global embedding_output_filter
        embedding_output_filter=num_filters_1

        global input_length
        input_length=10-2
        
        if num_embedding_layers > 1:
            self.layer2 = nn.Sequential(nn.Conv1d(num_filters_1, num_filters_2, kernel_size=kernel_size),
                                    nn.BatchNorm1d(num_filters_2, momentum=1, affine=True),
                                    nn.ReLU())
            embedding_output_filter=num_filters_2
            input_length = input_length-2
         
        if num_embedding_layers > 2:
            self.layer3 = nn.Sequential(nn.Conv1d(num_filters_2, num_filters_3, kernel_size=kernel_size),
                                    nn.BatchNorm1d(num_filters_3, momentum=1, affine=True),
                                    nn.ReLU())
            embedding_output_filter=num_filters_3
            input_length = input_length-2
            
        if num_embedding_layers > 3:
            self.layer4 = nn.Sequential(nn.Conv1d(num_filters_3, num_filters_4, kernel_size=kernel_size),
                                    nn.BatchNorm1d(num_filters_4, momentum=1, affine=True),
                                    nn.ReLU())
            embedding_output_filter=num_filters_4
            input_length = input_length-2
            

        self.dropout = nn.Dropout(p=dropout_rate)
        
    
    def forward(self, x):
        out = self.layer1(x)
        
        if not self.layer2 is None:
            out = self.layer2(out)
            
        if not self.layer3 is None:
            out = self.layer3(out)
            
        if not self.layer4 is None:
            out = self.layer4(out)
            
        out = self.dropout(out)
        
        return out
    
    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))
    
    
    
class relation_function(torch.nn.Module):
    def __init__(self, num_relation_layers, embedding_output_filter, input_length, num_rela_filters_1, num_rela_filters_2, num_rela_filters_3, num_rela_filters_4, rela_dropout_rate, kernel_size, num_fc_units):
        super(relation_function, self).__init__()
        
        self.input_num_filters = embedding_output_filter
        
        self.layer1 = nn.Sequential(nn.Conv1d(self.input_num_filters, num_rela_filters_1, kernel_size=kernel_size, padding=1),
                                    nn.BatchNorm1d(num_rela_filters_1, momentum=1, affine=True),
                                    nn.ReLU())
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        
        num_output_filters = num_rela_filters_1
        out_length = input_length
        
        if num_relation_layers > 1:
            self.layer2 = nn.Sequential(nn.Conv1d(num_rela_filters_1, num_rela_filters_2, kernel_size=kernel_size, padding=1),
                                    nn.BatchNorm1d(num_rela_filters_2, momentum=1, affine=True),
                                    nn.ReLU())
            num_output_filters = num_rela_filters_2
            out_length = out_length
         
        if num_relation_layers > 2:
            self.layer3 = nn.Sequential(nn.Conv1d(num_rela_filters_2, num_rela_filters_3, kernel_size=kernel_size, padding=1),
                                    nn.BatchNorm1d(num_rela_filters_3, momentum=1, affine=True),
                                    nn.ReLU())   
            num_output_filters = num_rela_filters_3
            out_length = out_length   
            
        if num_relation_layers > 3:     
            self.layer4 = nn.Sequential(nn.Conv1d(num_rela_filters_3, num_rela_filters_4, kernel_size=kernel_size, padding=1),
                                    nn.BatchNorm1d(num_rela_filters_4, momentum=1, affine=True),
                                    nn.ReLU())   
            num_output_filters = num_rela_filters_4
            out_length = out_length           
        
        self.dropout = nn.Dropout(p=rela_dropout_rate)
        
        
        self.conv_output_size = num_output_filters*out_length
        self.fc1 = nn.Linear(self.conv_output_size, num_fc_units)
        self.fc2 = nn.Linear(num_fc_units, 1)
        
    
    def forward(self, x):
        
        out = self.layer1(x)
        
        if not self.layer2 is None:
            out = self.layer2(out)
            
        if not self.layer3 is None:
            out = self.layer3(out)
            
        if not self.layer4 is None:
            out = self.layer4(out)
            
        out = self.dropout(out)
        
        out = out.view(160, -1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        
        return out
    
    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))
    
 
    
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
        
        
        
if __name__ == "__main__":
    
    # 작업자 시작
    worker = PytorchWorker(run_id='0')
    
    # 옵티마이저 실행
    cs = worker.get_configspace()

    # 결과 분석
    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=200, working_directory='.')
    #print(res)

