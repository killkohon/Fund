import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ANNModel(torch.nn.Module):
    def __init__(self,code,layer_shape):
        super(ANNModel, self).__init__()
        self.code=code
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cuda()
        if len(layer_shape)<3:
            raise Exception("layer shape length MUST greater than 2.")
        self.insize=layer_shape[0]
        self.outsize=layer_shape[-1]
        for i in range(len(layer_shape)-2):
            self.add_module("layer"+str(i),torch.nn.Sequential(torch.nn.Linear(layer_shape[i],layer_shape[i+1]).cuda(),torch.nn.Sigmoid()))
        self.add_module("layer"+str(i+1),torch.nn.Sequential(torch.nn.Linear(layer_shape[-2],layer_shape[-1]).cuda(),torch.nn.GELU()))
        self.layers=len(layer_shape)-1
    def forward(self, x):
        for i in range(self.layers):
            x=getattr(self,"layer"+str(i))(x)
        return x
    def model_attri(self):
        print(type(self))
        for i in range(self.layers):
            print(getattr(self,"layer"+str(i))[0].weight.device)
    #取输入和输出的参数数量
    def inout(self):
        pass
    def saveModel(self,file):
        torch.save(self.state_dict(),file)
    def train(self,ds,epoch=2000,learning_rate=0.000008,loss_threshold=0.1):
        dsloader=DataLoader(dataset=ds,batch_size=180, shuffle=True)
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        epoch_list = []
        loss_list = []
        for ep in range(epoch):
            loss_one_epoch = 0
            for i,data in enumerate(dsloader,0):
                inp,outp=data
                y_pred = self(inp)
                loss = criterion(y_pred, outp).cuda()
                loss_one_epoch += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_list.append(loss_one_epoch)
            epoch_list.append(ep)
            if ep%500 == 0:
                print('{} Epoch[{}/{}],loss:{:.6f}'.format(self.code, ep + 1, epoch, loss_one_epoch ))
            if loss_one_epoch<loss_threshold and ep>100:
                print('{} Epoch[{}/{}],loss:{:.6f}'.format(self.code, ep + 1, epoch, loss_one_epoch ))
                print("{}: {} < {} ".format(self.code,loss_one_epoch,loss_threshold))
                return
    def calculate(self,serial):
        if len(serial)<self.insize:
            return
        inp=torch.empty(0,self.insize,dtype=torch.float).cuda()
        for i in range(0,len(serial)-self.insize+1):
            piece=np.array([serial[i:i+self.insize]])
            inp=torch.cat((inp,torch.tensor(piece,dtype=torch.float).cuda()),0)
        return self(inp)
