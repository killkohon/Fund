import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import annmodel


class SerialDataset(Dataset):
    def __init__(self,serial,isize=100,osize=20):
        if len(serial)<isize+osize:
            raise Exception("")
        self.x_data=torch.empty(0,isize,dtype=torch.float).cuda()
        self.y_data=torch.empty(0,3,dtype=torch.float).cuda()
        for i in range(0,len(serial)-isize-osize+1):
            input=np.array([serial[i:i+isize]])
            self.x_data=torch.cat((self.x_data,torch.tensor(input,dtype=torch.float).cuda()),0)
            outRange=serial[i+isize:i+isize+osize]
            outMax=max(outRange)
            outAvg=sum(outRange)/osize
            outMin=min(outRange)
            self.y_data=torch.cat((self.y_data,torch.tensor([[outMax,outAvg,outMin]],dtype=torch.float).cuda()),0)
        self.len=len(self.x_data)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len
