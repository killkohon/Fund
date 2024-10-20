import annmodel
import annserial

import numpy as np
import pandas as pd
import torch
import time
import os,re
import sqlite3

from dbutils.persistent_db import PersistentDB
from concurrent.futures import ThreadPoolExecutor


class Fund():
    def __init__(self,fundcode):
        self.dbpool=PersistentDB(
            creator=sqlite3,  # 使用链接数据库的模块
            maxusage=1000,  # 一个链接最多被使用的次数，None表示无限制
            setsession=[],  # 开始会话前执行的命令
            ping=0,  # ping MySQL服务端,检查服务是否可用
            closeable=False,  # conn.close()实际上被忽略，供下次使用，直到线程关闭，自动关闭链接，而等于True时，conn.close()真的被关闭
            threadlocal=None,  # 本线程独享值的对象，用于保存链接对象
            database="./fund.db"
        )
        self.fundcode=fundcode
        self.fundmeta = self._fund_meta()
        self.isopen = lambda x: x.find("开放") >= 0
        self.threadPool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fund_")
        self.annmodel = annmodel.ANNModel(fundcode,[100,50,50,50,20,3])
        if os.path.exists("./ann/"+fundcode+".pth"):
            self.annmodel.load_state_dict(torch.load("./ann/"+fundcode+".pth"))
        else:
            self.threadPool.submit(self._train_model)
    def _fund_meta(self):
        try:
            conn=self.dbpool.connection()
            cursor=conn.cursor()
            cursor.execute("select fundcode,fundname,category,company,holding from fund_meta where fundcode='{}'".format(self.fundcode))
            result=cursor.fetchone()
            if result is not None:
                meta={}
                meta["fundcode"]=self.fundcode
                meta["fundname"]=result[1]
                meta["category"]=result[2]
                meta["company"]=result[3]
                meta["holding"]=result[4]
                return meta
        except Exception as ex:
            print(ex)
        finally:
            cursor.close()
            conn.close()
    def _net_value(self,count=500):
        try:
            conn=self.dbpool.connection()
            cursor=conn.cursor()
            cursor.execute("select * from fund_values where fundcode='{}' order by vdate desc limit {}".format(self.fundcode,count))
            df = pd.DataFrame(list(cursor.fetchall()), columns=[x[0] for x in cursor.description])
            return df
        except Exception as ex:
            print(ex)
        finally:
            cursor.close()
            conn.close()
    def _train_model(self,epoch=1501):
        starttime=time.time()
        data=self._net_value(800)
        serial=data["unfixedvalue"].values.astype(float)[::-1]
        ds=annserial.SerialDataset(np.array(serial))
        #self.annmodel.train(ds,epoch=epoch,learning_rate=0.000008,loss_threshold=0.03)
        self.annmodel.train(ds,epoch=epoch,learning_rate=0.008,loss_threshold=0.001)
        self.annmodel.saveModel("./ann/"+self.fundcode+".pth")
        endtime=time.time()
        print("duration:"+str(endtime-starttime))
    def _calculate(self,length=120):
        data=self._net_value(length)
        serial=data["unfixedvalue"].values.astype(float)[::-1]
        pred=self.annmodel.calculate(np.array(serial))
        return serial[self.annmodel.insize-1:],pred
    def _m2areal(self,length=120):
        data=self._net_value(length)
        serial=data["unfixedvalue"].values.astype(float)[::-1]
        if len(serial)>=120:
            maxV=[max(serial[i:i+20]) for i in range(100,len(serial)-19)]
            avgV=[round(sum(serial[i:i+20])/20,3) for i in range(100,len(serial)-19)]
            minV=[min(serial[i:i+20]) for i in range(100,len(serial)-19)]
            return maxV,avgV,minV
        return [],[],[]
        
    def print_meta(self):
        print(self.fundmeta)

def train_fund(fundcode,epoch=2001):
    fund=Fund(fundcode)
    fund.print_meta()
    fund._train_model(epoch)

if __name__ == '__main__':
    threadPool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="thread_")
    pattern="(\d{6})\.pth"

    fundcodes=["519712","519674","540010","519212","610004","001410","001437","004243","000042"]
    # fundcodes=[]
    # for file in os.scandir("./ann"):
    #     match=re.match(pattern,file.name)
    #     if not match == None:
    #         fundcodes.append(match.group(1))

    for code in fundcodes:
        try:
            threadPool.submit(train_fund,code,8001)
        except Exception as ex:
            print(ex)

    # while input() != "exit":
    #     pass
    # fund=Fund("160639")
    # fund.annmodel.model_attri()
    # fund.print_meta()
    # fund._train_model()
    # fund=Fund("004598")
    # pred=fund._calculate(150)
    # cpred=pred.to("cpu").detach().numpy()

    # print(pred)
    # print([round(cpred[i][0],4) for i in range(0,len(cpred))])
