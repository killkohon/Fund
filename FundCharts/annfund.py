import annmodel
import annserial

import numpy as np
import pandas as pd
import torch
import time
import os,re
import sqlite3
import threading
import collections
import sys

from dbutils.persistent_db import PersistentDB
from concurrent.futures import ThreadPoolExecutor
from queue import Queue


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
        self.threadPool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="annfund_")
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
            columns=[x[0] for x in cursor.description]
            data=list(cursor.fetchall())
            df={}
            for i in range(0,len(columns)):
                df[columns[i]]=[x[i] for x in data]
            # df = pd.DataFrame([list(x) for x in cursor.fetchall()], columns=[x[0] for x in cursor.description])
            return df
        except Exception as ex:
            print(ex)
        finally:
            cursor.close()
            conn.close()
    def _interrupt_train(self):
        self.annmodel.interrupt_train()
    def _train_model(self,epoch=1501,lr=0.005):
        starttime=time.time()
        data=self._net_value(800)
        # serial=data["unfixedvalue"].values.astype(float)[::-1]
        serial=data['unfixedvalue']
        ds=annserial.SerialDataset(np.array(serial))
        #self.annmodel.train(ds,epoch=epoch,learning_rate=0.000008,loss_threshold=0.03)
        self.annmodel.train(ds,epoch=epoch,learning_rate=lr,loss_threshold=0.0003)
        self.annmodel.saveModel("./ann/"+self.fundcode+".pth")
        endtime=time.time()
        print("\033[42m\033[37m"+self.fundcode+" duration:"+str(endtime-starttime)+"\033[0m")
    def _calculate(self,length=120):
        data=self._net_value(length)
        # serial=data["unfixedvalue"].values.astype(float)[::-1]
        serial=data['unfixedvalue']
        pred=self.annmodel.calculate(np.array(serial))
        return serial[self.annmodel.insize-1:],pred
    def _m2areal(self,length=120):
        data=self._net_value(length)
        # serial=data["unfixedvalue"].values.astype(float)[::-1]
        serial=data['unfixedvalue']
        if len(serial)>=120:
            maxV=[max(serial[i:i+20]) for i in range(100,len(serial)-19)]
            avgV=[round(sum(serial[i:i+20])/20,3) for i in range(100,len(serial)-19)]
            minV=[min(serial[i:i+20]) for i in range(100,len(serial)-19)]
            return maxV,avgV,minV
        return [],[],[]
        
    def print_meta(self):
        print(self.fundmeta)


fundcodes=[]
FundQueue=Queue()
funds=[]
Quit_flag=False
def train_fund(id=0,epoch=2001,lr=0.003):
    global fundcodes
    global Quit_flag
    while not Quit_flag:
        try:
            fundcode=FundQueue.get(timeout=60)
            fund=Fund(fundcode)
            funds.append(fund)
            # fund.print_meta()
            fund._train_model(epoch,lr)
            funds.remove(fund)
            del fund
        except Exception as e:
            pass
        if len(fundcodes)>0 and (not Quit_flag):
                code=fundcodes[0]
                fundcodes=fundcodes[1:]
                print(str(len(fundcodes))+" tasks left in queue.")
                FundQueue.put(code)
        else:
            print("no more task left @thread["+str(id)+"]")
            break
    print("thread finished")
    return

if __name__ == '__main__':  
    codepool=[]
    for file in os.scandir("./ann"):
        match=re.match(r"(\d{6})\.pth",file.name)
        if not match == None:
            codepool.append(match.group(1))
    #split tasks
    
    taskCount=5
    thisTask=1
    if len(sys.argv) >= 3:
        taskCount=int(sys.argv[1])
        thisTask=int(sys.argv[2])
    totalCount=len(codepool)
    if totalCount>=taskCount and taskCount>0:
        countPerTask=totalCount//taskCount
        fundcodes=codepool[(thisTask-1)*countPerTask:(thisTask*countPerTask if thisTask<taskCount else None )]
        fundcodes.reverse()
    if len(fundcodes)>0:
        print(fundcodes[0]+"-"+str(len(fundcodes))+"-"+fundcodes[-1])

    train_epoch=15001
    if len(sys.argv) >=4:
        train_epoch=int(sys.argv[3])
    thread_count=6
    if taskCount<=0:
        thread_count=thisTask
    threads=[]   
    for i in range(thread_count):
        if len(fundcodes)>0:
            code=fundcodes[0]
            fundcodes=fundcodes[1:]
            FundQueue.put(code)
        thread=threading.Thread(target=train_fund,args=(i,train_epoch,0.006))
        threads.append(thread)
        thread.start()
    while not Quit_flag:
        try:
            cmd=input(">>>:")
            if cmd=="quit":
                print("quitting....")
                Quit_flag=True
                for fund in funds:
                    fund._interrupt_train()
                break
            elif cmd=="+thread":
                thread=threading.Thread(target=train_fund,args=(thread_count,train_epoch,0.006))
                threads.append(thread)
                thread.start()
                thread_count+=1
                print("Thread:"+str(thread_count))
            else:
                match2=re.match(r"\d{6}",cmd)
                if not match2 == None:
                    fundcodes.insert(0,cmd)
                    print("add "+cmd +" into queue.["+str(len(fundcodes))+"]")
                match3=re.match(r"-(\d{6})",cmd)
                if not match3 == None:
                    code=match3[1]
                    if code in fundcodes:
                        fundcodes.remove(code)
                        print("remove "+code+" from queue.["+str(len(fundcodes))+"]")
                    else:
                        print(code+" isn't in queue.["+str(len(fundcodes))+"]")
        except KeyboardInterrupt:
            pass
    print("done")
