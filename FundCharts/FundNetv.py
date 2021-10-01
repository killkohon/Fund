#!/usr/bin/python
# -*- coding: UTF-8 -*-

import datetime as dt
from bs4 import BeautifulSoup
import requests
import pandas as pd 
import numpy as np 
import sqlite3
from dbutils.persistent_db import PersistentDB
import matplotlib.pyplot as plt
from scipy import stats
from numpy.linalg import solve
import urllib3 
import urllib.parse 
import json
import sys
import polyfit
from concurrent.futures import ThreadPoolExecutor

plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus']=False

class ParameterException(Exception):
	def __init__(self,value):
		self.value=value
	def __str__(self):
		print("Parameter Exception!")
		return repr(self.value)
class fundnetv:
	def __init__(self):
		self.PooL = PersistentDB(
		    creator = sqlite3,  #使用链接数据库的模块
			maxusage = None, #一个链接最多被使用的次数，None表示无限制
			setsession = [], #开始会话前执行的命令
			ping = 0, #ping MySQL服务端,检查服务是否可用
 			closeable = False, #conn.close()实际上被忽略，供下次使用，直到线程关闭，自动关闭链接，而等于True时，conn.close()真的被关闭
			threadlocal = None, # 本线程独享值的对象，用于保存链接对象
			database = "./fund.db"
			)
		self.isopen = lambda x: x.find("开放")>=0
		self.currentfund = None
		self.fundnames={}
		self.threadPool=ThreadPoolExecutor(max_workers=5, thread_name_prefix="thread_")
	def replacenan(self,x,y):
		if np.isnan(x):
			return y
		else:
			return x
	def fundname(self,tickcode):
		if tickcode in self.fundnames:
			return self.fundnames[tickcode]
		else:
			try:
				conn = self.PooL.connection()
				cursor = conn.cursor()
				cursor.execute("select fundname from fund_meta where fundcode='{}'".format(tickcode))
				result=cursor.fetchone()
				if result is not None:
					self.fundnames[tickcode]=result[0]
					return result[0]
				else:
					return "nil"
			except Exception as ex:
				print(ex)
			finally:
				cursor.close()
				conn.close()
	def fund_in_days(self,tickCode, startDate,days):
		endDate = dt.datetime.strptime(startDate, "%Y-%m-%d") + dt.timedelta(days)
		url = "http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code={}&sdate={}&edate={}&per={}".format(tickCode,startDate, endDate,days)
		response = requests.get(url)
		soup = BeautifulSoup(response.content, "lxml")
		df = pd.DataFrame()
		table_heads = []
		for head in soup.findAll("th"):
			table_heads.append(head.contents[0])
		table_rows = []
		for rows in soup.findAll("tbody")[0].findAll("tr"):
			table_records = []
			for record in rows.findAll('td'):
				val = record.contents
				# 处理空值
				if len(val) == 0:
					table_records.append(np.nan)
				else:
					if val[0] == "暂无数据!":
						print("暂无数据!")
						return df
					table_records.append(val[0])
			table_rows.append(table_records)
		# 写入DataFrame
		table_rows = np.array(table_rows)
		for col,col_name in enumerate(table_heads):
			df[col_name] = table_rows[:,col]
		return df
	def grabmeta(self,tickcode):
		url="https://fundsuggest.eastmoney.com/FundSearch/api/FundSearchAPI.ashx?callback=&m=1&key={}".format(tickcode)
		response = requests.get(url)
		meta= json.loads(urllib.parse.unquote(response.content.decode()))
		return tickcode,meta
	def savemeta(self,tickcode,name,category,categoryname,company,manager):
		try:
			conn = self.PooL.connection()
			cursor = conn.cursor()
			data=(tickcode,name,category,categoryname,company,manager)
			print("To save:{}".format(data))
			cursor.execute("insert into fund_meta(fundcode,fundname,category,categoryname,company,manager)values('%s','%s','%s','%s','%s','%s')"%data)
			result = cursor.fetchall()
			conn.commit()
		except Exception as ex:
			print("exception in savedata")
			print(ex)
		finally:
			cursor.close()
			conn.close()
	def grabmetafromdb(self):
		try:
			conn= self.PooL.connection()
			cursor=conn.cursor()
			cursor.execute("select distinct(fundcode) from fund_values where fundcode not in (select fundcode from fund_meta)")
			__result=cursor.fetchone()
			while __result is not None:
				print("fundcode:{}".format(__result[0]))
				code,meta=self.grabmeta(__result[0])
				for data in meta["Datas"]:
					if (data["CATEGORY"] == 700 and data["CODE"] == code) :
						self.savemeta(code,data["NAME"],data["CATEGORY"],data["CATEGORYDESC"],data["FundBaseInfo"]["JJGS"],data["FundBaseInfo"]["JJJL"])
				__result=cursor.fetchone()
		except Exception as ex :
			print("except at find the last vdate")
			print(ex)
		finally:
			cursor.close()
			conn.close()

	def grabfromdb(self):
		try:
			conn= self.PooL.connection()
			cursor=conn.cursor()
			cursor.execute("select distinct(fundcode) from fund_meta")
			__result=cursor.fetchone()
			while __result is not None:
				print("fundcode:{}".format(__result[0]))
				future=self.threadPool.submit(self.grabnetv,__result[0])
				future.add_done_callback(self.callback_grabnetv)
				__result=cursor.fetchone()
		except Exception as ex :
			print("except at find the last vdate")
			print(ex)
		finally:
			cursor.close()
			conn.close()
	def callback_grabnetv(self,future):
		pass
	def setholding(self,tickcode,hold=True):
		try:
			conn= self.PooL.connection()
			cursor=conn.cursor()
			if hold:
				cursor.execute("update fund_meta set holding=1 where fundcode='{}'".format(tickcode))
			else:
				cursor.execute("update fund_meta set holding=0 where fundcode='{}'".format(tickcode))
			conn.commit()
		except Exception as ex :
			print("except at setholding({})".format(tickcode))
			print(ex)
		finally:
			cursor.close()
			conn.close()
	def grabnetv(self,tickcode):
		nextdate='2018-10-01'
		__isnewfund=True
		try:
			conn= self.PooL.connection()
			cursor=conn.cursor()
			cursor.execute("select max(vdate) as lastdate from fund_values where fundcode='{}'".format(tickcode))
			__result=cursor.fetchone()
			if __result[0] is not None:
				nextdate=(dt.datetime.strptime(__result[0],'%Y-%m-%d')+dt.timedelta(1)).__format__('%Y-%m-%d')
				__isnewfund=False
		except Exception as ex :
			print("except at grabnetv({})".format(tickcode))
			print(ex)
		finally:
			cursor.close()
			conn.close()
		GoAhead= True
		while GoAhead:
			#print("Next date: {}".format(nextdate))
			df=self.fund_in_days(tickcode,nextdate,8)
			data=df.values
			if (np.shape(data)[0]>0):
				row=np.shape(data)[0]-1
				while row>=0:
					self.savedata(tickcode,data[row][0],float(data[row][1]),float(data[row][2]),buyable=self.isopen(data[row][4]),salable=self.isopen(data[row][5]),dividend=data[row][6])
					row-=1
				nextdate=(dt.datetime.strptime(data[0][0],"%Y-%m-%d")+dt.timedelta(1)).__format__("%Y-%m-%d")
			else:
				__tempdate=dt.datetime.strptime(nextdate,"%Y-%m-%d")
				if __tempdate.__lt__(dt.datetime.now()-dt.timedelta(7)):
					nextdate=(__tempdate+dt.timedelta(7)).__format__("%Y-%m-%d")
				else:
					GoAhead=False
		#拉取元数据
		if __isnewfund == True :
			code,meta=self.grabmeta(tickcode)
			for data in meta["Datas"]:
				if (data["CATEGORY"] == 700 and data["CODE"] == code) :
					self.savemeta(code,data["NAME"],data["CATEGORY"],data["CATEGORYDESC"],data["FundBaseInfo"]["JJGS"],data["FundBaseInfo"]["JJJL"])
	def savedata(self,tickcode,valuedate,netv,unfixednetv,buyable=True,salable=True,dividend=""):
		try:
			conn = self.PooL.connection()
			cursor = conn.cursor()
			data=(tickcode,valuedate,netv,self.replacenan(unfixednetv,netv),buyable,salable,dividend)
			print("To save:{}".format(data))
			cursor.execute("insert into fund_values(fundcode,vdate,netvalue,unfixedvalue,buyable,salable,dividend)values('%s','%s',%f,%f,%d,%d,'%s')"%data)
			result = cursor.fetchall()
			conn.commit()
		except Exception as ex :
			print("exception in savedata")
			print(ex)
		finally:
			cursor.close()
			conn.close()
	def loadnetv(self,tickcode,count=500):
		try:
			conn=self.PooL.connection()
			cursor=conn.cursor()
			cursor.execute("select * from (select * from fund_values where fundcode='{}' order by vdate desc limit {}) order by vdate".format(tickcode,count))
			df=pd.DataFrame(list(cursor.fetchall()),columns=[x[0] for x in cursor.description ])
			return df 
		except Exception as ex :
			print(ex)
		finally:
			cursor.close()
			conn.close()
	def movingaverage(self,source,window=20):
		count=source.size
		target=np.zeros(count,dtype=float)
		sum=0.0
		__count=0
		for i in range(count):
			sum+=source[i]
			__count+=1
			if i>=window:
				sum-=source[i-window]
				__count=window
			target[i]=sum/__count
		return target
	def continousaverage(self,source):
		count=source.size
		target=np.zeros(count,dtype=float)
		__sum=0.0
		__count=0
		for i in range(count):
			__sum+=source[i]
			__count+=1
			target[i]=__sum/__count
		return target
	def showfigure(self,tickcode,window=45,figure=None):
		df=self.loadnetv(tickcode)
		lastdate=df['vdate'].values[-1]
		netv=df['unfixedvalue'].values.astype(float)
		ma=self.movingaverage(netv,window)
		ca=self.continousaverage(netv)
		dv=netv-ma
		madv=self.movingaverage(dv,window)
		if figure == None :
			figure=plt.figure()
		figure.subplots_adjust(left=0.03,right=0.99,wspace=0.05, hspace=0.05, bottom=0.05,top=0.95)
		figure.suptitle("{} {} {}".format(tickcode,self.fundname(tickcode),lastdate))
		x_axis=np.linspace(1,netv.size,netv.size)


		pf=polyfit.polyfit()
		pf.fitting(x_axis,netv)
		fitting=pf.calc(x_axis)
		gradient=pf.gradient(x_axis)
		dpf=netv-fitting
		#sfitting=self.movingaverage(dpf,window)
		cdf=stats.norm.cdf((madv-np.mean(madv))/np.std(madv))
		cdf2=stats.norm.cdf((dv-np.mean(dv))/np.std(dv))
		cdf3=self.movingaverage(stats.norm.cdf((dpf-np.mean(dpf))/np.std(dpf)))
		cdf4=self.movingaverage(stats.norm.cdf((dv-np.mean(dv))/np.std(dv)))
		#逐日计cdf
		cdf5=[]
		for i in range(2,len(madv)):
			cdf5.append(stats.norm.cdf((madv[:-(len(madv)-i)]-np.mean(madv[:-(len(madv)-i)]))/np.std(madv[:-(len(madv)-i)]))[-1])
		__temp=cdf5[0]
		cdf5.insert(0,__temp)
		cdf5.insert(0,__temp)

		p1=figure.add_subplot(3,1,1)
		p2=figure.add_subplot(3,1,2)
		p3=figure.add_subplot(3,1,3)
		p1.plot(x_axis,netv,label='netv[{}]'.format(round(netv[-1],3)))
		p1.plot(x_axis,ma,label='ma[{}]'.format(round(ma[-1],3)))
		p1.plot(x_axis,ca,label='cont. average[{}]'.format(round(ca[-1],3)))
		p1.plot(x_axis,fitting,label='fit[{}]'.format(round(fitting[-1],4)))
		p1.plot(x_axis,np.min(netv)+(np.max(netv)-np.min(netv))*(gradient-np.min(gradient))/(np.max(gradient)-np.min(gradient)),label='grad[{}]'.format(round(gradient[-1],6)))
		p2.plot(x_axis,dv,label='dv(netv-ma)')
		p2.plot(x_axis,madv,label='madv')
		p3.plot(x_axis,(netv-np.min(netv))/(np.max(netv)-np.min(netv)),label='reg. netv')
		p3.plot(x_axis,cdf,label='CDF(madv)[{}]'.format(round(cdf[-1],4)))

		p3.plot(x_axis,cdf2,label='CDF(dv)[{}]'.format(round(cdf2[-1],2)))
		p3.plot(x_axis,cdf3,label='CDF(safit)[{}]'.format(round(cdf3[-1],2)))
		p3.plot(x_axis,cdf4,label='MA(cdfdv)[{}]'.format(round(cdf4[-1],2)))
		p3.plot(x_axis,cdf5,label='CDFp(madv)[{}]'.format(round(cdf5[-1],4)))
		
		p1.grid()
		p2.grid()
		p3.grid()
		p1.legend(loc=2)
		p2.legend(loc=2)
		p3.legend(loc=2)
  
	def showdata(self,tickcode,window=45):
		df=self.loadnetv(tickcode)
		netv=df['unfixedvalue'].values.astype(float)
		val=df['netvalue'].values.astype(float)
		sa=self.movingaverage(netv,window)
		ca=self.continousaverage(netv)
		dv=netv-sa
		sadv=self.movingaverage(dv,window)
		x_axis=np.linspace(1,netv.size,netv.size)
		pf=polyfit()
		pf.fitting(x_axis,netv)
		fitting=pf.calc(x_axis)
		gradient=pf.gradient(x_axis)
		cdf=stats.norm.cdf((sadv-np.mean(sadv))/np.std(sadv))
		cdf2=stats.norm.cdf((dv-np.mean(dv))/np.std(dv))
		print("{} \t {}".format(tickcode,self.fundname(tickcode)))
		print("VAL: {}".format(netv[-10:]))
		print(" FV: {}".format(val[-10:]))
		print(" SA: {}".format(sa[-10:]))
		print("FIT: {}".format(fitting[-10:]))
		print("Grad: {}").format(gradient[-10:])
		print("CDF: {}".format(cdf[-10:]))
		print("CDF(sadv): {}".format(cdf2[-10:]))
	def showvalue(self,tickcode,window=45):
		df=self.loadnetv(tickcode)
		netv=df['unfixedvalue'].values.astype(float)
		sa=self.movingaverage(netv,window)
		ca=self.continousaverage(netv)
		dv=netv-sa
		sadv=self.movingaverage(dv,window)
		cdf=stats.norm.cdf((sadv-np.mean(sadv))/np.std(sadv))
		return tickcode,netv[-1],sa[-1],ca[-1],cdf[-1]
	def showtest(self,tickcode,window=45):
		df=self.loadnetv(tickcode)
		netv=df['unfixedvalue'].values.astype(float)
		sa=self.movingaverage(netv,window)
		ca=self.continousaverage(netv)
		dv=netv-sa
		sadv=self.movingaverage(dv,window)
		cdf=stats.norm.cdf((sadv-np.mean(sadv))/np.std(sadv))
		x_axis=np.linspace(1,netv.size,netv.size)
		sorted_cdf=np.sort(cdf)
		cdf2=stats.norm.cdf((cdf-np.mean(cdf))/np.std(cdf))
		sorted_cdf2=np.sort(cdf2)
		fig=plt.figure()
		p1=fig.add_subplot(2,1,1)
		p2=fig.add_subplot(2,1,2)
		p1.plot(x_axis,cdf)
		p1.plot(x_axis,sorted_cdf)
		p2.plot(x_axis,cdf2)
		p2.plot(x_axis,sorted_cdf2)
		p1.grid()
		p2.grid()
		fig.show()
	def predict(self,tickcode,predict=10):
		df=self.loadnetv(tickcode)
		netv=df['unfixedvalue'].values.astype(float)
		x_axis=np.linspace(1,netv.size,netv.size)
		pf=polyfit()
		pf.fitting(x_axis,netv)
		x_axis2=np.linspace(1,netv.size+predict,netv.size+predict)
		fitting=pf.calc(x_axis2)
		gradient=pf.gradient(x_axis2)
		fig=plt.figure()
		p1=fig.add_subplot(1,1,1)
		p1.plot(x_axis,netv)
		p1.plot(x_axis2,fitting)
		p1.plot(x_axis2,gradient)
		p1.grid()
		fig.show()
	def showall(self):
		__ind=1
		try:
			conn=self.PooL.connection()
			cursor=conn.cursor()
			cursor.execute("select fundcode,fundname from fund_meta order by cdfma,fundcode")
			__result=cursor.fetchone()
			while __result is not None:
				print("{} \t {} \t {}".format(__ind,__result[0],__result[1]))
				self.fundnames[__result[0]]=__result[1]
				__result=cursor.fetchone()
				__ind+=1
		except Exception as ex :
			print(ex)
		finally:
			cursor.close()
			conn.close()		
	def getitems(self,sorttype=1):
		items={}
		try:
			conn=self.PooL.connection()
			cursor=conn.cursor()
			if sorttype == 1 :
				sql="select fundcode,fundname,holding from fund_meta order by cdfma,fundcode"
			elif sorttype == 2 :
				sql="select fundcode,fundname,holding from fund_meta order by fundname,fundcode"
			elif sorttype == 3 :
				sql="select fundcode,fundname,holding from fund_meta order by fundcode"
			else:
				sql="select fundcode,fundname,holding from fund_meta order by cdfma,fundcode"
			cursor.execute(sql)
			__result=cursor.fetchone()
			while __result is not None:
				self.fundnames[__result[0]]=__result[1]
				items[__result[0]]=(__result[1],__result[2])
				__result=cursor.fetchone()
		except Exception as ex :
			print(ex)
		finally:
			cursor.close()
			conn.close()
		return items
	def async_refreshcdfma(self,window=45):
		future=self.threadPool.submit(self.refreshcdfma,window)
		future.add_done_callback(self.callback_refreshcdfma)
	def callback_refreshcdfma(self,future):
		pass
	def refreshcdfma(self,window=45):
		try:
			conn = self.PooL.connection()
			cursor = conn.cursor()
			items=self.getitems()
			__it=iter(items)
			count=0
			for item in __it :
				df=self.loadnetv(item)
				netv=df['unfixedvalue'].values.astype(float)
				ma=self.movingaverage(netv,window)
				dv=netv-ma
				madv=self.movingaverage(dv,window)
				cdf=stats.norm.cdf((madv-np.mean(madv))/np.std(madv))
				cmd="update fund_meta set cdfma={} where fundcode='{}'".format(cdf[-1],item)
				cursor.execute(cmd)
				count+=1
				if (count%10)==0 :
					conn.commit()
			conn.commit()
		except Exception as ex :
			print("exception in refreshcdfma")
			print(ex)
		finally:
			cursor.close()
			conn.close()

#计算delays阶以内的自相关系数，返回delays个值，分别计算序列均值，标准差
def autocorrelation(x,delays=1):
	n = len(x)
	x = np.array(x)
	result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]\
		/(x[i:].std()*x[:n-i].std()*(n-i)) for i in range(1,delays+1)]
	return result