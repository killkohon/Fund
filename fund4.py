import datetime as dt
from bs4 import BeautifulSoup
import requests
import pandas as pd 
import numpy as np 
import pymysql
from DBUtils.PersistentDB import PersistentDB
import matplotlib.pyplot as plt
from scipy import stats
from numpy.linalg import solve
import urllib.parse 
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QListWidgetItem, QGridLayout, QListWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys

plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False

class ParameterException(Exception):
	def __init__(self,value):
		self.value=value
	def __str__(self):
		print("Parameter Exception！")
		return repr(self.value)

class polyfit:
	def __init__(self):
		self.degree=4
		self.params=np.zeros(self.degree+1,dtype=float)
	def calc(self,x):
		dat=x.copy()
		for e in np.nditer(dat,op_flags=['readwrite']):
			e[...]=self.params[0]+self.params[1]*e+self.params[2]*e**2+self.params[3]*e**3+self.params[4]*e**4
		return dat
	def fitting(self,x,y):
		if len(x) != len(y):
			raise ParameterException("The shape of parameters should be equivalent.")
		m=len(x)
		A = np.ones(m).reshape((m,1))
		for i in range(self.degree):
			A = np.hstack([A,(x**(i+1)).reshape((m,1))])
		self.params = solve(np.dot(A.T,A),np.dot(A.T,y.T))

class fundnetv:
	def __init__(self):
		self.PooL = PersistentDB(
		    creator = pymysql,  #使用链接数据库的模块
			maxusage = None, #一个链接最多被使用的次数，None表示无限制
			setsession = [], #开始会话前执行的命令
			ping = 0, #ping MySQL服务端,检查服务是否可用
 			closeable = False, #conn.close()实际上被忽略，供下次使用，直到线程关闭，自动关闭链接，而等于True时，conn.close()真的被关闭
			threadlocal = None, # 本线程独享值的对象，用于保存链接对象
			host = '192.168.136.31',
			port = 3306,
			user = 'fund',
			password = '123456',
			database = 'fund',
			charset = 'utf8'
			)
		self.isopen = lambda x: x.find("开放")>=0
		self.fundnames={}
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
			except:
				pass
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
		except:
			print("exception in savedata")
			pass
		finally:
			cursor.close()
			conn.close()
	def grabmetafromdb(self):
		try:
			conn= self.PooL.connection()
			cursor=conn.cursor()
			cursor.execute("select distinct(fundcode) from fund_values ")
			__result=cursor.fetchone()
			while __result is not None:
				print("fundcode:{}".format(__result[0]))
				code,meta=self.grabmeta(__result[0])
				for data in meta["Datas"]:
					if (data["CATEGORY"] == 700 and data["CODE"] == code) :
						self.savemeta(code,data["NAME"],data["CATEGORY"],data["CATEGORYDESC"],data["FundBaseInfo"]["JJGS"],data["FundBaseInfo"]["JJJL"])
				__result=cursor.fetchone()
		except:
			print("except at find the last vdate")
			pass
		else:
			pass
		finally:
			cursor.close()
			conn.close()
	def grabfromdb(self):
		try:
			conn= self.PooL.connection()
			cursor=conn.cursor()
			cursor.execute("select distinct(fundcode) from fund_values ")
			__result=cursor.fetchone()
			while __result is not None:
				print("fundcode:{}".format(__result[0]))
				self.grabnetv(__result[0])
				__result=cursor.fetchone()
		except:
			print("except at find the last vdate")
			pass
		else:
			pass
		finally:
			cursor.close()
			conn.close()
	def grabnetv(self,tickcode):
		nextdate='2020-01-02'
		try:
			conn= self.PooL.connection()
			cursor=conn.cursor()
			cursor.execute("select max(vdate) as lastdate from fund_values where fundcode='{}'".format(tickcode))
			__result=cursor.fetchone()
			if __result[0] is not None:
				nextdate=(__result[0]+dt.timedelta(1)).__format__('%Y-%m-%d')
		except:
			print("except at find the last vdate")
			pass
		else:
			pass
		finally:
			cursor.close()
			conn.close()
		GoAhead= True
		while GoAhead:
			print("Next date: {}".format(nextdate))
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
	def savedata(self,tickcode,valuedate,netv,unfixednetv,buyable=True,salable=True,dividend=""):
		try:
			conn = self.PooL.connection()
			cursor = conn.cursor()
			data=(tickcode,valuedate,netv,self.replacenan(unfixednetv,netv),buyable,salable,dividend)
			print("To save:{}".format(data))
			cursor.execute("insert into fund_values(fundcode,vdate,netvalue,unfixedvalue,buyable,salable,dividend)values('%s','%s',%f,%f,%d,%d,'%s')"%data)
			result = cursor.fetchall()
			conn.commit()
		except:
			print("exception in savedata")
			pass
		finally:
			cursor.close()
			conn.close()
	def loadnetv(self,tickcode,count=200):
		try:
			conn=self.PooL.connection()
			cursor=conn.cursor()
			cursor.execute("select * from fund_values where fundcode='{}' order by vdate limit {}".format(tickcode,count))
			df=pd.DataFrame(list(cursor.fetchall()),columns=[x[0] for x in cursor.description ])
			return df 
		except:
			pass
		finally:
			cursor.close()
			conn.close()
	def slideaverage(self,source,window=20):
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
		netv=df['unfixedvalue'].values.astype(float)
		sa=self.slideaverage(netv,window)
		ca=self.continousaverage(netv)
		dv=netv-sa
		sadv=self.slideaverage(dv,window)
		if figure == None :
			figure=plt.figure()
		figure.suptitle("{} {}".format(tickcode,self.fundname(tickcode)))
		x_axis=np.linspace(1,netv.size,netv.size)
		pf=polyfit()
		pf.fitting(x_axis,netv)
		fitting=pf.calc(x_axis)
		cdf=stats.norm.cdf((sadv-np.mean(sadv))/np.std(sadv))
		cdf2=stats.norm.cdf((dv-np.mean(dv))/np.std(dv))
		p1=figure.add_subplot(3,1,1)
		p2=figure.add_subplot(3,1,2)
		p3=figure.add_subplot(3,1,3)
		p1.plot(x_axis,netv,label='netv')
		p1.plot(x_axis,sa,label='sa')
		p1.plot(x_axis,ca,label='cont. average')
		p1.plot(x_axis,fitting,label='fit')
		p2.plot(x_axis,dv,label='dv(netv-sa)')
		p2.plot(x_axis,sadv,label='sadv')
		p3.plot(x_axis,(netv-np.min(netv))/(np.max(netv)-np.min(netv)),label='reg. netv')
		p3.plot(x_axis,cdf,label='CDF(sadv)')
		p3.plot(x_axis,cdf2,label='CDF(dv)')
		p1.grid()
		p2.grid()
		p3.grid()
		p1.legend()
		p2.legend()
		p3.legend()
	def showdata(self,tickcode,window=45):
		df=self.loadnetv(tickcode)
		netv=df['unfixedvalue'].values.astype(float)
		val=df['netvalue'].values.astype(float)
		sa=self.slideaverage(netv,window)
		ca=self.continousaverage(netv)
		dv=netv-sa
		sadv=self.slideaverage(dv,window)
		x_axis=np.linspace(1,netv.size,netv.size)
		pf=polyfit()
		pf.fitting(x_axis,netv)
		fitting=pf.calc(x_axis)
		cdf=stats.norm.cdf((sadv-np.mean(sadv))/np.std(sadv))
		cdf2=stats.norm.cdf((dv-np.mean(dv))/np.std(dv))
		print("{} \t {}".format(tickcode,self.fundname(tickcode)))
		print("VAL: {}".format(netv[-10:]))
		print(" FV: {}".format(val[-10:]))
		print(" SA: {}".format(sa[-10:]))
		print("FIT: {}".format(fitting[-10:]))
		print("CDF: {}".format(cdf[-10:]))
		print("CDF(sadv): {}".format(cdf2[-10:]))
	def showvalue(self,tickcode,window=45):
		df=self.loadnetv(tickcode)
		netv=df['unfixedvalue'].values.astype(float)
		sa=self.slideaverage(netv,window)
		ca=self.continousaverage(netv)
		dv=netv-sa
		sadv=self.slideaverage(dv,window)
		cdf=stats.norm.cdf((sadv-np.mean(sadv))/np.std(sadv))
		return tickcode,netv[-1],sa[-1],ca[-1],cdf[-1]
	def showtest(self,tickcode,window=45):
		df=self.loadnetv(tickcode)
		netv=df['unfixedvalue'].values.astype(float)
		sa=self.slideaverage(netv,window)
		ca=self.continousaverage(netv)
		dv=netv-sa
		sadv=self.slideaverage(dv,window)
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
		fig=plt.figure()
		p1=fig.add_subplot(1,1,1)
		p1.plot(x_axis,netv)
		p1.plot(x_axis2,fitting)
		p1.grid()
		fig.show()
	def showall(self):
		__ind=1
		try:
			conn=self.PooL.connection()
			cursor=conn.cursor()
			cursor.execute("select fundcode,fundname from fund_meta order by fundcode")
			__result=cursor.fetchone()
			while __result is not None:
				print("{} \t {} \t {}".format(__ind,__result[0],__result[1]))
				self.fundnames[__result[0]]=__result[1]
				__result=cursor.fetchone()
				__ind+=1
		except:
			pass
		finally:
			cursor.close()
			conn.close()
	def getitems(self):
		items={}
		try:
			conn=self.PooL.connection()
			cursor=conn.cursor()
			cursor.execute("select fundcode,fundname from fund_meta order by fundcode")
			__result=cursor.fetchone()
			while __result is not None:
				self.fundnames[__result[0]]=__result[1]
				items[__result[0]]=__result[1]
				__result=cursor.fetchone()
		except:
			pass
		finally:
			cursor.close()
			conn.close()
		return items
class FundChart(QWidget):
	def __init__(self):
		super(FundChart, self).__init__()
		self.fund=fundnetv()
		self.initUI()
	def initUI(self):
		self.gridlayout=QGridLayout()
		self.gridlayout.setSpacing(10)
		self.fundlist=QListWidget(self)
		self.fundlist.setMinimumWidth(240)
		self.fundlist.setMaximumWidth(240)
		items=self.fund.getitems()
		__it=iter(items)
		for item in __it :
			__item=QListWidgetItem("{} {}".format(item,items[item]))
			__item.data=item
			self.fundlist.addItem(__item)
		self.fundlist.itemClicked.connect(self.selectItem)
		self.gridlayout.addWidget(self.fundlist,1,0)
		self.figure=plt.figure()
		self.canvas=FigureCanvas(self.figure)
		self.gridlayout.addWidget(self.canvas,1,1)
		self.canvas.draw()
		self.setLayout(self.gridlayout)
	def selectItem(self,obj):
		self.figure.clf()
		self.fund.showfigure(obj.data,figure=self.figure)
		self.canvas.draw()


if __name__ == '__main__':
	app=QApplication([])
	ui=FundChart()
	ui.show()
	sys.exit(app.exec_())
