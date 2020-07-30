import datetime as dt
from bs4 import BeautifulSoup
import requests
import pandas as pd 
import numpy as np 
import pymysql
from DBUtils.PersistentDB import PersistentDB

class fundnetv:
	def __init__(self):
		self.PooL = PersistentDB(
		    creator = pymysql,  #使用链接数据库的模块
			maxusage = None, #一个链接最多被使用的次数，None表示无限制
			setsession = [], #开始会话前执行的命令
			ping = 0, #ping MySQL服务端,检查服务是否可用
 			closeable = False, #conn.close()实际上被忽略，供下次使用，直到线程关闭，自动关闭链接，而等于True时，conn.close()真的被关闭
			threadlocal = None, # 本线程独享值的对象，用于保存链接对象
			host = '192.168.136.9',
			port = 3306,
			user = 'fund',
			password = '123456',
			database = 'fund',
			charset = 'utf8'
			)
	def fund(self,tickCode, startDate, endDate):
		"""
		- Imput 
			- tickCode, 代码
			- startDate, 查询起始日
			- endDate, 查询截止日
		- Output, df with col:
			1. 净值日期
			2. 单位净值
			3. 累计净值
			4. 日增长率
			5. 申购状态
			6. 赎回状态
			7. 分红送配
		"""
		days = (dt.datetime.strptime(endDate, "%Y-%m-%d") - dt.datetime.strptime(startDate, "%Y-%m-%d")).days
		url = "http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code={}&sdate={}&edate={}&per={}".format(tickCode,startDate, endDate,days)
		response = requests.get(url)
		soup = BeautifulSoup(response.content, "lxml")
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
					table_records.append(val[0])
			table_rows.append(table_records)
		# 写入DataFrame
		table_rows = np.array(table_rows)
		df = pd.DataFrame()
		for col,col_name in enumerate(table_heads):
			df[col_name] = table_rows[:,col]
		return df
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
	def grabnetv(self,tickcode):
		nextdate='2020-01-02'
		try:
			conn= self.PooL.connection()
			cursor=conn.cursor()
			cursor.execute("select valuedate from netvalues where code='{}' and valuedate=max(valuedate) ".format(tickcode))
			__result=cursor.fetchone()
			if __result is not None:
				nextdate=(__result[0]+dt.datedelta(1)).__format__('%Y-%m-%d')
		except:
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
					self.savedata(tickcode,data[row][0],float(data[row][1]),float(data[row][2]))
					row-=1
				nextdate=(dt.datetime.strptime(data[0][0],"%Y-%m-%d")+dt.timedelta(1)).__format__("%Y-%m-%d")
			else:
				__tempdate=dt.datetime.strptime(nextdate,"%Y-%m-%d")
				if __tempdate.__lt__(dt.datetime.now()-dt.timedelta(7)):
					nextdate=(__tempdate+dt.timedelta(7)).__format__("%Y-%m-%d")
				else:
					GoAhead=False
	def savedata(self,tickcdoe,valuedate,netv,unfixednetv):
		try:
			conn = self.PooL.connection()
			cursor = conn.cursor()
			data=(tickcdoe,valuedate,netv,unfixednetv)
			print("To save:{}".format(data))
			cursor.execute("insert into netvalues(code,valuedate,netv,unfixednetv)values('%s','%s',%f,%f)"%data)
			result = cursor.fetchall()
			conn.commit()
		except:
			print("exception in savedata")
			pass
		finally:
			cursor.close()
			conn.close()


