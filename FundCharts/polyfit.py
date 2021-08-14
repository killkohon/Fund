import numpy as np
from numpy.linalg import solve

class ParameterException(Exception):
	def __init__(self,value):
		self.value=value
	def __str__(self):
		print("Parameter Exception！")
		return repr(self.value)

class polyfit:
	def __init__(self,rank=3):
		self.degree=rank
		self.params=np.zeros(self.degree+1,dtype=float)
	def calc(self,x):
		dat=x.copy()
		for e in np.nditer(dat,op_flags=['readwrite']):
			e[...]=self.params[0]+self.params[1]*e+self.params[2]*e**2+self.params[3]*e**3
		return dat
	def fitting(self,x,y):
		if len(x) != len(y):
			raise ParameterException("The shape of parameters should be equivalent.")
		m=len(x)
		A = np.ones(m).reshape((m,1))
		for i in range(self.degree):
			A = np.hstack([A,(x**(i+1)).reshape((m,1))])
		self.params = solve(np.dot(A.T,A),np.dot(A.T,y.T))
	def gradient(self,x):
		dat=x.copy()
		for e in np.nditer(dat,op_flags=['readwrite']):
			e[...]=self.params[1]+2*self.params[2]*e+3*self.params[3]*e**2
		return dat

