import numpy as np

class neuralnetwork:
	def __init__(self,layers,rate=0.2,activation='sigmoid'):
		self.activation=getattr(self,activation)
		self.activation_derivative=getattr(self,activation+"_derivative")
		self.layers=layers
		self.rate=rate
		self.weights=[]
		for i in range(0,len(layers)-1):
			self.weights.append(np.mat(np.ones([layers[i],layers[i+1]])))

	def sigmoid(self,x):
		return 1/(1+np.exp(-x))
	def sigmoid_derivative(self,x):
		return sigmoid(x)*(1-sigmoid(x))

	def fit(self,inp,exp):
		print("input:"+str(inp)+" shape:"+str(inp.shape)+" type:"+str(type(inp)))
		print("expect:"+str(exp)+" shape:"+str(exp.shape)+" type:"+str(type(exp)))
		if inp.shape[1] != self.layers[0]:
			print("输入层格式错误.")
			return
		if exp.shape[1] != self.layers[len(self.layers)-1]:
			print("结果格式错误")
			return
		if inp.shape[0] != exp.shape[0]:
			print("输入层数量应和结果数量一致")
			return	

		#前向
		for i in range(0,inp.shape[0]):
			ohid=[]
			ohid.append(np.mat(inp[i]))
			print("ohid="+str(ohid)+ " type:"+str(type(ohid)))
			#dinp=inp[i]
			#print("dinp="+str(dinp))
			#dactivate=dinp
			#前向传播
			for j in range(0,len(self.weights)-1):
				dhid=ohid[j]*self.weights[j]
				print("dhid["+str(j)+"]="+str(dhid))
				ohid.append(self.activation(dhid))
				print("ohid="+str(ohid))
			dout=ohid[-1]*self.weights[len(self.weights)-1]
			ohid.append(dout)
			#print("type(dout)="+str(type(dout)))
			#print("dout[0]="+str(dout[0])+" shape:"+str(dout[0].shape))
			print("dout="+str(dout)+" type:"+str(type(dout)))


			#后向
			#-e(d)
			ned=-np.mat(np.asarray(exp[i]-dout))
			print("ned="+str(ned)+" type:"+str(type(ned)))

			print("Etotal="+str(ned@ned.T))
			
			
			#从d*d单位阵开始
			Operators=[]
			de=ned*np.diagflat(np.ones([1,self.layers[-1]])).T  # 实际就是-ed*I,输出层梯度
			print("de="+str(de))
			print("ohid="+str(ohid))
			Operators.insert(0,ohid[-2].T*de)
			print("Operators="+str(Operators))
			#以下计算隐含层梯度
			for i in range(len(ohid)-2,0,-1):
				print("==========="+str(i)+"==============")
				print("ohid["+str(i)+"]="+str(ohid[i]))
				ta=np.diagflat(np.multiply(ohid[i],np.ones(ohid[i].shape)-ohid[i]))
				print("ta="+str(ta))
				print("self.weights["+str(i)+"]="+str(self.weights[i]))
				print("de="+str(de))
				de=de*(ta*self.weights[i]).T
				print("de="+str(de))
				Operators.insert(0,ohid[i-1].T*de)
				print("Opertors="+str(Operators))
			#修正权重
			for j in range(0,len(self.weights)):
				self.weights[j]=self.weights[j]+self.rate*Operators[j]
			print("weights="+str(self.weights))
	def predict(self,x):
		x=np.array(x)
		ohid=[]
		ohid.append(np.mat(x))
		for j in range(0,len(self.weights)-1):
			dhid=ohid[j]*self.weights[j]
			print("dhid["+str(j)+"]="+str(dhid))
			ohid.append(self.activation(dhid))
			print("ohid="+str(ohid))
		dout=ohid[-1]*self.weights[len(self.weights)-1]
		print("result="+str(dout))




