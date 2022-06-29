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
		print("input:"+str(inp)+" shape:"+str(inp.shape))
		print("expect:"+str(exp)+" shape:"+str(exp.shape))
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
			print("type(dout)="+str(type(dout)))
			print("dout[0]="+str(dout[0])+" shape:"+str(dout[0].shape))
			derror=0
			#e(k)
			diff=np.mat(np.asarray(dout[0]-exp[i])[0])
			print("diff="+str(diff)+" type:"+str(type(diff)))
			print("diff[0]="+str(diff[0]))
			for k in range(0,len(diff)):
				derror=derror+np.asarray(diff)[k]**2/2
			print("derror="+str(derror))
			print("weights[-1]="+str(self.weights[-1]))
			dd=self.rate*ohid[-1].T*diff
			print("dd="+str(dd))
		print("ohid="+str(ohid))
		#后向
		for i in range(0,len(ohid)):
			print(ohid[i])
			Imat=np.ones(ohid[i].shape)
			print(Imat)
			print("weights["+str(i)+"]:")
			print(self.weights[i])
			P1=
			Param=np.multiply(ohid[i],Imat-ohid[i])
			print(Param)





