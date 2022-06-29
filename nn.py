import numpy as np
def tanh(x):
  return np.tanh(x)
 

def tanh_deriv(x):
  return 1.0-np.tanh(x)*np.tanh(x)
 

def logistic(x):
  return 1/(1+np.exp(-x))
 

def logistic_derivative(x):
  return logistic(x)*(1-logistic(x))
 
 
class NeuralNetwork:
  def __init__(self,layers,activation='tanh'):
  #根据类实例化一个函数，_init_代表的是构造函数
  #self相当于java中的this
    """
    :param layers:一个列表，包含了每层神经网络中有几个神经元，至少有两层，输入层不算作
        [, , ,]中每个值代表了每层的神经元个数
    :param activation：激活函数可以使用tanh 和 logistics，不指明的情况下就是tanh函数
    """
    if activation =='logistic':
      self.activation = logistic
      self.activation_deriv = logistic_derivative
    elif activation =='tanh':
      self.activation =tanh
      self.activation_deriv=tanh_deriv
    #初始化weights,  
    self.weights =[]
    #len(layers)layer是一个list[10,10,3]，则len(layer)=3
    #除了输出层都要赋予一个随机产生的权重
    for i in range(1,len(layers)-1):
      #np.random.random为nunpy随机产生的数
      #实际是以第二层开始，前后都连线赋予权重，权重位于[-0.25,0.25]之间
      self.weights.append((2*np.random.random((layers[i-1]+1,layers[i]+1))-1)*0.25)
      self.weights.append((2*np.random.random((layers[i]+1,layers[i+1]))-1)*0.25)
  #定义一个方法，训练神经网络
  def fit(self,X,y,learning_rate=0.2,epochs=10000):
    #X：数据集,确认是二维，每行是一个实例，每个实例有一些特征值
    X=np.atleast_2d(X)
    #np.ones初始化一个矩阵，传入两个参数全是1
    #X.shape返回的是一个list[行数，列数]
    #X.shape[0]返回的是行，X.shape[1]+1：比X多1，对bias进行赋值为1
    temp = np.ones([X.shape[0],X.shape[1]+1])
    #“ ：”取所有的行
    #“0：-1”从第一列到倒数第二列，-1代表的是最后一列
    temp[:,0:-1]=X
    X=temp
    #y：classlabel，函数的分类标记
    y=np.array(y)
    #K代表的是第几轮循环 
    for k in range(epochs):
      #从0到X.shape[0]随机抽取一行做实例
      i =np.random.randint(X.shape[0])
      a=[X[i]]
      
      #正向更新权重  ，len(self.weights)等于神经网络层数
      for l in range(len(self.weights)):
        #np.dot代表两参数的内积，x.dot(y) 等价于 np.dot(x,y)
        #即a与weights内积，之后放入非线性转化function求下一层
        #a输入层，append不断增长，完成所有正向的更新
        a.append(self.activation(np.dot(a[l],self.weights[l])))
      #计算错误率，y[i]真实标记   ，a[-1]预测的classlable   
      error=y[i]-a[-1]
      #计算输出层的误差，根据最后一层当前神经元的值，反向更新
      deltas =[error*self.activation_deriv(a[-1])]
      
      #反向更新
      #len(a)所有神经元的层数，不能算第一场和最后一层
      #从最后一层到第0层，每次-1
      for l in range(len(a)-2,0,-1):
        #
        deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
      #reverse将deltas的层数跌倒过来
      deltas.reverse()
      for i in range(len(self.weights)):
        #
        layer = np.atleast_2d(a[i])
        #delta代表的是权重更新
        delta = np.atleast_2d(deltas[i])
        #layer.T.dot(delta)误差和单元格的内积
        self.weights[i]+=learning_rate*layer.T.dot(delta)
    
  def predict(self,x):
    x=np.array(x)
    temp=np.ones(x.shape[0]+1)
    #从0行到倒数第一行
    temp[0:-1]=x 
    a=temp
    for l in range(0,len(self.weights)):
      a=self.activation(np.dot(a,self.weights[l]))
    return a

if __name__== '__main__':
  nn=NeuralNetwork([2,3,4,1])
  X=np.array([[0,0],[0,1],[1,0],[1,1]])
  y=np.array([0,1,1,0])
  nn.fit(X,y)
  nn.predict([1,0])
  