from dwnet.tensor import Tensor
import numpy as np
# abstract
from abc import ABC, abstractmethod

class Layer(ABC):
    """神经网络层的抽象类"""
    def __init__(self) -> None:
        self.params = {}
        self.grads = {}
    
    @abstractmethod
    def forward(self,inputs)->Tensor:
        """前向传播"""
        pass
    
    @abstractmethod
    def backward(self,grad):
        """反向传播"""
        pass
    
    def __call__(self,inputs):
        return self.forward(inputs)


    


class Linear(Layer):
    def __init__(self,input_size,output_size) -> None:
        super().__init__()
        # 初始化权重和偏置
        self.params['w'] = np.random.randn(input_size,output_size)  # 初始化权重
        self.params['b'] = np.random.randn(output_size)  # 初始化偏置
    
    def forward(self,inputs) -> Tensor:
        self.inputs = inputs
        # 前向传播
        return inputs @ self.params['w'] + self.params['b']
    
    def backward(self,grad) -> Tensor:
        # 反向传播
        self.grads['w'] = self.inputs.T @ grad  # 计算权重的梯度
        self.grads['b'] = np.sum(grad,axis=0)  # 计算偏置的梯度
        return grad @ self.params['w'].T 





class Activation(Layer):
    def __init__(self,activation,derivative) -> None:
        super().__init__()
        self.activation = activation  # 激活函数
        self.derivative = derivative  # 激活函数的导数
    
    def forward(self,inputs) -> Tensor:
        self.inputs = inputs  # 保存输入
        return self.activation(inputs)  # 返回激活函数的输出
    
    def backward(self,grad) -> Tensor:
        return grad * self.derivative(self.inputs)  # 返回梯度乘以激活函数的导数



class Sigmoid(Activation):
    def __init__(self) -> None:
        # 定义sigmoid激活函数
        sigmoid = lambda x: 1/(1+np.exp(-x))
        # 调用父类构造函数
        super().__init__(sigmoid,lambda x: sigmoid(x)*(1-sigmoid(x)))



class Tanh(Activation):
    def __init__(self) -> None:
        # 定义tanh激活函数
        tanh = lambda x: np.tanh(x)
        # 调用父类构造函数
        super().__init__(tanh,lambda x: 1-tanh(x)**2)

