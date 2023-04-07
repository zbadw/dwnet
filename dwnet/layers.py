from dwnet.tensor import Tensor
import numpy as np
# abstract
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self) -> None:
        self.params = {}
        self.grads = {}
    
    @abstractmethod
    def forward(self,inputs):
        pass
    
    @abstractmethod
    def backward(self,grad):
        pass
    def __call__(self,inputs):
        return self.forward(inputs)
    


class Linear(Layer):
    def __init__(self,input_size,output_size) -> None:
        super().__init__()
        self.params['w'] = np.random.randn(input_size,output_size)
        self.params['b'] = np.random.randn(output_size)
    
    def forward(self,inputs) -> Tensor:
        self.inputs = inputs
        return inputs @ self.params['w'] + self.params['b']
    
    def backward(self,grad) -> Tensor:
        self.grads['w'] = self.inputs.T @ grad
        self.grads['b'] = np.sum(grad,axis=0)
        return grad @ self.params['w'].T

class Activation(Layer):
    def __init__(self,activation,derivative) -> None:
        super().__init__()
        self.activation = activation
        self.derivative = derivative
    
    def forward(self,inputs) -> Tensor:
        self.inputs = inputs
        return self.activation(inputs)
    
    def backward(self,grad) -> Tensor:
        return grad * self.derivative(self.inputs)

class Sigmoid(Activation):
    def __init__(self) -> None:
        sigmoid = lambda x: 1/(1+np.exp(-x))
        super().__init__(sigmoid,lambda x: sigmoid(x)*(1-sigmoid(x)))

class Tanh(Activation):
    def __init__(self) -> None:
        tanh = lambda x: np.tanh(x)
        super().__init__(tanh,lambda x: 1-tanh(x)**2)
        