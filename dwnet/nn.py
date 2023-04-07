from typing import Sequence, Tuple
from dwnet.layers import Layer
from dwnet.tensor import Tensor

class Network:
    def __init__(self,layers) -> None:
        self.layers = layers  # 初始化神经网络的层

    def forward(self,inputs:Tensor) -> Tensor:
        for layer in self.layers:  # 对于每一层
            inputs = layer(inputs)  # 进行前向传播
        return inputs  # 返回输出结果

    def backward(self,grad:Tensor) -> Tensor:
        for layer in reversed(self.layers):  # 对于每一层
            grad = layer.backward(grad)  # 进行反向传播
        return grad  # 返回梯度

    def __call__(self,inputs:Tensor) -> Tensor:
        return self.forward(inputs)  # 重载__call__方法，使得可以直接调用forward方法

    def params_and_grads(self) -> Sequence[Tuple[Tensor,Tensor]]:
        for layer in self.layers:  # 对于每一层
            for name,param in layer.params.items():  # 对于每一个参数
                grad = layer.grads[name]  # 获取对应的梯度
                yield param,grad  # 返回参数和梯度的元组