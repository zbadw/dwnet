# 导入Tensor类和numpy库
from dwnet.tensor import Tensor
import numpy as np

# 定义Loss类
class Loss:
    def __init__(self) -> None:
        pass

    # 前向传播函数
    def forward(self,inputs:Tensor,targets:Tensor) -> Tensor:
        pass

    # 反向传播函数
    def backward(self,inputs:Tensor,targets:Tensor) -> Tensor:
        pass

    # 定义__call__函数，使得类的实例可以像函数一样被调用
    def __call__(self,inputs:Tensor,targets:Tensor) -> Tensor:
        return self.forward(inputs,targets)

# 定义MSE类，继承自Loss类
class MSE(Loss):
    def __init__(self) -> None:
        super().__init__()

    # 前向传播函数
    def forward(self,inputs:Tensor,targets:Tensor) -> Tensor:
        self.inputs = inputs
        self.targets = targets
        # 返回inputs和targets之间的均方误差
        return np.sum((inputs-targets)**2)

    # 计算梯度
    def grad(self,inputs:Tensor,targets:Tensor) -> Tensor:
        # 返回inputs和targets之间的梯度
        return 2*(inputs-targets)
