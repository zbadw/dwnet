from dwnet.tensor import Tensor
import numpy as np

class Loss:
    def __init__(self) -> None:
        pass

    def forward(self,inputs:Tensor,targets:Tensor) -> Tensor:
        pass

    def backward(self,inputs:Tensor,targets:Tensor) -> Tensor:
        pass

    def __call__(self,inputs:Tensor,targets:Tensor) -> Tensor:
        return self.forward(inputs,targets)

class MSE(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,inputs:Tensor,targets:Tensor) -> Tensor:
        self.inputs = inputs
        self.targets = targets
        return np.sum((inputs-targets)**2)

    def grad(self,inputs:Tensor,targets:Tensor) -> Tensor:
        return 2*(inputs-targets)