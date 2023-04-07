from typing import Sequence, Tuple
from dwnet.layers import Layer
from dwnet.tensor import Tensor

class Network:
    def __init__(self,layers) -> None:
        self.layers = layers

    def forward(self,inputs:Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def backward(self,grad:Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def __call__(self,inputs:Tensor) -> Tensor:
        return self.forward(inputs)

    def params_and_grads(self) -> Sequence[Tuple[Tensor,Tensor]]:
        for layer in self.layers:
            for name,param in layer.params.items():
                grad = layer.grads[name]
                yield param,grad