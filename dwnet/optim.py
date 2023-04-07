
class Optimizer:
    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self,lr=0.01) -> None:
        self.lr = lr

    def step(self, net) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad

    def zero_grad(self, net):
        for param, grad in net.params_and_grads():
            param.grad = 0