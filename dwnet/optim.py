class Optimizer:
    def step(self):
        # 抛出未实现异常
        raise NotImplementedError

    def zero_grad(self):
        # 抛出未实现异常
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self,lr=0.01) -> None:
        # 初始化学习率
        self.lr = lr

    def step(self, net) -> None:
        # 遍历网络的参数和梯度
        for param, grad in net.params_and_grads():
            # 更新参数
            param -= self.lr * grad

    def zero_grad(self, net):
        # 遍历网络的参数和梯度
        for param, grad in net.params_and_grads():
            # 梯度清零
            param.grad = 0