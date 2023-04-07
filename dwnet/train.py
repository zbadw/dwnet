from dwnet.tensor import Tensor  # 引入Tensor类
from dwnet.nn import Network  # 引入Network类
from dwnet.loss import Loss, MSE  # 引入Loss类和MSE类
from dwnet.optim import Optimizer, SGD  # 引入Optimizer类和SGD类
from dwnet.data import DataIterator, BatchIterator  # 引入DataIterator类和BatchIterator类


def train(net: Network,  # 定义train函数，接收Network类的实例net、输入数据inputs、目标数据targets、训练轮数num_epochs、数据迭代器iterator、损失函数loss、优化器optimizer
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:
    for epoch in range(num_epochs):  # 进行num_epochs轮训练
        epoch_loss = 0.0  # 初始化epoch_loss为0
        for batch in iterator(inputs, targets):  # 遍历数据迭代器
            predicted = net(batch.inputs)  # 计算网络的输出
            epoch_loss += loss(predicted, batch.targets)  # 计算损失
            grad = loss.grad(predicted, batch.targets)  # 计算损失函数关于网络输出的梯度
            net.backward(grad)  # 反向传播
            optimizer.step(net)  # 更新网络参数
        print(epoch, epoch_loss)  # 输出当前轮数和损失

