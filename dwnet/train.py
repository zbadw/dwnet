from dwnet.tensor import Tensor
from dwnet.nn import Network
from dwnet.loss import Loss, MSE
from dwnet.optim import Optimizer, SGD
from dwnet.data import DataIterator, BatchIterator


def train(net: Network,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net(batch.inputs)
            epoch_loss += loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)