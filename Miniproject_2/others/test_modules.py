import sys,os
sys.path.append('../')
from model import Linear, Sequential, ReLU, SGD, MSELoss, Conv2d, Upsample2d
from torch import rand, ones
import math
from torch.nn.functional import unfold, fold
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = rand(3, 5)
    y = ones(3,1)*1
    
    # Linear model test
    seq_model = Sequential(
    Linear(5, 10),
    ReLU(),
    Linear(10, 1),   
    )
    
    optimizer = SGD(seq_model.param(), 0.05)
    criterion = MSELoss()    
    loss = math.inf
    
    loss_hist = []
    while loss >= 0.001:
        pred = seq_model(x)
        optimizer.zero_grad()
        loss = criterion(pred, y)
        dloss = criterion.backward()
        print(loss.item())
        loss_hist.append(loss.item())
        seq_model.backward(dloss)
        optimizer.step()
    plt.plot(loss_hist)
    plt.show()

    # Conv2d test
    model = Sequential(
    Conv2d(3, 5, kernel_size=3, padding=0, stride=1),
    ReLU()
    )

    x = rand(10, 3, 32, 32)
    y = ones(10, 5, 30, 30)*0.5

    optimizer = SGD(model.param(), 0.00001)
    criterion = MSELoss()

    loss = 9999
    loss_hist = []
    while loss >= 0.01:
        pred = model(x)
        optimizer.zero_grad()
        loss = criterion(pred, y)
        dloss = criterion.backward()
        print(loss.item())
        loss_hist.append(loss.item())
        model.backward(dloss)
        optimizer.step()
    plt.plot(loss_hist)
    plt.show()

    # Upsample test
    x = rand(1, 3, 2, 2)
    y = ones(1, 5, 6, 6)*1
    model = Upsample2d(4, 3, 5)
    optimizer = SGD(model.param(), 0.0001)
    criterion = MSELoss()

    loss = 9999
    loss_hist = []
    while loss >= 0.002:
        pred = model(x)
        optimizer.zero_grad()
        loss = criterion(pred, y)
        dloss = criterion.backward()
        print(loss.item())
        loss_hist.append(loss.item())
        model.backward(dloss)
        optimizer.step()
    plt.plot(loss_hist)
    plt.show()

