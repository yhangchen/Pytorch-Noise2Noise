from torch import empty, rand
import math
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import unfold, fold

class Module(object):
    def __init__(self) -> None:
        pass
    def forward(self, *input):
        raise NotImplementedError
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []
    
class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        
    def forward(self, input):
        self.input = input
        out = input.clamp(min=0)
        return out
    
    def backward(self, gradwrtoutput):
        input = self.input
        drelu_din = input.sign().clamp(min=0)
        return drelu_din * gradwrtoutput
    
    def param(self):
        return [(None, None)]
    
    
class sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = None
    
    def forward(self, input):
        self.input = input
        out = 1.0 / (1 + torch.exp(-input))
        return out
    
    def backward(self, gradwrtoutput):
        sig = 1.0 / (1 + torch.exp(-self.input))
        return gradwrtoutput * (1 - sig) * sig
    
    def param(self):
        return [(None, None)]
    
class SGD():
    """Stochastic Gradient Descent
    
    functions to implement:
    1. zero_grad():
        set the gradients in all the modules to 0
    2. step():
        update the param in all modules by w += grad * lr
    Usage:
    optim = SGD(model.parameters(), lr=0.1, *)
    
    """
    def __init__(self, params, lr) -> None:
        self.params = params
        self.lr = lr
        
    def zero_grad(self):
        for modules in self.params:
            weight, grad = modules
            if grad != None: grad.zero_()
            
    def step(self):
        for modules in self.params:
            weight, grad = modules
            if weight != None and (grad != None):
                weight.add_(-self.lr * grad)
        

class MSELoss(Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.pred = None
        self.gt = None
        
    def forward(self, pred, gt):
        self.pred = pred
        self.gt = gt
        
        loss = (pred - gt).pow(2).mean()
        return loss

    def backward(self):
        dloss = 2 * (self.pred - self.gt)/self.pred.size(0)
        return dloss
    
    def param(self):
        return [(None, None)]
    
    
class Sequential(Module):
    def __init__(self, *layers) -> None:
        super().__init__()
        self.modules = []
        for layer in layers:
            self.modules.append(layer)
            
    def forward(self, x):
        ret = x
        for layer in self.modules:
            ret = layer.forward(ret)
        return ret
    
    def backward(self, gradwrtoutput):
        grad_from_back = gradwrtoutput
        for layer in reversed(self.modules):
            grad_from_back = layer.backward(grad_from_back)
            
    def param(self):
        ret = []
        for layer in self.modules:
            ret.append(layer.param()[0])
        return ret
            
            
class Upsample(Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        pass
    
    def backward(self):
        return []
    
    
class Linear(Module):
    def __init__(self, in_dim, out_dim, bias=True) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.use_bias = bias
        self.weight = rand(out_dim, in_dim) * 2 / math.sqrt(in_dim) - 1 / math.sqrt(in_dim)
        self.bias = rand(1) * 2 / math.sqrt(in_dim) - 1 / math.sqrt(in_dim)
        self.dl_dw = empty(self.weight.size())
        self.dl_db = empty(self.bias.size())
        self.input = 0
        
    def forward(self, input):
        self.input = input
        ret = torch.einsum('oi,ni->no',self.weight, self.input)
        if self.use_bias:
            return ret + self.bias
        return ret
    
    def backward(self, grdwrtoutput):
        dl_dx = torch.einsum('oi,no->ni', self.weight, grdwrtoutput)
        self.dl_dw.add_(torch.einsum('no,ni->oi', grdwrtoutput, self.input))
        self.dl_db.add_(grdwrtoutput.mean())
        return dl_dx
        
    def param(self):
        return [(self.weight, self.dl_dw), (self.bias, self.dl_db)]
    
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True, dilation=1, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        
        # torch initialization
        n = self.in_channels
        n *= kernel_size**2
        stdv = 1. / math.sqrt(n)
        
        self.weight = empty(out_channels, in_channels, kernel_size, kernel_size).uniform_(-stdv, stdv)
        self.bias  = empty(out_channels).uniform_(-stdv, stdv)
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.unfolded = None
        self.x = None
        
        # gradient
        self.dl_dw = empty(self.weight.size()) 
        self.dl_db = empty(self.bias.size())
    
    def forward(self, x):
        self.x = x
        self.unfolded = unfold(x, kernel_size = self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        weight = self.weight.reshape(self.out_channels, -1)
        wxb = torch.einsum('ow,bws->bso', weight, self.unfolded)
        if self.use_bias:
            wxb += self.bias
        wxb = torch.einsum('bso->bos', wxb)
        out_h = math.floor((x.shape[-2] - (self.kernel_size-1)*self.dilation + 2*self.padding - 1) / self.stride + 1)
        out_w = math.floor((x.shape[-1] - (self.kernel_size-1)*self.dilation + 2*self.padding - 1)  / self.stride + 1)
        
        #ret = fold(wxb, output_size=(out_h, out_w), kernel_size=(1,1))  
        ret = wxb.reshape(-1, self.out_channels, out_h, out_w)
        return ret
    
    def backward(self, grdwrtoutput):
        grdwrtoutput = grdwrtoutput.flatten(2, -1)
        weight = self.weight.reshape(self.out_channels, -1)
        # here x is cin*kernelwidth^2 and s is Hout*Wout
        dl_dx = torch.einsum('ox,nos->nxs', weight, grdwrtoutput)
        out_size = (self.x.size(-2), self.x.size(-1))
        dl_dx = fold(dl_dx, output_size=out_size, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        
        self.dl_dw.add_(torch.einsum('nos,nxs->ox', grdwrtoutput, self.unfolded).reshape(self.weight.size()))
        self.dl_db.add_(grdwrtoutput.transpose(0, 1).flatten(1, -1).mean(1))
        return dl_dx
    
    def param(self):
        return [(self.weight, self.dl_dw), (self.bias, self.dl_db)]


if __name__ == '__main__':
    x = rand(3, 5)
    y = torch.ones(3,1)*1
    
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
        pred = seq_model.forward(x)
        optimizer.zero_grad()
        loss = criterion.forward(pred, y)
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
    ReLU(),
    )

    x = rand(10, 3, 32, 32)
    y = torch.ones(10, 5, 30, 30)*1

    optimizer = SGD(model.param(), 0.0005)
    criterion = MSELoss()

    loss = 9999
    loss_hist = []
    while loss >= 0.0001:
        pred = model.forward(x)
        optimizer.zero_grad()
        loss = criterion.forward(pred, y)
        dloss = criterion.backward()
        print(loss.item())
        loss_hist.append(loss.item())
        model.backward(dloss)
        optimizer.step()
    plt.plot(loss_hist)
    plt.show()