from torch import empty, rand, repeat_interleave, zeros, exp, einsum, load, device, cuda
import math
from torch.nn.functional import unfold, fold
import os
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
import numpy as np
from tqdm import tqdm
from others.utils import psnr
import pickle
logged = True
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    # when tensorboard is not installed, don't log.
    logged = False

class Module(object):
    def __init__(self) -> None:
        pass
    def forward(self, *input):
        raise NotImplementedError
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []
    def to(self, device):
        return self
    def load_param(self, *param):
        return None
    
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
    
    def __call__(self, input):
        return self.forward(input)
    
class sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = None
    
    def forward(self, input):
        self.input = input
        out = 1.0 / (1 + exp(-input))
        return out
    
    def backward(self, gradwrtoutput):
        sig = 1.0 / (1 + exp(-self.input))
        return gradwrtoutput * (1 - sig) * sig
    
    def param(self):
        return [(None, None)]

    def __call__(self, input):
        return self.forward(input)
    
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
    
    def __call__(self, pred, gt):
        return self.forward(pred, gt)
    
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
            if len(layer.param()) > 1:
                ret.append(layer.param()[1])
        return ret
    
    def __call__(self, input):
        return self.forward(input)
    
    def to(self, device):
        for i, module in enumerate(self.modules):
            self.modules[i] = module.to(device)
        return self
    
    def load_param(self, param):
        model_idx = param_idx = 0
        while model_idx < len(self.modules) and (param_idx < len(param)):
            required_length = len(self.modules[model_idx].param())
            self.modules[model_idx].load_param(param[param_idx:required_length+param_idx])
            param_idx += required_length
            model_idx += 1
    
# Implemented for gradient testing
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
        ret = einsum('oi,ni->no',self.weight, self.input)
        if self.use_bias:
            return ret + self.bias
        return ret
    
    def backward(self, grdwrtoutput):
        dl_dx = einsum('oi,no->ni', self.weight, grdwrtoutput)
        self.dl_dw.add_(einsum('no,ni->oi', grdwrtoutput, self.input))
        self.dl_db.add_(grdwrtoutput.mean())
        return dl_dx
        
    def param(self):
        return [(self.weight, self.dl_dw), (self.bias, self.dl_db)]

    def __call__(self, input):
        return self.forward(input)
    
    def to(self, device):
        self.dl_dw = self.dl_dw.to(device)
        self.dl_db = self.dl_db.to(device)
        self.weight = self.weight.to(device)
        self.bias = self.bias.to(device)
        return self
    
    def load_param(self, param):
        self.weight, _ = param[0]
        self.bias, _ = param[1]
    
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
        n = in_channels * kernel_size**2
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
        wxb = einsum('ow,bws->bso', weight, self.unfolded)
        if self.use_bias:
            wxb += self.bias
        wxb = einsum('bso->bos', wxb)
        out_h = math.floor((x.shape[-2] - (self.kernel_size-1)*self.dilation + 2*self.padding - 1) / self.stride + 1)
        out_w = math.floor((x.shape[-1] - (self.kernel_size-1)*self.dilation + 2*self.padding - 1)  / self.stride + 1)
        
        #ret = fold(wxb, output_size=(out_h, out_w), kernel_size=(1,1))  
        ret = wxb.reshape(-1, self.out_channels, out_h, out_w)
        return ret
    
    def backward(self, grdwrtoutput):
        grdwrtoutput = grdwrtoutput.flatten(2, -1)
        weight = self.weight.reshape(self.out_channels, -1)
        # here x is cin*kernelwidth^2 and s is Hout*Wout
        dl_dx = einsum('ox,nos->nxs', weight, grdwrtoutput)
        out_size = (self.x.size(-2), self.x.size(-1))
        dl_dx = fold(dl_dx, output_size=out_size, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        
        self.dl_dw.add_(einsum('nos,nxs->ox', grdwrtoutput, self.unfolded).reshape(self.weight.size()))
        self.dl_db.add_(grdwrtoutput.transpose(0, 1).flatten(1, -1).mean(1))
        return dl_dx
    
    def param(self):
        return [(self.weight, self.dl_dw), (self.bias, self.dl_db)]

    def __call__(self, input):
        return self.forward(input)
    
    def to(self, device):
        self.dl_dw = self.dl_dw.to(device)
        self.dl_db = self.dl_db.to(device)
        self.weight = self.weight.to(device)
        self.bias = self.bias.to(device)
        return self

    def load_param(self, param):
        self.weight, _ = param[0]
        self.bias, _ = param[1]
        
class Upsample2d(Module):
    """
    Upsample of 2-d images, here we assume the input is [N, Cin, Hin, Win]
    """
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size=3, bias=True, dilation=1, stride=1, padding=0) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = Conv2d(in_channels, out_channels, kernel_size, bias, dilation, stride, padding)
        
    def nearest_neighbor_sampling(self, inp, scale_factor):
        """
        Input tensor shape:
        N, C, H, W
        Input scale_factor: int
        Output tensor shape:
        N, C, sf*H, sf*W
        """
        assert isinstance(scale_factor, int), 'Scale factor should be integer type!'

        inter = repeat_interleave(inp, repeats=scale_factor, dim=-1)
        ret = repeat_interleave(inter, repeats=scale_factor, dim=-2)
        return ret
    
    def forward(self, x):
        conv_in = self.nearest_neighbor_sampling(x, self.scale_factor)
        conv_out = self.conv.forward(conv_in)
        return conv_out
    
    def backward(self, grdwrtoutput):
        dl_dw = self.conv.backward(grdwrtoutput)
        dl_dx = zeros(dl_dw.size(0), dl_dw.size(1), dl_dw.size(2)//self.scale_factor, dl_dw.size(3)//self.scale_factor).to(dl_dw.device)
        for i in range(dl_dw.size(2)):
            for j in range(dl_dw.size(3)):
                dl_dx[:, :, i//self.scale_factor, j//self.scale_factor] += dl_dw[:, :, i, j]
        return dl_dx
    
    def param(self):
        return self.conv.param()
    
    def __call__(self, input):
        return self.forward(input)

    def to(self, device):
        self.conv = self.conv.to(device)
        return self   
    
    def load_param(self, param):
        self.conv.load_param(param)

class Model():
    
    def __init__(self, batch_size=4, lr=0.0001) -> None:
        
        self.batch_size = batch_size
        
        self.device = device('cuda' if cuda.is_available() else 'cpu')
        
        self.model = Sequential(
            Conv2d(3, 64, 3, stride=2),
            ReLU(), 
            Conv2d(64, 64, 3, stride=2, padding=0),
            ReLU(), 
            Upsample2d(18, 64, 64, stride=2, kernel_size=3, padding=3),
            ReLU(), 
            Upsample2d(1, 64, 3, stride=2, kernel_size=3, padding=0),
            sigmoid()
        ).to(self.device)
        # self.model = Sequential(
        #     Conv2d(3, 64, 2, stride=2),
        #     ReLU(), 
        #     Conv2d(64, 64, 2, stride=2, padding=1),
        #     ReLU(), 
        #     Upsample2d(2, 64, 64, stride=1, kernel_size=3, padding=0),
        #     ReLU(), 
        #     Upsample2d(2, 64, 3, stride=1, kernel_size=1, padding=0),
        #     sigmoid()
        # ).to(self.device)
        
        # self.model = Sequential(
        #     Conv2d(3, 32, 5, stride=2),
        #     ReLU(), 
        #     Conv2d(32, 32, 1, stride=2, padding=0),
        #     ReLU(), 
        #     Upsample2d(4, 32, 32, stride=2, kernel_size=5, padding=0),
        #     ReLU(), 
        #     Upsample2d(3, 32, 3, stride=1, kernel_size=5, padding=0),
        #     sigmoid()
        # ).to(self.device)
        # Good case with small stride
        # self.model = Sequential(
        #     Conv2d(3, 32, 3, stride=1),
        #     ReLU(), 
        #     Conv2d(32, 64, 3, stride=1),
        #     ReLU(), 
        #     Upsample2d(2, 64, 32, stride=2, padding=3),
        #     ReLU(), 
        #     Upsample2d(2, 32, 3, stride=2, padding=3),
        #     sigmoid()
        # ).to(self.device)
        ## Best model structure
        # self.model = Sequential(
        #     Conv2d(3, 32, 3, stride=1),
        #     ReLU(), 
        #     Conv2d(32, 64, 3, stride=1, padding=3),
        #     ReLU(), 
        #     Upsample2d(2, 64, 32, stride=2),
        #     ReLU(), 
        #     Upsample2d(2, 32, 3, stride=2),
        #     sigmoid()
        # ).to(self.device)
        
        # Bad case
        # self.model = Sequential(
        #     Conv2d(3, 32, 5, stride=2),
        #     ReLU(), 
        #     Conv2d(32, 64, 7, stride=2),
        #     ReLU(), 
        #     Upsample2d(3, 64, 32, stride=2, kernel_size=7, padding=3),
        #     ReLU(), 
        #     Upsample2d(5, 32, 3, stride=1, kernel_size=5, padding=3),
        #     sigmoid()
        # ).to(self.device)     
           
        # self.model = Sequential(
        #     Conv2d(3, 32, 3, stride=2),
        #     ReLU(), 
        #     Conv2d(32, 64, 3, stride=2),
        #     ReLU(), 
        #     Upsample2d(3, 64, 32, stride=1, kernel_size=3, padding=0),
        #     ReLU(), 
        #     Upsample2d(5, 32, 3, stride=3, kernel_size=3, padding=1),
        #     sigmoid()
        # ).to(self.device) 
        
        # self.model = Sequential(
        #     Conv2d(3, 32, 3, stride=2),
        #     ReLU(), 
        #     Conv2d(32, 64, 3, stride=2, dilation=2),
        #     ReLU(), 
        #     Upsample2d(3, 64, 32, stride=1, kernel_size=3, padding=0),
        #     ReLU(), 
        #     Upsample2d(2, 32, 3, stride=1, kernel_size=3, padding=1),
        #     sigmoid()
        # ).to(self.device) 
        # self.model = Sequential(
        #     Conv2d(3, 32, 3, stride=2),
        #     ReLU(), 
        #     Conv2d(32, 64, 3, stride=2, dilation=2),
        #     ReLU(), 
        #     Upsample2d(3, 64, 32, stride=1, kernel_size=3, padding=0),
        #     ReLU(), 
        #     Upsample2d(2, 32, 3, stride=1, kernel_size=1, padding=0),
        #     sigmoid()
        # ).to(self.device) 
        
        self.optimizer = SGD(self.model.param(), lr)
        self.criterion = MSELoss()

        self.writer = SummaryWriter(comment='lr_{}_bz_{}'.format(lr, self.batch_size)) if logged else None

    def load_pretrained_model(self, model_dir):
        params = pickle.load(open(model_dir, 'rb'))
        self.model.load_param(params)
        return True
        

    def save_model(self, dir):
        pickle.dump(self.model.param(),open(dir, 'wb'))
            
    
    def train(self, train_input, train_target, num_epoch=50, load_model=False):
        if load_model:
            self.load_pretrained_model('bestmodel.pth')
        
        noisy_imgs, clean_imgs = self.load_raw('val_data.pkl')
        val_loader = self.load_dataset(noisy_imgs, clean_imgs)
        best_psnr = -np.inf
        for epoch in range(num_epoch):
            print(f'Epoch: {epoch}')
            train_sampler = SubsetRandomSampler(
                np.random.choice(len(train_input),
                                 500 *
                                 self.batch_size,
                                 replace=False))
            
            train_loader = self.load_dataset(train_input,
                                             train_target,
                                             sampler=train_sampler)
            trbar = tqdm(enumerate(train_loader),
                         total=len(train_loader),
                         unit='batch')
            
            train_loss = 0.0
            
            for batch_idx, (source, target) in trbar:
                source, target = source.to(self.device), target.to(self.device)
                denoise_source = self.model(source)
                loss = self.criterion(denoise_source, target)
                train_loss += loss.item()
                dloss = self.criterion.backward()
                self.optimizer.zero_grad()
                self.model.backward(dloss)
                self.optimizer.step()
            print(train_loss)

            self.writer.add_scalar('Loss/train', train_loss /
                                   (batch_idx + 1), epoch) if logged else None

            val_loss = 0
            val_psnr = 0
            valbar = tqdm(enumerate(val_loader),
                          total=len(val_loader),
                          unit='batch')
            for batch_idx, (source, target) in valbar:
                source, target = source.to(self.device), target.to(self.device)
                denoised_source = self.model(source)
                
                if logged:
                    self.writer.add_images('Source/Val', source, epoch) 
                    self.writer.add_images('Target/Val', target, epoch)
                    self.writer.add_images('Denoised/Val', denoised_source, epoch) 
                
                loss = self.criterion(denoise_source, target)
                val_loss += loss.item()
                val_psnr_bth = 0
                for i in range(self.batch_size):
                    val_psnr_bth += psnr(denoised_source[i], target[i])
                val_psnr_bth /= self.batch_size
                val_psnr += val_psnr_bth
            
            # Mean PSNR of this batch
            val_psnr = val_psnr / (batch_idx + 1)            
            if logged:
                self.writer.add_scalar('Loss/Val', val_psnr, epoch)
                self.writer.add_scalar('PSNR/Val', val_psnr, epoch) 
                
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                print('New best_psnr: ', best_psnr)
                print('Saving model....')
                self.save_model('bestmodel.pth')
                
    def predict(self, test_input):
        denoise_source = self.model(test_input)
        return denoise_source.mul(255.0)

    def load_raw(self, name):
        return load(os.path.join('./../data/', name))

    def load_dataset(self, inputs, targets, sampler=None):
        dataset = TensorDataset(inputs.float().div(255.0),
                                                 targets.float().div(255.0))
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False, # sampler option is mutually exclusive with shuffle
                          sampler=sampler,
                          drop_last=True)

if __name__ == '__main__':
    for bz in [16]:
        for lr_test in [10e-5]:
            model = Model(lr=lr_test, batch_size=bz)
            train_input, train_target = model.load_raw('train_data.pkl')
            model.train(train_input, train_target, load_model=False, num_epoch=100)