from model import ReLU, sigmoid, SGD, MSELoss, Sequential, Conv2d, Upsample2d
from torch import rand, load, device, cuda
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

class Model():
    
    def __init__(self) -> None:
        
        self.batch_size = 4
        
        self.device = device('cuda' if cuda.is_available() else 'cpu')
        
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
        self.model = Sequential(
            Conv2d(3, 32, 3, stride=2),
            ReLU(), 
            Conv2d(32, 64, 3, stride=2, dilation=2),
            ReLU(), 
            Upsample2d(3, 64, 32, stride=1, kernel_size=3, padding=0),
            ReLU(), 
            Upsample2d(2, 32, 3, stride=1, kernel_size=1, padding=0),
            sigmoid()
        ).to(self.device) 
        
        self.optimizer = SGD(self.model.param(), 0.0001)
        self.criterion = MSELoss()

        self.writer = SummaryWriter() if logged else None

    def load_pretrained_model(self, model_dir):
        params = pickle.load(open(model_dir, 'rb'))
        self.model.load_param(params)
        return True
        

    def save_model(self, dir):
        pickle.dump(self.model.param(),open(dir, 'wb'))
            
    
    def train(self, train_input, train_target, num_epoch=100, load_model=False):
        if load_model:
            self.load_pretrained_model('model.pickle')
        
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
            # self.model.eval()
            # self.scheduler.step()
            valbar = tqdm(enumerate(val_loader),
                          total=len(val_loader),
                          unit='batch')
            for batch_idx, (source, target) in valbar:
                source, target = source.to(self.device), target.to(self.device)
                denoised_source = self.model(source)
                self.writer.add_images('Source/Val', source, epoch) if logged else None
                self.writer.add_images('Target/Val', target, epoch) if logged else None
                self.writer.add_images('Denoised/Val', denoised_source, epoch) if logged else None
                loss = self.criterion(denoise_source, target)
                val_loss += loss.item()
                val_psnr_bth = 0
                for i in range(self.batch_size):
                    val_psnr_bth += psnr(denoised_source[i], target[i])
                val_psnr_bth /= self.batch_size
                val_psnr += val_psnr_bth
            self.writer.add_scalar('Loss/Val', val_loss /
                                   (batch_idx + 1), epoch) if logged else None
            self.writer.add_scalar('PSNR/Val', val_psnr /
                                   (batch_idx + 1), epoch) if logged else None
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                print('New best_psnr: ', best_psnr/(batch_idx + 1))
                print('Saving model....')
                self.save_model('model.pickle')
                # if not os.path.isdir(self.params.save_dir):
                #     os.mkdir(self.params.save_dir)
                # torch.save(self.model.state_dict(), 'bestmodel.pth')

    def predict(self, test_input):
        pass

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
    model = Model()
    train_input, train_target = model.load_raw('train_data.pkl')
    model.train(train_input, train_target, load_model=False)
