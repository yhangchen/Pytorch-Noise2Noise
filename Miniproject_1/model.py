import os
from random import sample
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
try:
    from others.utils import *
    from others.unet import *
except:
    from .others.utils import *
    from .others.unet import *

logged = True
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    # when tensorboard is not installed, don't log.
    logged = False


class Model():

    def __init__(self) -> None:
        self.params = self._parse()
        torch.manual_seed(self.params.seed)
        self.use_cuda = torch.cuda.is_available() and self.params.cuda
        if self.params.noise == 'monte':
            self.params.lr = 0.0003  # paper Appendix A.2
            self.model = UNet(in_channels=9)
        else:
            self.model = UNet(in_channels=3)

        self.optim = Adam(self.model.parameters(),
                          lr=self.params.lr,
                          betas=[self.params.beta1, self.params.beta2],
                          eps=self.params.eps)
        self.scheduler = RampedLR(
            self.optim,
            max_epoch=self.params.epoch,
            ramp_down_percent=self.params.ramp_down_percent,
            verbose=True)

        if self.params.loss == 'hdr':
            assert self.params.noise == 'monte', 'Use HDR for Monte Carlo'
            self.loss = HDRLoss()
        elif self.params.loss == 'l2':
            self.loss = nn.MSELoss()
        elif self.params.loss == 'l1':
            self.loss = nn.L1Loss()
        else:
            raise NotImplementedError

        if self.use_cuda:
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()

        self.writer = SummaryWriter(comment=self.params.comment) if logged else None

    def load_pretrained_model(self) -> None:
        path2best = os.path.join(self.params.ckpt, 'bestmodel.pth')
        if self.use_cuda:
            self.model.load_state_dict(torch.load(path2best))
        else:
            self.model.load_state_dict(
                torch.load(path2best, map_location='cpu'))

    def train(self, train_input, train_target) -> None:
        noisy_imgs, clean_imgs = self.load_raw('val_data.pkl')
        val_loader = self.load_dataset(noisy_imgs, clean_imgs)
        best_psnr = -np.inf
        for epoch in range(self.params.epoch):
            print(f'Epoch: {epoch}')
            self.model.train()
            train_sampler = torch.utils.data.SubsetRandomSampler(
                np.random.choice(len(train_input),
                                 self.params.iteration_per_epoch *
                                 self.params.batch_size,
                                 replace=False))
            train_loader = self.load_dataset(train_input,
                                             train_target,
                                             sampler=train_sampler)
            trbar = tqdm(enumerate(train_loader),
                         total=len(train_loader),
                         unit='batch')
            train_loss = 0.0
            for batch_idx, (source, target) in trbar:
                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()
                denoise_source = self.model(source)
                loss = self.loss(denoise_source, target)
                train_loss += loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            self.writer.add_scalar('Loss/train', train_loss /
                                   (batch_idx + 1), epoch) if logged else None

            val_loss = 0
            val_psnr = 0
            self.model.eval()
            self.scheduler.step()
            valbar = tqdm(enumerate(val_loader),
                          total=len(val_loader),
                          unit='batch')
            for batch_idx, (source, target) in valbar:
                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()
                denoised_source = self.model(source)
                loss = self.loss(denoise_source, target)
                val_loss += loss.item()
                val_psnr_bth = 0
                for i in range(self.params.batch_size):
                    val_psnr_bth += psnr(denoised_source[i], target[i])
                val_psnr_bth /= self.params.batch_size
                val_psnr += val_psnr_bth
            self.writer.add_scalar('Loss/Val', val_loss /
                                   (batch_idx + 1), epoch) if logged else None
            self.writer.add_scalar('PSNR/Val', val_psnr /
                                   (batch_idx + 1), epoch) if logged else None
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                print('New best_psnr: ', best_psnr/(batch_idx + 1))
                if not os.path.isdir(self.params.save_dir):
                    os.mkdir(self.params.save_dir)
                torch.save(self.model.state_dict(), 'bestmodel.pth')

    def predict(self, test_input) -> torch.Tensor:
        self.model.eval()
        test_input = test_input.float().div(255.0)
        test_input = test_input.unsqueeze(0) if len(
            test_input.size()) == 3 else test_input
        if self.use_cuda:
            test_input = test_input.cuda()
        denoise_input = self.model(test_input)
        return denoise_input.detach().squeeze()

    def load_dataset(self, inputs, targets, sampler=None):
        dataset = torch.utils.data.TensorDataset(inputs.float().div(255.0),
                                                 targets.float().div(255.0))
        return DataLoader(dataset,
                          batch_size=self.params.batch_size,
                          shuffle=False, # sampler option is mutually exclusive with shuffle
                          sampler=sampler,
                          drop_last=True)

    def load_raw(self, name):
        return torch.load(os.path.join(self.params.data_dir, name))

    def _parse(self):
        from argparse import ArgumentParser
        parser = ArgumentParser(
            description='PyTorch implementation of Noise2Noise')
        parser.add_argument('--data-dir',
                            help='data directory',
                            default='./../data/')
        parser.add_argument('--ckpt', help='checkpoint directory', default='.')
        parser.add_argument('--save-dir', help='save best models', default='.')
        parser.add_argument('--cuda', help='use cuda', default=True, type=bool)
        parser.add_argument('--seed', help='random seed', default=0, type=int)

        parser.add_argument('--beta1',
                            help='Adam: beta1',
                            default=0.9,
                            type=float)
        parser.add_argument('--beta2',
                            help='Adam: beta2',
                            default=0.99,
                            type=float)
        parser.add_argument('--eps',
                            help='Adam: eps',
                            default=1e-8,
                            type=float)
        parser.add_argument('--lr',
                            help='learning rate',
                            default=0.001,
                            type=float)
        parser.add_argument('--loss',
                            help='loss function',
                            choices=['l1', 'l2', 'hdr'],
                            default='l2',
                            type=str)
        parser.add_argument('--batch-size',
                            help='minibatch size',
                            default=4,
                            type=int)
        parser.add_argument('--epoch',
                            help='number of epochs',
                            default=100,
                            type=int)
        parser.add_argument('--ramp-down-percent',
                            help='percent of ramp down epochs',
                            default=0.7,
                            type=float)
        parser.add_argument('--iteration-per-epoch',
                            help='iteration per epoch',
                            default=500,
                            type=int)
        parser.add_argument('--noise',
                            help='noise type',
                            choices=['gaussian', 'poisson', 'text', 'monte'],
                            default='gaussian',
                            type=str)
        parser.add_argument('--comment',
                            help='runs comment',
                            default='',
                            type=str)

        params = parser.parse_args()
        return params


if __name__ == '__main__':
    """Trains Noise2Noise."""
    model = Model()
    train_input, train_target = model.load_raw('train_data.pkl')
    model.train(train_input, train_target)
