import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        self._block_1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1), nn.Conv2d(48, 48, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1), nn.MaxPool2d(2))

        self._block_2_1 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1), nn.MaxPool2d(2))

        self._block_2_2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1), nn.MaxPool2d(2))

        self._block_2_3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1), nn.MaxPool2d(2))

        self._block_2_4 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1), nn.MaxPool2d(2))

        self._block_3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))
            # nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))

        self._block_4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))
            # nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))

        self._block_5_1 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))
            # nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))

        self._block_5_2 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))
            # nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))

        self._block_5_3 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))
            # nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))

        self._block_6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1))

        self.init_weights()

    def forward(self, x):
        pool_1 = self._block_1(x)
        pool_2 = self._block_2_1(pool_1)
        pool_3 = self._block_2_2(pool_2)
        pool_4 = self._block_2_3(pool_3)
        pool_5 = self._block_2_4(pool_4)

        upsample_5 = self._block_3(pool_5)
        concat_5 = torch.cat((upsample_5, pool_4), dim=1)
        upsample_4 = self._block_4(concat_5)
        concat_4 = torch.cat((upsample_4, pool_3), dim=1)
        upsample_3 = self._block_5_1(concat_4)
        concat_3 = torch.cat((upsample_3, pool_2), dim=1)
        upsample_2 = self._block_5_2(concat_3)
        concat_2 = torch.cat((upsample_2, pool_1), dim=1)
        upsample_1 = self._block_5_3(concat_2)
        concat_1 = torch.cat((upsample_1, x), dim=1)

        return self._block_6(concat_1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
