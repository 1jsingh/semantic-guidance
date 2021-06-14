import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, SGD
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
import torch.nn.utils.weight_norm as weightNorm
import utils.util as util
from torch.nn.parallel import DistributedDataParallel as DDP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# dim = 128
# LAMBDA = 10  # Gradient penalty lambda hyperparameter


class TReLU(nn.Module):
    def __init__(self):
        super(TReLU, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0)

    def forward(self, x):
        x = F.relu(x - self.alpha) + self.alpha
        return x


class Discriminator(nn.Module):
    def __init__(self, high_res=False):
        super(Discriminator, self).__init__()

        self.conv0 = weightNorm(nn.Conv2d(6, 16, 5, 2, 2))
        self.conv1 = weightNorm(nn.Conv2d(16, 32, 5, 2, 2))
        self.conv2 = weightNorm(nn.Conv2d(32, 64, 5, 2, 2))
        self.conv3 = weightNorm(nn.Conv2d(64, 128, 5, 2, 2))
        self.conv4 = weightNorm(nn.Conv2d(128, 1, 5, 2, 2))
        self.relu0 = TReLU()
        self.relu1 = TReLU()
        self.relu2 = TReLU()
        self.relu3 = TReLU()
        self.high_res = high_res

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        if self.high_res is True:
            x = F.avg_pool2d(x, 8)
        else:
            x = F.avg_pool2d(x, 4)
        x = x.view(-1, 1)
        return x


class WGAN:
    def __init__(self, gpu=0, distributed=False, dim=128, high_res=False):
        self.gpu = gpu
        self.distributed = distributed
        self.high_res = high_res

        if self.high_res is True:
            self.dim = 256
        else:
            self.dim = 128

        self.netD = Discriminator(high_res=self.high_res)
        self.target_netD = Discriminator(high_res=self.high_res)

        self.netD = self.netD.cuda(gpu)
        self.target_netD = self.target_netD.cuda(gpu)

        for param in self.target_netD.parameters():
            param.requires_grad = False

        if distributed:
            self.netD = DDP(self.netD, device_ids=[self.gpu])
            # self.target_netD = DDP(self.target_netD, device_ids=[self.gpu])

        util.hard_update(self.target_netD, self.netD)

        self.optimizerD = Adam(self.netD.parameters(), lr=3e-4, betas=(0.5, 0.999))
        # self.dim = dim
        self.LAMBDA = 10  # Gradient penalty lambda hyperparameter

    def cal_gradient_penalty(self, real_data, fake_data, batch_size):
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
        alpha = alpha.view(batch_size, 6, self.dim, self.dim)
        alpha = alpha.cuda(self.gpu)
        fake_data = fake_data.view(batch_size, 6, self.dim, self.dim)
        interpolates = Variable(alpha * real_data.data + ((1 - alpha) * fake_data.data), requires_grad=True)
        disc_interpolates = self.netD(interpolates)
        gradients = autograd.grad(disc_interpolates, interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(self.gpu),
                                  create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def cal_reward(self, fake_data, real_data):
        return self.target_netD(torch.cat([real_data, fake_data], 1))

    def save_gan(self, path, num_episodes):
        # self.netD.cpu()
        torch.save(self.netD.state_dict(), '{}/wgan_{:05}.pkl'.format(path, num_episodes))
        # self.netD.cuda(self.gpu)

    def load_gan(self, path, map_location, num_episodes):
        self.netD.load_state_dict(torch.load('{}/wgan_{:05}.pkl'.format(path, num_episodes), map_location=map_location))

    def random_masks(self):
        """
        Generate random masks for complement discriminator (need to make mask overlap with the stroke)
        :return: mask
        """
        # initialize mask
        mask = np.ones((3, self.dim, self.dim))

        # generate one of 4 random masks
        choose = 1  # np.random.randint(0, 1)
        if choose == 0:
            mask[:, :self.dim // 2] = 0
        elif choose == 1:
            mask[:, :, :self.dim // 2] = 0
        elif choose == 2:
            mask[:, :, self.dim // 2:] = 0
        elif choose == 3:
            mask[:, self.dim // 2:] = 0

        return mask

    def update(self, fake_data, real_data):
        fake_data = fake_data.detach()
        real_data = real_data.detach()

        # standard conditional training for discriminator
        fake = torch.cat([real_data, fake_data], 1)
        real = torch.cat([real_data, real_data], 1)

        # # complement discriminator conditional training for discriminator
        # mask = torch.tensor(random_masks()).float().to(device)
        # fake = torch.cat([(1 - mask) * real_data, mask * fake_data], 1)
        # real = torch.cat([(1 - mask) * real_data, mask * real_data], 1)

        # compute discriminator scores for real and fake data
        D_real = self.netD(real)
        D_fake = self.netD(fake)

        gradient_penalty = self.cal_gradient_penalty(real, fake, real.shape[0])
        self.optimizerD.zero_grad()
        D_cost = D_fake.mean() - D_real.mean() + gradient_penalty
        D_cost.backward()
        self.optimizerD.step()
        util.soft_update(self.target_netD, self.netD, 0.001)
        return D_fake.mean(), D_real.mean(), gradient_penalty
