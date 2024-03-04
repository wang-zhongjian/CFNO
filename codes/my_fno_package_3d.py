# This package is developed from Zongyi Li's package
# 3D version
import torch
import math
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from my_fno_package import *
import warnings 



################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

## Time/Theta embedded FNO
class FNO3d_emdT(nn.Module):
    def __init__(self, modes1, modes2,  modes3, width , theta_channels = 1,temb_model='MLP'):
        super(FNO3d_emdT, self).__init__()

        """
        Modified by Zhongjian
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 3 # pad the domain if input is non-periodic
        x_channels = 4
        self.theta_width = max(self.width,theta_channels)*4
        # Time Embedding: supported- transformer, MLP (default)
        if temb_model == 'transformer':
            self.time_emb = TimeEmbedding_transformer(self.width*4)
        elif temb_model == 'MLP':
            self.time_emb = ParamsEmbedding(1,self.width*4,self.width*4,activation_func=Swish())
        else:
            raise Exception('Unknown Time Lifting Model')
        
        self.theta_emb = ParamsEmbedding(theta_channels,self.theta_width,self.theta_width,activation_func=F.gelu)
        self.FNOBlock = FNOBlock_t_theta3d
        
        self.p = nn.Conv3d(x_channels, self.width,1) 

        self.FNOBlock0 = self.FNOBlock(self.modes1, self.modes2, self.modes3, self.width,self.theta_width)
        self.FNOBlock1 = self.FNOBlock(self.modes1, self.modes2, self.modes3, self.width,self.theta_width)
        self.FNOBlock2 = self.FNOBlock(self.modes1, self.modes2, self.modes3, self.width,self.theta_width)
        self.FNOBlock3 = self.FNOBlock(self.modes1, self.modes2, self.modes3, self.width,self.theta_width)

        self.q = nn.Conv3d(self.width, 1, 1) # output channel is 1: u(x, y)

    def forward(self, x, theta, t):

        theta = self.theta_emb(theta)
        t = self.time_emb(t)

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        x = self.p(x)
        x = F.pad(x, [0,self.padding, 0,self.padding, 0,self.padding])

        x = self.FNOBlock0(x, theta, t)

        x = self.FNOBlock1(x, theta, t)

        x = self.FNOBlock2(x, theta, t)

        x = self.FNOBlock3(x, theta, t)
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding, :-self.padding]
        x = self.q(x)
        return x
    



    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[2], shape[3], shape[4]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1, 1).repeat([batchsize, 1, 1, size_y, size_z])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y, 1).repeat([batchsize, 1, size_x, 1, size_z])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, 1, size_z).repeat([batchsize, 1, size_x, size_y, 1])
        return torch.cat((gridx, gridy, gridz), dim=1).to(device)

class MLP_t_theta3d(nn.Module):
    # updated since jan 9 24, 3d version
    # include w effect
    def __init__(self, in_channels, out_channels, mid_channels,t_channels,theta_channels):
        super(MLP_t_theta3d, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, out_channels, 1)
        self.time_emb1 = nn.Linear(t_channels, out_channels)
        self.time_emb2 = nn.Linear(out_channels, out_channels)
        self.time_act = Swish()
        self.theta_emb1 = nn.Linear(theta_channels, out_channels)
        self.theta_emb2 = nn.Linear(out_channels, out_channels)
        self.theta_act = F.gelu

    def forward(self, x, theta, t):
        x = self.mlp1(x) 
        x = x + self.theta_emb2(self.theta_act(self.theta_emb1(theta)))[:, :, None, None, None] #
        x = x + self.time_emb2(self.time_act(self.time_emb1(t)))[:, :, None, None, None]
        return x


class FNOBlock_t_theta3d(nn.Module): # FNO Block with t, theta embedding
    def  __init__(self, modes1, modes2, modes3,  width,theta_width):
        super(FNOBlock_t_theta3d, self).__init__()
        self.conv = SpectralConv3d(width, width, modes1, modes2,modes3)
        self.mlp = MLP_t_theta3d(width, width, width,width*4,theta_width) 
        self.act = F.gelu
    def forward(self,x, theta, t):
        x1 = self.conv(x)
        #New FNO Code: x1 = self.mlp0(self.conv(x))
        # Emb T 1
        x2 = self.mlp(x, theta, t)
        return self.act(x1 + x2)




     



