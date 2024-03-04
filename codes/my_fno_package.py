# Author: Zhongjian Wang
# This package is developed from Zongyi Li's FNO package
import torch
import math
import numpy as np
import scipy.io
import h5py
import torch.nn as nn
import torch.nn.functional as F
from labml_helpers.module import Module
import warnings 



class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class SpectralConv2d_flip(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_flip, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize, C, Nx, Ny = x.shape
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x = torch.cat([x, -x.flip(dims=[-2])[..., 1:Nx - 1,:]], dim=-2)
        x = torch.cat([x, -x.flip(dims=[-1])[..., 1:Ny - 1]], dim=-1)

        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        x = x[:,:,:Nx,:Ny]
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class MLP_t_theta(nn.Module):
    # updated since jan 9 24, 
    def __init__(self, in_channels, out_channels, mid_channels,t_channels,theta_channels):
        super(MLP_t_theta, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, out_channels, 1)
        self.time_emb1 = nn.Linear(t_channels, out_channels)
        self.time_emb2 = nn.Linear(out_channels, out_channels)
        self.time_act = Swish()
        self.theta_emb1 = nn.Linear(theta_channels, out_channels)
        self.theta_emb2 = nn.Linear(out_channels, out_channels)
        self.theta_act = F.gelu

    def forward(self, x, theta, t):
        x = self.mlp1(x) 
        x = x + self.theta_emb2(self.theta_act(self.theta_emb1(theta)))[:, :, None, None] #
        x = x + self.time_emb2(self.time_act(self.time_emb1(t)))[:, :, None, None]
        return x


class FNOBlock_t_theta(nn.Module): # FNO Block with t, theta embedding
    def  __init__(self, modes1, modes2,  width,flip=False):
        super(FNOBlock_t_theta, self).__init__()
        if flip:
            self.conv = SpectralConv2d_flip(width, width, modes1, modes2)
        else:
            self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.mlp0=MLP(width, width, width)
        self.mlp = MLP_t_theta(width, width, width,width*4,width*4) 
        self.w = nn.Conv2d(width, width, 1)
        self.act = F.gelu
    def forward(self,x, theta, t):
        x1 = self.conv(x)
        #New FNO Code: x1 = self.mlp0(self.conv(x))
        # Emb T 1
        x2 = self.mlp(x, theta, t)
        return self.act(x1 + x2)



##### Time Embedding copied from UNET:
class Swish(Module):
    """
    ### Swish actiavation function

    $$x * sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding_transformer(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, time_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = time_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)

        emb = t.reshape(-1,1) * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb
        

##################################################
# Theta/Time Embedding
class ParamsEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int, activation_func):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        # First linear layer
        self.lin1 = nn.Linear(in_channels, mid_channels)
        # Activation
        self.act = activation_func
        # Second linear layer
        self.lin2 = nn.Linear(mid_channels, out_channels)

    def forward(self, theta: torch.Tensor):
        # Transform with the MLP
        emb = self.act(self.lin1(theta))
        emb = self.lin2(emb)

        return emb


## Time embedded FNO
class FNO2d_emdT(nn.Module):
    def __init__(self, modes1, modes2,  width , theta_channels = 1,temb_model='MLP',flip=False):
        super(FNO2d_emdT, self).__init__()

        """
        Modified by Zhongjian
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.flip = flip
        if not self.flip:
            self.padding = 8 # pad the domain if input is non-periodic
        x_channels = 3
        
        # Time Embedding: supported- transformer, MLP (default)
        if temb_model == 'transformer':
            self.time_emb = TimeEmbedding_transformer(self.width*4)
        elif temb_model == 'MLP':
            self.time_emb = ParamsEmbedding(1,self.width*4,self.width*4,activation_func=Swish())
        else:
            raise Exception('Unknown Time Lifting Model')
        
        self.theta_emb = ParamsEmbedding(theta_channels,self.width*4,self.width*4,activation_func=F.gelu)
        self.FNOBlock = FNOBlock_t_theta
        
        self.p = nn.Conv2d(x_channels, self.width,1) 

        self.FNOBlock0 = self.FNOBlock(self.modes1, self.modes2, self.width,flip)
        self.FNOBlock1 = self.FNOBlock(self.modes1, self.modes2, self.width,flip)
        self.FNOBlock2 = self.FNOBlock(self.modes1, self.modes2, self.width,flip)
        self.FNOBlock3 = self.FNOBlock(self.modes1, self.modes2, self.width,flip)

        self.q = nn.Conv2d(self.width, 1, 1) # output channel is 1: u(x, y)

    def forward(self, x, theta, t):

        theta = self.theta_emb(theta)
        t = self.time_emb(t)

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        x = self.p(x)
        if not self.flip:
            x = F.pad(x, [0,self.padding, 0,self.padding])

        x = self.FNOBlock0(x, theta, t)

        x = self.FNOBlock1(x, theta, t)

        x = self.FNOBlock2(x, theta, t)

        x = self.FNOBlock3(x, theta, t)

        if not self.flip:
            x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)
     



# functions in original utilities3.py

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path,'r')
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T in 1D
        # x could be in shape of ntrain*w*l or ntrain*T*w*l or ntrain*w*l*T in 2D
        self.mean = torch.mean(x, dim=0)
        self.std = torch.std(x, dim=0)
        self.eps = eps
        self.time_last = time_last # if the time dimension is the last dim

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        # sample_idx is the spatial sampling mask
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if self.mean.ndim > sample_idx.ndim and not self.time_last:
                    std = self.std[...,sample_idx] + self.eps # T*batch*n
                    mean = self.mean[...,sample_idx]
        # x is in shape of batch*(spatial discretization size) or T*batch*(spatial discretization size)
        x = (x * std) + mean
        return x

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = torch.from_numpy(self.mean).to(device)
            self.std = torch.from_numpy(self.std).to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x



