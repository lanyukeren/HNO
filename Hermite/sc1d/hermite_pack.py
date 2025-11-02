import torch
import numpy as np
import scipy.io
import torch.nn as nn
import torch.nn.functional as F
import math
import functools
from functools import partial as PARTIAL
import chebypack as ch

from numpy.polynomial import hermite as np_hermite


norm_hermite_factor_fun = lambda n: 1./(np.pi**(0.25)*np.sqrt(2.**n)*np.sqrt(float(math.factorial(n))))
norm_hermite_exp_fun = lambda x: np.exp(-x**2/2)


class ZerosFilling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.zeros(1, device = x.device)

def norm_Hermite_func_Vand_expr(Nx, modes) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    factor_fun = lambda n: 1. / (np.pi ** (0.25) * np.sqrt(2. ** n) * np.sqrt(float(math.factorial(n))))
    exp_fun = lambda x: np.exp(-x ** 2 / 2)

    x_hg, w_hg = np_hermite.hermgauss(Nx)
    w_hg *= np.exp(x_hg ** 2)
    n_factor = np.array([factor_fun(n) for n in range(modes)])
    x_exp = exp_fun(x_hg)
    return torch.from_numpy(x_hg), torch.torch.from_numpy(w_hg),\
           torch.from_numpy(x_exp[:, np.newaxis] * np_hermite.hermvander(x_hg, modes-1) * n_factor)

class Vandermode_transform:
    def __init__(self, Nx, modes, Vand_expr):
        self.apply_N = Nx
        _, weights, self.back_mat = Vand_expr(Nx, modes)
        self.forw_mat = torch.diag(weights) @ self.back_mat
        self.fwd = PARTIAL(ch.Wrapper, [self._forward])
        self.inv = PARTIAL(ch.Wrapper, [self._backward])

    def _forward(self, u):  
        return u @ self.forw_mat
    def _backward(self, u):
        return u @ self.back_mat.T

    def type(self, dtype):
        self.forw_mat, self.back_mat = self.forw_mat.type(dtype), self.back_mat.type(dtype)
    def to(self, device):
        self.forw_mat, self.back_mat = self.forw_mat.to(device), self.back_mat.to(device)
        return self
    def __call__(self, *args, **kwargs):
        return self.fwd(*args, **kwargs)

class PseudoSpectra1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, bandwidth=1, triL=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.bandwidth = bandwidth
        self.triL = triL

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(modes, in_channels, out_channels, bandwidth))

    def quasi_diag_mul(self, x, weights):
        xpad = x.unfold(-1, self.bandwidth, 1)
        return torch.einsum("bixw, xiow->box", xpad, weights)

    def forward(self, u, T):
        batch_size, _, Nx= u.shape

        b = T(u, -1)

        
        
        
        out = self.quasi_diag_mul(b[..., :self.modes+self.bandwidth-1], self.weights)

        u = T.inv(out, -1)
        return u

class SOL1d_Vandermonde(nn.Module):
    def __init__(self, in_channels, modes, width, bandwidth, out_channels=1, dim=1, skip=True, triL=0):
        super().__init__()

        self.modes = modes
        self.width = width
        self.triL = triL
        self.T = None
        self.dim = dim

        self.conv0 = PseudoSpectra1d(width, width, modes, bandwidth, triL)
        self.conv1 = PseudoSpectra1d(width, width, modes, bandwidth, triL)
        self.conv2 = PseudoSpectra1d(width, width, modes, bandwidth, triL)
        self.conv3 = PseudoSpectra1d(width, width, modes, bandwidth, triL)

        self.lift = PseudoSpectra1d(in_channels, width-in_channels, modes, bandwidth)

        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        self.skip = nn.Identity() if skip else ZerosFilling()

    def forward(self, x):

        
        x = x.permute(0, -1, 1)

        if self.T == None or self.T.apply_N != x.shape[-1]:
            del self.T
            self.T = Vandermode_transform(x.shape[-1], self.modes, norm_Hermite_func_Vand_expr)
            self.T = self.T.to(x.device)
            
        x = torch.cat([x, F.gelu(self.lift(x, self.T))], dim=1)

        x = self.skip(x) + F.gelu(self.w0(x) + self.conv0(x, self.T))

        x = self.skip(x) + F.gelu(self.w1(x) + self.conv1(x, self.T))

        x = self.skip(x) + F.gelu(self.w2(x) + self.conv2(x, self.T))

        x = self.skip(x) + F.gelu(self.w3(x) + self.conv3(x, self.T))

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        

        return x


if __name__ == '__main__':
    
    Nx = 129
    modes = 10

    norm_hermite_factor_fun = lambda n: 1. / (np.pi ** (0.25) * np.sqrt(2. ** n) * np.sqrt(float(math.factorial(n))))
    norm_hermite_exp_fun = lambda x: np.exp(-x ** 2 / 2)

    x_hg, w_hg = np_hermite.hermgauss(Nx)
    w_hg *= np.exp(x_hg ** 2)
    n_factor = np.array([norm_hermite_factor_fun(n) for n in range(modes)])
    x_exp = norm_hermite_exp_fun(x_hg)
    Vandmat = np.exp(-x_hg ** 2 / 2)[:, np.newaxis] * np_hermite.hermvander(x_hg, modes - 1) * n_factor

    coef = np.array([1, 0, 0])
    coef = coef * n_factor[:coef.shape[0]]
    u = x_exp * np_hermite.Hermite(coef)(x_hg)
    print(w_hg * u @ Vandmat)
    print(u @ np.diag(w_hg) @ Vandmat)

    
    Nx = 9
    modes = 1
    x_hg, w_hg = np_hermite.hermgauss(Nx)
    coef = np.array([1])
    coef = coef
    u = np_hermite.Hermite(coef)(x_hg)
    Vandmat = np_hermite.hermvander(x_hg, modes - 1) / np.sqrt(np.pi)  

    print(w_hg * u @ Vandmat)


    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    