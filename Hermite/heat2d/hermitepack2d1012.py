import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import partial as PARTIAL
import chebypack as ch 
from scipy.special import factorial
from numpy.polynomial import hermite as np_hermite
norm_hermite_factor_fun = lambda n: 1./(np.pi**(0.25)*np.sqrt(2.**n)*np.sqrt(factorial(n, exact=False)))
norm_hermite_exp_fun = lambda x: np.exp(-x**2/2)


class ZerosFilling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.zeros(1, device=x.device)

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
    

def norm_Hermite_func_Vand_expr(Nx, modes) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    
    factor_fun = lambda n: 1. / (np.pi ** (0.25) * np.sqrt(2. ** n) * np.sqrt(float(math.factorial(n))))
    exp_fun = lambda x: np.exp(-x ** 2 / 2)

    x_hg, w_hg = np_hermite.hermgauss(Nx)
    w_hg *= np.exp(x_hg ** 2)
    

    n_factor = np.array([factor_fun(n) for n in range(modes[0])])
    x_exp = exp_fun(x_hg)
    Vandmat = x_exp[:, np.newaxis] * np_hermite.hermvander(x_hg, modes[0]-1) * n_factor

    x_hg_tensor = torch.from_numpy(x_hg)
    w_hg_tensor = torch.from_numpy(w_hg)
    Vandmat_tensor = torch.from_numpy(Vandmat)

    return x_hg_tensor, w_hg_tensor, Vandmat_tensor


class Vandermode_transform2d:
    def __init__(self, Nx, modes, Vand_expr):
        self.apply_N = Nx
        _, weights, self.back_mat = Vand_expr(Nx, modes)
        self.forw_mat = torch.diag(weights).float() @ self.back_mat.float()
        self.back_mat = self.back_mat.float()
        
        self.fwd = PARTIAL(ch.Wrapper, [self._forward])
        self.inv = PARTIAL(ch.Wrapper, [self._backward])


    
    
    def _forward(self, u):  
        result = u @ self.forw_mat

        return result
    
    def _backward(self, u):
        
        
        return u @ self.back_mat.T

    def type(self, dtype):
        self.forw_mat, self.back_mat = self.forw_mat.type(dtype), self.back_mat.type(dtype)
    def to(self, device):
        
        self.forw_mat, self.back_mat = self.forw_mat.float().to(device), self.back_mat.float().to(device)
        return self
    def __call__(self, *args, **kwargs):
        return self.fwd(*args, **kwargs)







class PseudoSpectra2d(nn.Module):
    def __init__(self, dim,  in_channels, out_channels, modes, bandwidth=1, triL=0):
        super(PseudoSpectra2d, self).__init__()

        
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.bandwidth = bandwidth
        self.triL = triL
        self.X_dims = np.arange(-dim, 0)

        
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels*bandwidth.prod().item(), out_channels, modes.prod().item()))
        self.unfold = torch.nn.Unfold(kernel_size=bandwidth,
                                      padding=triL)
        
        self.X_slices = [slice(None), slice(None)] + [slice(freq) for freq in modes]
        self.pad_slices = [slice(None), slice(None)] + [slice(freq) for freq in modes+bandwidth-1-triL*2]

    def quasi_diag_mul(self, input, weights):
        xpad = self.unfold(input)
        return torch.einsum("bix, iox->box", xpad, weights)

    def forward(self, u,T):
        batch_size = u.shape[0]

        b = T(u, self.X_dims) 

        
        
        
        out = torch.zeros(batch_size, self.out_channels, *self.modes, device=u.device, dtype=u.dtype)
        out[self.X_slices] = self.quasi_diag_mul(b[self.pad_slices], self.weights).reshape(
                batch_size, self.out_channels, *self.modes)
        u = T.inv(out, self.X_dims)   
        return u









class SOL2d_Vandermonde(nn.Module):
    def __init__(self,  in_channels, modes, width, bandwidth, out_channels=1, dim=2, skip=True, triL = 0):
        super(SOL2d_Vandermonde, self).__init__()

        modes = np.array([modes]*dim) if isinstance(modes, int) else np.array(modes)
        bandwidth = np.array([bandwidth]*dim) if isinstance(bandwidth, int) else np.array(bandwidth)
        triL = np.array([triL]*dim) if isinstance(triL, int) else np.array(triL)

        self.modes = modes
        self.width = width
        self.triL = triL
        self.T = None
        self.dim = dim
        
        self.X_dims = np.arange(-dim, 0)
        
        self.conv0 = PseudoSpectra2d(dim,   width, width, modes, bandwidth, triL)
        self.conv1 = PseudoSpectra2d(dim,   width, width, modes, bandwidth, triL)
        self.conv2 = PseudoSpectra2d(dim,   width, width, modes, bandwidth, triL)
        self.conv3 = PseudoSpectra2d(dim,   width, width, modes, bandwidth, triL)
        
        


        self.lift = PseudoSpectra2d(dim,  in_channels, width-in_channels, modes, bandwidth, triL)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        
        
        



        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        self.fc3 = nn.Linear(width, out_channels)
        self.skip = nn.Identity() if skip else ZerosFilling()

    def forward(self, x):

        x = x.permute(0, -1, 1, 2)
        if self.T == None or self.T.apply_N != x.shape[-1]:
            del self.T
            self.T = Vandermode_transform2d(x.shape[-1], self.modes, norm_Hermite_func_Vand_expr)
            self.T = self.T.to(x.device)

        x = torch.cat([x, F.gelu(self.lift(x,self.T))], dim=1)

        
        
        
        
        
        x = self.skip(x) + F.gelu(self.w0(x) + self.conv0(x, self.T))

        x = self.skip(x) + F.gelu(self.w1(x) + self.conv1(x, self.T))

        x = self.skip(x) + F.gelu(self.w2(x) + self.conv2(x, self.T))

        x = self.skip(x) + F.gelu(self.w3(x) + self.conv3(x, self.T))

        
        

        

        

        

        x = x.permute(0, 2, 3, 1)
        
        
        x = self.fc3(x)

        

        return x
    
