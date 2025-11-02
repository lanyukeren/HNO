import torch
import numpy as np
import scipy.io
import torch.nn as nn
import torch.nn.functional as F
import math
import functools

def dct(u):
    Nx = u.shape[-1]

    # transform x -> theta, a discrete cosine transform of "cheap" version
    V = torch.cat([u, u.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)
    a = torch.fft.ifft(V, dim=-1)[..., :Nx].real
    a[..., 1:Nx-1] *= 2
    return a

def idct(a):
    Nx = a.shape[-1]

    v = a.clone()
    v[..., (0, Nx-1)] *= 2
    V = torch.cat([v, v.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)
    u = torch.fft.fft(V, dim=-1)[..., :Nx].real / 2
    return u

def cmp(a):
    Nx = a.shape[-1]

    sgn = torch.zeros_like(a)
    sgn[..., ::2] = 1.0
    b = torch.fft.irfft(torch.fft.rfft(sgn, n=2*Nx, dim=-1)
                        * torch.fft.rfft(a, n=2*Nx, dim=-1), dim=-1)[..., :Nx]
    # b[..., -4:-2] = -a[..., -2:]
    # print(b[-2:])
    # b[..., -2:] = 0
    return b

# def cross_phi(hat_f):
#     # <f, \phi_k>_w = \pi/2 * ( c_k \hat f_k - c_{k+2} \hat f_{k+2})
#     Nx = hat_f.shape[-1]
#     c_k = torch.ones((Nx,), dtype=torch.float64); c_k[0] += 1
#     print(c_k.shape, hat_f.shape)
#     c_k_mul_f = c_k * hat_f
#     return torch.pi/2 * (c_k_mul_f[:-2] - c_k_mul_f[2:])

def galerkin(a):
    # c_k = [2, 1, 1, 1....]
    Nx = a.shape[-1]

    b = torch.zeros_like(a)
    b[..., :-2] = torch.pi/2 * (a[..., :-2] - a[..., 2:])
    b[..., 0] += torch.pi/2 * a[..., 0] # c0=2
    return b


def cmp_decrease(a, res_return=False):
    Nx = a.shape[-1]

    sgn = torch.zeros(*a.shape[:-1], 2*Nx, dtype=a.dtype, device=a.device)
    sgn[..., -(Nx-1)//2*2::2] = -1.0

    b = torch.fft.irfft(torch.fft.rfft(sgn, n=2*Nx, dim=-1)
                        * torch.fft.rfft(a, n=2*Nx, dim=-1), dim=-1)[..., :Nx]
    b[..., -2:] = 0
    # res = b[..., :2] - a[..., :2]
    # b[..., :2] = a[..., :2]

    return (b,res) if res_return else b

# def cmp_neumann(a):
#     Nx = a.shape[-1]
#     fac = torch.linspace(0, Nx-1, Nx, dtype=a.dtype, device=a.device) ** 2
#
#     sgn = torch.zeros_like(a)
#     sgn[..., ::2] = 1
#
#     b = torch.fft.irfft(torch.fft.rfft(sgn, n=2*Nx, dim=-1)
#                         * torch.fft.rfft(a*fac, n=2*Nx, dim=-1), dim=-1)[..., :Nx]
#
#     b[..., :2] = a[..., :2]
#     b[..., 2:-2] /= fac[2:-2]
#     # b[..., -2:] = 0
#     # b[..., -4:-2] = -a[..., -2:] / torch.tensor(\
#     #     [(Nx-4.0)/(Nx-2.0),(Nx-3.0)/(Nx-1.0)], dtype=torch.float64, device=a.device)**2
#
#     return b

def cmp_neumann(a):
    Nx = a.shape[-1]
    fac = torch.linspace(0, Nx-1, Nx, dtype=a.dtype, device=a.device) ** 2

    sgn = torch.zeros_like(a)
    sgn[..., ::2] = 1

    b = torch.fft.irfft(torch.fft.rfft(sgn, n=2*Nx, dim=-1)
                        * torch.fft.rfft(a*fac, n=2*Nx, dim=-1), dim=-1)[..., :Nx]

    b[..., :2] = a[..., :2]
    b[..., 2:-2] /= fac[2:-2]
    b[..., -2:] = 0
    # b[..., -4:-2] = -a[..., -2:] / torch.tensor(\
    #     [(Nx-4.0)/(Nx-2.0),(Nx-3.0)/(Nx-1.0)], dtype=torch.float64, device=a.device)**2

    return b

def cmp_robin0(a):
    Nx = a.shape[-1]
    fac = torch.linspace(0, Nx-1, Nx, dtype=a.dtype, device=a.device)
    fac = (fac-1.0)*(fac+1.0)

    sgn = torch.zeros_like(a)
    sgn[..., ::2] = 1

    b = torch.fft.irfft(torch.fft.rfft(sgn, n=2*Nx, dim=-1)
                        * torch.fft.rfft(a*fac, n=2*Nx, dim=-1), dim=-1)[..., :Nx]

    b[..., :2] = a[..., :2]
    b[..., 2:-2] /= fac[2:-2]
    b[..., -2:] = 0
    # b[..., -4:-2] = -a[..., -2:] / torch.tensor(\
    #     [(Nx-4.0)/(Nx-2.0),(Nx-3.0)/(Nx-1.0)], dtype=torch.float64, device=a.device)**2
    return b

def cmp_robin(a):
    Nx = a.shape[-1]
    fac = torch.linspace(0, Nx-1, Nx, dtype=a.dtype, device=a.device)
    fac = (fac**2+1)
    # fac = (fac-1.0)*(fac+1.0)

    sgn = torch.zeros_like(a)
    sgn[..., ::2] = 1

    b = torch.fft.irfft(torch.fft.rfft(sgn, n=2*Nx, dim=-1)
                        * torch.fft.rfft(a*fac, n=2*Nx, dim=-1), dim=-1)[..., :Nx]

    b[..., :2] = a[..., :2]
    b[..., 2:-2] /= fac[2:-2]
    return b

# def icmp(b):
#     Nx = b.shape[-1]
#     a = torch.zeros_like(b)
#     a[..., :2] = b[..., :2]
#     a[..., 2:] = b[..., 2:] - b[..., :Nx-2]
#     a[..., -2:] = -b[..., Nx-4:Nx-2]

#     return a
def icmp(b):
    Nx = b.shape[-1]
    a = torch.zeros_like(b)

    # 检查是否有足够的长度进行切片
    if Nx >= 4:
        a[..., -2:] = -b[..., Nx-4:Nx-2]
    else:
        # 处理 Nx 太小时的情况
        a[..., -2:] = 0

    a[..., 0:2] = b[..., 0:2]
    a[..., 2:] = b[..., 2:] - b[..., :Nx-2]

    return a

def icmp_neumann(b):
    Nx = b.shape[-1]
    a = torch.zeros_like(b)

    p = torch.linspace(0, Nx-3, Nx-2, dtype=torch.float64, device=b.device)
    p = ( (p/(p+2.0))**2)
    a[..., 0:2] = b[..., 0:2]
    a[..., 2:Nx-2] = b[..., 2:Nx-2] - p[:Nx-4] * b[..., :Nx-4]
    a[..., Nx-2:Nx] = -p[Nx-4:Nx-2] * b[..., Nx-4:Nx-2]

    return a

def icmp_robin0(b):
    Nx = b.shape[-1]
    a = torch.zeros_like(b)

    p = torch.linspace(0, Nx-3, Nx-2, dtype=torch.float64, device=b.device)
    p = (p-1.0)/(p+3.0)

    a[..., 0:2] = b[..., 0:2]
    a[..., 2:Nx-2] = b[..., 2:Nx-2] - p[:Nx-4] * b[..., :Nx-4]
    a[..., Nx-2:Nx] = -p[Nx-4:Nx-2] * b[..., Nx-4:Nx-2]

    return a

def icmp_robin(b):
    # pk = (p * k^2 +1) / (p * (k+2)^2 + 1)
    Nx = b.shape[-1]
    a = torch.zeros_like(b)

    pk = torch.linspace(0, Nx-3, Nx-2, dtype=torch.float64, device=b.device)
    pk = (pk**2+1) / ((pk+2.0)**2+1)

    a[..., 0:2] = b[..., 0:2]
    a[..., 2:Nx-2] = b[..., 2:Nx-2] - pk[:Nx-4] * b[..., :Nx-4]
    a[..., Nx-2:Nx] = -pk[Nx-4:Nx-2] * b[..., Nx-4:Nx-2]

    return a


def Wrapper(func_list, u, dim):
    # a wrapper to apply a list of function on given axises.
    # the func will be applied in turn.
    if type(dim) == int:
        dim = [dim]
    total_dim = u.dim()
    # print(dim)
    for d in dim:

        if (d != total_dim-1) and (d != -1):
            u = torch.transpose(u, d, -1)

        for func in func_list:
            u = func(u)

        if (d != total_dim-1) and (d != -1):
            u = torch.transpose(u, d, -1)
    return u

def Wrapper2(func_matrix, u):
    total_dim = u.dim()

    for d in u.dim():
        if (d != total_dim-1) and (d != -1):
            u = torch.transpose(u, d, -1)

        for func_list in func_matrix[d]:
            for func in func_list:
                u = func(u)

        if (d != total_dim-1) and (d != -1):
            u = torch.transpose(u, d, -1)
    return u



#x2phi = functools.partial(Wrapper, [dct, cmp_increase])
#x2phi2 = functools.partial(Wrapper, [dct, cmp])
#x2phi3 = functools.partial(Wrapper, [dct, cmp])
#phi2x = functools.partial(Wrapper, [icmp, idct])
#dctn = functools.partial(Wrapper, [dct])
#idctn = functools.partial(Wrapper, [idct])

"""
def cheb_partial(u, d):
    Nx, total_dim = u.shape[d], u.dim()
    if d != total_dim-1:
        u = torch.transpose(u, d, total_dim-1)

    tmp = torch.cat([u, u.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)

    a = torch.fft.ifft(tmp, dim=-1) * 2
    a = torch.real(a[..., :Nx])
    a[..., 0] /= 2; a[..., Nx-1] /= 2

    a = a[..., 1:] # make sure that N=2^k for FFT

    a *= 2 * torch.linspace(1, Nx-1, Nx-1, dtype=torch.float64, device=u.device)

    a = torch.flip(a, [-1])
    sgn = torch.zeros_like(a, device=u.device)
    sgn[..., 1::2] = 1

    b = torch.fft.irfft(torch.fft.rfft(sgn, n=2*(Nx-1), dim=-1)
                        * torch.fft.rfft(a, n=2*(Nx-1), dim=-1), dim=-1)

    b = torch.flip(b[..., :Nx], [-1])
    b[..., 0] /= 2
    #b[..., Nx-1] = 0

    a = b

    a[..., 0] *= 2; a[..., Nx-1] *= 2

    tmp = torch.cat([a, a.flip(dims=[-1])[..., 1:Nx - 1]], dim=-1)
    #tmp = np.concatenate([a, np.flip(a, axis=[-1])[..., 1:Nx-1]], axis=-1)
    u = torch.fft.fft(tmp, dim=-1) / 2
    u = torch.real(u[..., :Nx])

    u = torch.transpose(u, d, total_dim-1)
    return u
"""

def cheb_partial(u, d, truc = None):
    Nx, total_dim = u.shape[d], u.dim()
    if d != total_dim-1 and d != -1:
        u = torch.transpose(u, d, total_dim-1)

    V = torch.cat([u, u.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)
    a = torch.fft.ifft(V, dim=-1)[..., :Nx].real
    a[..., 1:Nx-1] *= 2

    a *= 2 * torch.linspace(0, Nx-1, Nx, dtype=torch.float64, device=u.device)
    sgn = torch.zeros(2*Nx, device=a.device, dtype=torch.float64)
    sgn[..., Nx//2*2+1::2] = 1

    b = torch.fft.irfft(torch.fft.rfft(sgn, n=2*Nx, dim=-1)
                        * torch.fft.rfft(a, n=2*Nx, dim=-1), dim=-1)[..., :Nx]

    # for j in range(Nx):
    #     b[..., j] = b

    if truc != None:
        b[..., truc:] = 0

    b[..., 0] /= 2
    #b[..., Nx-1] = 0

    a = b

    # a[..., 0] *= 2; a[..., Nx-1] *= 2
    a[..., 1:Nx-1] /= 2
    V = torch.cat([a, a.flip(dims=[-1])[..., 1:Nx - 1]], dim=-1)
    u = torch.fft.fft(V, dim=-1)[..., :Nx].real# / 2

    if d != total_dim-1 and d != -1:
        u = torch.transpose(u, d, total_dim-1)
    return u

def Value_on_boundary(u, d, truc = None):
    Nx, total_dim = u.shape[d], u.dim()
    if d != total_dim-1 and d != -1:
        u = torch.transpose(u, d, total_dim-1)

    V = torch.cat([u, u.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)
    a = torch.fft.ifft(V, dim=-1)[..., :Nx].real
    a[..., 1:Nx-1] *= 2

    a *= 2 * torch.linspace(0, Nx-1, Nx, dtype=torch.float64, device=u.device)

    b = torch.zeros_like(a)

    b[..., (Nx-3, Nx-2)] = a[..., (Nx-2, Nx-1)]
    for j in reversed(range(Nx-3)):
        b[..., j] = b[..., j+2] + a[..., j+1]

    print(b.sum(dim=-1))
    return b
    # sgn = torch.zeros(2*Nx, device=a.device, dtype=torch.float64)
    # sgn[..., Nx//2*2+1::2] = 1
    #
    # b = torch.fft.irfft(torch.fft.rfft(sgn, n=2*Nx, dim=-1)
    #                     * torch.fft.rfft(a, n=2*Nx, dim=-1), dim=-1)[..., :Nx]
    #
    # b[..., 0] /= 2

    # return torch.sum(b, dim=-1)

Dx = cheb_partial

def cmp_UpperDirichlet(a):
    b = a.cumsum(dim=-1)
    b[..., -2] = -a[..., -1]
    b[..., -1] = 0
    return b

def icmp_UpperDirichlet(b):
    a = torch.zeros_like(b)
    a[..., 1:-1] = b[..., 1:-1] - b[..., :-2]
    a[..., 0] = b[..., 0]
    a[..., -1] = -b[..., -2]
    return a

if __name__ == "__main__":
    Nx = 128
    x = torch.rand(Nx, dtype=torch.float64)
    # print(cmp_increase(x)-cmp(x))
    x = torch.cos(torch.linspace(0, np.pi, Nx, dtype=torch.float64))

    xsin = torch.sin(x)
    print(Dx(Dx(xsin, 0), 0) + xsin)

####u += torch.linspace(L, R, Nx, dtype=torch.float64, device=u.device)