from multiprocessing.sharedctypes import Value
from unittest import result
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint
import math
import numpy as np
import matplotlib.pyplot as plt
import torch

def input(type, t, Am, period, offset, bias):
    if type == 'sin':
        np.pi=torch.acos(torch.zeros(1)).item()*2
        output=torch.add(torch.Tensor([Am])*torch.sin((2*np.pi/torch.Tensor([period]))*(t+offset*period)),torch.Tensor([bias]))
    elif type == 'square':
        output = torch.Tensor([Am])*torch.sign(torch.sin((2*np.pi/torch.Tensor([period]))*(t+offset*period))) + torch.Tensor([bias])
    elif type == 'triangle':
        shifted_t = t - offset
        phase = torch.Tensor(shifted_t / period)
        output = torch.Tensor([Am])* (2 * torch.abs(2 * (phase - torch.floor(phase + 0.5))) - 1) + torch.Tensor([bias])
    elif type == 'sawtooth':
        phase = ((t / period) + offset) % 1
        output = torch.Tensor([Am]) * (2 * phase - 1) + torch.Tensor([bias])
    elif type == 'rand':
        output = torch.Tensor([Am])*torch.rand_like(torch.tensor(t))
    elif type == 'mod sin':
        baseband_signal = torch.Tensor([Am])*torch.sin((2*np.pi/torch.Tensor([period]))*(t+offset*period))
        carrier_signal = np.cos((4*np.pi/torch.Tensor([period]))*(t+offset*period))
        modulated_signal = -1*baseband_signal * carrier_signal  + torch.Tensor([bias])
        output = modulated_signal
    elif type == 'randn':
        output = torch.Tensor([Am])*torch.subtract(torch.rand_like(torch.tensor(t)), torch.tensor([0.5]))
    return output


def HPCVRM(type, t, Am, period, offset, bias, w0,
           Ron = torch.tensor(1e2), 
           Roff = torch.tensor(160*1e2), 
           k = torch.tensor(10e-10*(1e2/10e-9)), 
           D = torch.tensor(10e-9), 
           uV = torch.tensor(10e-10), device='cpu'):
    def f_wt(t, x):
        i = input(type, t, Am, period, offset, bias)
        out = k*i
        return out
    def f(t, i, w0):
        wt = odeint(f_wt, w0, t, rtol=1e-4, atol=1e-4, method='rk4').squeeze(-1)
        v = (Ron*(wt/D)+Roff*(torch.Tensor([1])-(wt/D)))*i
        return v, wt

    i = input(type, t, Am, period, offset, bias)
    v, wt = f(t, i, w0)

    t = torch.tensor(t, dtype=torch.float32).to(device)
    ext = torch.tensor(i, dtype=torch.float32).to(device)
    true_y = 0.5*torch.divide(wt, max(wt)).to(device)
    true_y0 = torch.tensor((true_y[0]), dtype=torch.float32).to(device)

    return t, ext, true_y, true_y0