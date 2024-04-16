import sys
import os
sys.path.append(os.getcwd())
from torchdiffeq import odeint
import numpy as np
import torch

# 设置随机数种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

def odeint_lorenz96(N, F, y0, args, device):
    torch.set_default_tensor_type(torch.DoubleTensor)
    def L96(t, x):
        """Lorenz 96 model with constant forcing"""
        # Setting up vector
        d = torch.zeros(N)
        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        for i in range(N):
            d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
        return d

    setup_seed(args.seed)
    t = torch.linspace(0, args.integrate_time, int(args.interval*(1 + args.discard)))
    true_y = odeint(L96, y0, t, rtol=1e-3, atol=1e-3, method='rk4')
    t = t[:args.interval].to(device)
    true_y = true_y[int(args.interval*args.discard):].to(device)
    y0 = true_y[0]

    #fit and predication slice
    fit_slice = int(args.for_fit * args.interval)
    predication_slice = int((1 - args.for_predication)* args.interval)

    fit_y_true = true_y[:predication_slice]
    predicate_y_true = true_y[predication_slice:]

    return t, true_y, y0, fit_y_true, predicate_y_true

# N = torch.tensor(6) #The number of variable
# F = torch.tensor(6) # Forcing
# # y0 = torch.tensor([-1.2061,  0.0617,  1.1632, -1.5008, -1.5944, -0.0187]) 
# setup_seed(20)
# y0 = torch.randn(N)
# tmax = 36
# interval = 1800
# discard = 0
# t, output, y0 = odeint_lorenz96(N, F, y0, tmax, interval, discard)