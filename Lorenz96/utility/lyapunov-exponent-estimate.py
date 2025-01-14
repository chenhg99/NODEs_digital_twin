import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
sys.path.append('/home/hegan/Lorenz96_odenet/Fig4/utility/lyapunov-exponent-estimate/')
from LyapunovExponentEstimate import lyapunov_solve, lyapunov_solve_unknown, solve_roessler, solve_lorenz
sys.path.append('/home/hegan/Lorenz96_odenet/Fig4/utility/')
from Lorenz96.utility.data_utils import odeint_lorenz96

# 设置随机数种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

torch.set_default_tensor_type(torch.DoubleTensor)
#The number of variable
N = torch.tensor(6)
# Forcing
F = torch.tensor(6)
y0 = torch.tensor([-1.2061,  0.0617,  1.1632, -1.5008, -1.5944, -0.0187]) 
setup_seed(20)
# y0 = torch.randn(N)
tmax = 36
interval = 2000
discard = 0
t, output, y0 = odeint_lorenz96(N, F, y0, tmax, interval, discard)

output = np.array(output)
lyapunov_solve(output, float(tmax/interval), float(tmax*1e-3), plot_replace=True, plot_convergence=True)