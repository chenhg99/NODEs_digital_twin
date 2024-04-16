import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

class ResBlock(nn.Module):
    def __init__(self, resnet):
        super(ResBlock, self).__init__()
        self.resnet = resnet

    def forward(self, ic, batch_ext):
        output = ic.unsqueeze(0)
        for i in range(batch_ext.size(0) - 1):
            ic = self.resnet(ic, batch_ext[i])
            output = torch.cat((output, ic.unsqueeze(0)), dim=0)
        return output
      

class Resnet(nn.Module):
    def __init__(self, dim, ext_dim, neural_num, layer_num, activation, device):
        super(Resnet, self).__init__()
        self.device = device
        self.act = activation
        self.fc1 = nn.Linear(dim + ext_dim, neural_num)
        self.fc2 = nn.Linear(neural_num, dim)
    
        modules = [self.fc1, self.act]
        for i in range(layer_num - 2):
            modules.append(nn.Linear(neural_num, neural_num))
            modules.append(self.act)
        modules.append(self.fc2)
        self.net = nn.Sequential(*modules).to(self.device)
        #weight init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0.1)

    def forward(self, ic, ext): 
        x = torch.cat((ext, ic), dim=-1).to(self.device)
        x = self.net(x).to(self.device)
        return x + ic

    
class ODEBlock(nn.Module):
    def __init__(self, odefunc, method='rk4'):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.method = method

    def forward(self, batch_y0, batch_t, s, rtol, atol):
        self.odefunc.s = s
        self.batch_t = batch_t
        out = odeint(self.odefunc, batch_y0, batch_t, rtol=rtol, atol=atol, method=self.method)
        return out
    
class ODEFunc(nn.Module):
    def __init__(self, dim, ext_dim, neural_num, layer_num, activation, device, Am, period, bias):
        super(ODEFunc, self).__init__()
        self.dim          =   dim
        self.ext_dim      =   ext_dim
        self.neural_num   =   neural_num
        self.device       =   device
        self.Am           =   torch.tensor([Am])
        self.period       =   torch.tensor([period])
        self.bias         =   torch.tensor([bias])
        self.divisor      =   torch.Tensor([100])
        self.pi = torch.tensor([torch.acos(torch.zeros(1)).item()*2], dtype=torch.float32)

        self.act = activation
        self.fc1 = nn.Linear(dim + ext_dim, neural_num)
        self.fc2 = nn.Linear(neural_num, dim)
    
        modules = [self.fc1, self.act]
        for i in range(layer_num - 2):
            modules.append(nn.Linear(neural_num, neural_num))
            modules.append(self.act)
        modules.append(self.fc2)
        self.net = nn.Sequential(*modules).to(self.device)
        #weight init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0.1)

    def get_ext(self, t):
        time = self.period*torch.div(self.s%self.divisor, self.divisor)
        output = torch.add(self.Am*torch.sin((2*self.pi/self.period)*(time+t)),self.bias)
        return output.unsqueeze(1).to(self.device)

    def forward(self, t, y):
        ext = self.get_ext(t).to(self.device)
        x = torch.cat((y.to(self.device), ext), dim=-1).to(self.device)
        x = self.net(x).to(self.device)
        return x