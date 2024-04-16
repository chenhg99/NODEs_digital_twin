import torch
import torch.nn as nn
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
from thop import profile
sys.path.append(os.getcwd())

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def log(dirname):
    makedirs(dirname)
    log_location = '%s/0_log.txt' %(dirname)
    log = open(log_location,'a')
    sys.stdout = log

class Logger(object):
    def __init__(self, filename="Default.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def save_model(dirname, func, name):
    dict = {}
    count = 1
    for d in range(len(func.net.net)):
        if type(func.net.net[d]) == torch.nn.modules.linear.Linear:
            dict['L'+str(count)+'_weight'] = func.net.net[d].weight.cpu().numpy()
            dict['L'+str(count)+'_bias'] = func.net.net[d].bias.cpu().numpy()
            count+=1
    # np.save(f'{dirname}/{name}_net_params.npy', dict)
    torch.save(func.state_dict(), f'{dirname}/{name}_net_params.pkl')

def visualize(true_y, pred_y, t, load_path, name):
        t = t.cpu().numpy()
        true_y = true_y.cpu().numpy()
        pred_y = pred_y
        #plot true curve and predication curve
        for i in range(len(true_y[0])):
            plt.subplot(len(true_y[0]), 1, i+1)
            plt.plot(t, true_y[:, i])
            plt.plot(t, pred_y[:, i], linestyle=':')
            plt.ylabel('x' + str(i+1))

        plt.show()
        plt.savefig(f'{load_path}/{name}.png', bbox_inches='tight')
        plt.draw()
        #保存predication为.npy
        np.save(f'{load_path}/{name}.npy', pred_y)
        plt.close()

def visualize_nn(true_y, pred_y, t, load_path, name):
        t = np.float64(t.cpu().numpy())
        true_y = np.float64(true_y.cpu().numpy())
        pred_y = np.float64(pred_y)
       #plot true curve and predication curve
        for i in range(len(true_y[0])):
            plt.subplot(len(true_y[0]), 1, i+1)
            plt.plot(t, true_y[:, i])
            plt.plot(t, pred_y[:, i], linestyle=':')
            plt.ylabel('x' + str(i+1))
        plt.show()
        plt.savefig(f'{load_path}/{name}.png', bbox_inches='tight')
        plt.draw()
        plt.save(f'{load_path}/{name}.npy', pred_y)
        plt.close()

def plot_loss(load_path, loss, name):
    fig = plt.figure(figsize=(6, 4),  facecolor='white')
    plt.plot(np.arange(len(loss)), loss)
    plt.yscale('log') 
    plt.savefig(f'{load_path}/{name}_loss.png', bbox_inches='tight')
    np.save(f'{load_path}/{name}_loss.npy', loss)
    plt.close()

def get_current_time():
      local_t = time.localtime()
      year_mon_mday = str(local_t.tm_year)+str(local_t.tm_mon)+str(local_t.tm_mday)
      hour_min_sec = str(local_t.tm_hour)+str(local_t.tm_min)
      return year_mon_mday, hour_min_sec

def choose_loss_fn(args):
    if args.loss_fn == 'MSE':
        loss_fn = nn.MSELoss()
    elif args.loss_fn == "L1":
        loss_fn = nn.L1Loss()
    elif args.loss_fn == "DTW":
        from pytorch_softdtw.soft_dtw import SoftDTW
        sdtw = SoftDTW(gamma=0.1, normalize=False) # just like nn.MSELoss()
        loss_fn = lambda pred_y, y : sdtw(pred_y, y)
    return loss_fn

def choose_optimizer(args, func):
    if args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(func.parameters(), lr=args.learning_rate)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(func.parameters(), lr=args.learning_rate)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(func.parameters(), lr=args.learning_rate)   
    return optimizer 

def print_log(args, device, func):
    if args.model_type == 'ODEnet':
        print('------------------------------------------------')
        print(f'Using {device} device')
        print(f'Model type: {args.model_type}')
        print('------------------------------------------------')
        print(f'ODE method: {args.method}')
        print('------------------------------------------------')
        print(f'Data dimension: {args.input_dim}.')
        print(f'External Force: {args.F}.')
        print(f'Discard probation: {args.discard}.')
        print(f'Time interval: {args.interval}, integrate time: {args.integrate_time}.')
        print(f'Proportion for fit: ({args.for_fit}) and predication: ({args.for_predication}).')
        print('------------------------------------------------')
        # print('Seed: %s' %(args.seed))
        print(f'Batch time: {args.batch_time}' )
        print(f'Batch size: {args.batch_size}')
        print(f'Loss Function: {args.loss_fn}')
        print(f'Optimizer: {args.optimizer}')
        print(f'Learning rate: {args.learning_rate}')
        print(f'Network depth: {args.depth}, hidden num: {args.hn}, activation: {args.activation}')
        print(f'Niters(Epoch): {args.niters}')
        print(f'Frequence of plot: {args.plot_freq}')
        print('------------------------------------------------')
        print('Model:')
        print(func)
        print('------------------------------------------------')
    else:
        print('------------------------------------------------')
        print(f'Using {device} device')
        print(f'Model type: {args.model_type}')
        print('------------------------------------------------')
        print(f'Data dimension: {args.input_dim}.')
        print(f'Time interval points: {args.interval}.')
        print(f'Proportion for fit: {args.for_fit} and predication: {args.for_predication}.')
        print('------------------------------------------------')
        print(f'Seed: {args.seed}')
        print(f'Batch_size size: {args.batch_size}')
        print(f'Loss Function: {args.loss_fn}')
        print(f'Optimizer: {args.optimizer}')
        print(f'Learning rate: {args.learning_rate}')
        print(f'Network depth: {args.depth}, hidden num: {args.hn}.')
        print(f'Niters(Epoch): {args.niters}')
        print('------------------------------------------------')
        print('Model:')
        print(func)
        print('------------------------------------------------')

class Dense(nn.Module):
    def __init__(self, input_dim, layer_num, hidden_num, activation, device):
        super(Dense, self).__init__()
        self.device = device
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Relu':
            self.activation = nn.ReLU()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()

        if activation == 'No':
            modules = [nn.Linear(input_dim, hidden_num)]
        else:
            modules = [nn.Linear(input_dim, hidden_num), self.activation]

        for i in range(layer_num):
            modules.append(nn.Linear(hidden_num, hidden_num))
            if activation != 'No':
                modules.append(self.activation)#tanh or sigmod or relu

        modules.append(nn.Linear(hidden_num, input_dim))
        self.net = nn.Sequential(*modules).to(self.device)
        #weight init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0.1)
        #record
        self.lst = [k for k in self.net.state_dict().keys()]

        
    def forward(self, y):
        out = self.net(y).to(self.device)
        return out

    def write_noise_update(self, var):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                with torch.no_grad():
                    w_n = torch.normal(mean=1, std=var, size = m.weight.size()).to(self.device)
                    weight_noise = torch.mul(m.weight, w_n)
                    m.weight.copy_(weight_noise)
                    r_n = torch.normal(mean=1, std=var, size = m.bias.size()).to(self.device)
                    bias_noise =torch.mul(m.bias, r_n)
                    m.bias.copy_(bias_noise)

    def read_noise_update(self, var, net, lst):
        for i in range(len(lst)):
            noise = nn.init.normal_(self.net.state_dict()[self.lst[i]], mean=0, std=var)
            self.net.state_dict()[self.lst[i]].copy_(torch.mul(net.state_dict()[lst[i]].to(self.device), noise)).to(self.device)

class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc

    def forward(self, batch_y0, batch_t, method):
        self.batch_t = batch_t.cpu()
        out = odeint(self.odefunc, batch_y0, batch_t, rtol=1e-3, atol=1e-3, method=method)  
        return out

class ODEFunc_noise(nn.Module):
    def __init__(self, input_dim, depth, hidden_num, activation, wr, rr, device):
        super(ODEFunc_noise, self).__init__()
        self.device = device
        self.wr = wr
        self.rr = rr
        self.net = Dense(input_dim, depth, hidden_num, activation, self.device).to(self.device)
        self.net_with_read_noise = Dense(input_dim, depth, hidden_num, activation, self.device).to(self.device)
        self.if_noise = True
        # add read noise
        self.lst = [k for k in self.net.state_dict().keys()]

    def forward(self, t, y):
        if self.if_noise:
            # add write noise
            if t == 0 and self.wr != 0:
                self.net.write_noise_update(var = self.wr)

            if self.rr != 0:
                self.net_with_read_noise.read_noise_update(var=self.rr, net=self.net, lst=self.lst)
            #calculate the output
            out = self.net(y) + self.net_with_read_noise(y)
        else:
            out = self.net(y)
        return out


class Net(nn.Module):
    def __init__(self, model_type, input_size, hn, depth, batch_first, wr, rr, device):
        super(Net, self).__init__()
        self.device = device
        self.wr = wr
        self.rr = rr
        self.model_type = model_type
        if self.model_type == "RNN":
                self.rnn = nn.RNN(
                    input_size=input_size,                         #feature_len=1
                    hidden_size=hn,                                #隐藏记忆单元尺寸hidden_len
                    num_layers=depth,                              #层数
                    batch_first=batch_first,                       #在传入数据时,按照[batch,seq_len,feature_len]的格式
                )
        elif self.model_type == "LSTM": 
                self.rnn = nn.LSTM(
                    input_size=input_size,                         #feature_len=1
                    hidden_size=hn,                       #隐藏记忆单元尺寸hidden_len
                    num_layers=depth,                         #层数
                    batch_first=batch_first,                       #在传入数据时,按照[batch,seq_len,feature_len]的格式
                )
        elif self.model_type == "GRU": 
                    self.rnn = nn.GRU(
                    input_size=input_size,                         #feature_len=1
                    hidden_size=hn,                       #隐藏记忆单元尺寸hidden_len
                    num_layers=depth,                         #层数
                    batch_first=batch_first,                       #在传入数据时,按照[batch,seq_len,feature_len]的格式
                )

        self.rnn_read_noise = nn.RNN(
                input_size=input_size,                         #feature_len=1
                hidden_size=hn,                                #隐藏记忆单元尺寸hidden_len
                num_layers=depth,                              #层数
                batch_first=batch_first,                       #在传入数据时,按照[batch,seq_len,feature_len]的格式
        )

        for p in self.rnn.parameters():                    #对RNN层的参数做初始化
                nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear = nn.Linear(hn, input_size)  #输出层
        self.lst = [k for k in self.rnn.state_dict().keys()]
        self.if_cal = False

    def forward(self, x, hidden_prev):
        '''
        x：一次性输入所有样本所有时刻的值(batch,seq_len,feature_len)
        hidden_prev：第一个时刻空间上所有层的记忆单元(batch,num_layer,hidden_len)
        输出out(batch,seq_len,hidden_len)和hidden_prev(batch,num_layer,hidden_len)
        '''
        # # add write noise
        # if self.wr != 0 and self.symbol == 0:
        #       self.write_noise_update(var = self.wr)
        #       self.symbol = 1
        #   # add read noise
        #   self.read_noise_update(var = self.rr)
        #   hidden_prev_input = hidden_prev.detach()

        #   r_out, hidden_prev = self.rnn(x, hidden_prev)
        #   r_out_noise, hidden_prev_noise = self.rnn_read_noise(x, hidden_prev_input)

        #   r_out = r_out + r_out_noise
        #   hidden_prev = hidden_prev + hidden_prev_noise
        r_out, hidden_prev = self.rnn(x, hidden_prev)
        
        if self.if_cal:
            macs, _ = profile(self.rnn, inputs=(x, hidden_prev), verbose=False)
            self.total_flop['rnn'] += macs*2

        #因为要把输出传给线性层处理，这里将batch和seq_len维度打平，再把batch=1添加到最前面的维度（为了和y做MSE）
        outs=[]#获取所有时间点下得到的预测值
        for time_step in range(r_out.size(1)): #将记忆rnn层的输出传到全连接层来得到最终输出。 这样每个输入对应一个输出，所以会有长度为10的输出
                outs.append(self.linear(r_out[:,time_step,:]))
                if self.if_cal:
                    macs, _ = profile(self.linear, inputs=(r_out[:,time_step,:], ), verbose=False)
                    self.total_flop['mlp'] += macs*2
        #   out = out.view(-1, hn)    #[batch=1,seq_len,hidden_len]->[seq_len,hidden_len]
        #   out = self.linear(out)             #[seq_len,hidden_len]->[seq_len,feature_len=1]
        out = torch.stack(outs,dim=1)        #[seq_len,feature_len=1]->[batch=1,seq_len,feature_len=1]
        return out, hidden_prev

    def write_noise_update(self, var):
            for i in self.lst:
                  with torch.no_grad():
                        # weight_noise = torch.mul(m.weight,nn.init.normal_(m.weight, mean=1, std=var))
                        m = self.rnn.state_dict()[i]
                        w_n = torch.normal(mean=1, std=var, size = m.size()).to(self.device)
                        weight_noise = torch.mul(m, w_n)
                        self.rnn.state_dict()[i].copy_(weight_noise)

            for m in self.linear.modules():
                  if isinstance(m, nn.Linear):
                        with torch.no_grad():
                              # weight_noise = torch.mul(m.weight,nn.init.normal_(m.weight, mean=1, std=var))
                              w_n = torch.normal(mean=1, std=var, size = m.weight.size()).to(self.device)
                              weight_noise = torch.mul(m.weight, w_n)
                              m.weight.copy_(weight_noise)
                              # bias_noise =torch.mul(m.bias, nn.init.normal_(m.bias, mean=1, std=var))
                              r_n = torch.normal(mean=1, std=var, size = m.bias.size()).to(self.device)
                              bias_noise =torch.mul(m.bias, r_n)
                              m.bias.copy_(bias_noise)

    def read_noise_update(self, var):
            for i in self.lst:
                  with torch.no_grad():
                        noise = nn.init.normal_(self.rnn.state_dict()[i], mean=0, std=var)
                        self.rnn_read_noise.state_dict()[i].copy_(torch.mul(self.rnn.state_dict()[i], noise))
