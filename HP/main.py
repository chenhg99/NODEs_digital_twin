import time
import argparse
import torch
import sys
from torch import optim
import os
sys.path.append(os.getcwd())
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from pytorch_softdtw.soft_dtw import SoftDTW
sdtw = SoftDTW(gamma=0.1, normalize=False) # just like nn.MSELoss()
sys.path.append('/HP_memristor_result')
from utils import *
from model import *
from HP.data_utils import Biolek
parser = argparse.ArgumentParser('ODE predicate HP memristor state')
# network parameters
parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--model', type=str, choices=['odenet', 'resnet'], default='resnet')
parser.add_argument('--method', type=str, choices=['dopri5', 'rk4'], default='dopri5')
parser.add_argument('--t', type=float, default=0.5)
parser.add_argument('--w0', type=float, default=0.01)
parser.add_argument('--dim', type=int, default=1)
parser.add_argument('--ext_dim', type=int, default=1)
parser.add_argument('--neural_num', type=int, default=14)
parser.add_argument('--rtol', type=np.float32, default=1e-3)
parser.add_argument('--atol', type=np.float32, default=1e-3)

# input parameters
parser.add_argument('--Am', type=float, default=0.091)
parser.add_argument('--period', type=float, default=0.1)
parser.add_argument('--bias', type=float, default=0.0485)
parser.add_argument('--offset', type=float, default=0)
parser.add_argument('--input_type', type=str, choices=['sin', 'square', 'triangle', 'mod sin'], default='sin')

#Seting Training
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--data_size', type=int, default=500)
parser.add_argument('--batch_time', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--optimizer', type=str, choices=['Adam', 'RMSprop', 'SGD'], default='Adam')
parser.add_argument('--loss_fn', type=str, choices=['L1', 'mse', 'rse', 'dtw'], default='L1')
parser.add_argument('--learning_rate', type=np.float32, default=1e-3)
parser.add_argument('--iteration', type=np.int, default=1)
parser.add_argument('--epoch', type=int, default=2000)
parser.add_argument('--plot_freq', type=int, default=50)
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def get_batch(data_size, batch_time, batch_size, t, true_y, ext):
    s = torch.from_numpy(np.random.choice(np.arange(0, data_size - batch_time - 1, dtype=np.int64), batch_size, replace=False))
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    batch_ext = torch.stack([ext[s + i] for i in range(batch_time)], dim=0)
    batch_y0 = torch.unsqueeze(true_y[s], dim=-1)
    batch_y = torch.unsqueeze(batch_y, dim=-1)
    batch_ext = torch.unsqueeze(batch_ext, dim=-1)
    return batch_y0, batch_t, batch_y, batch_ext, s

if __name__ == "__main__":
    #record time
    local_t = time.localtime()
    year_mon_mday = str(local_t.tm_year)+str(local_t.tm_mon)+str(local_t.tm_mday)
    hour_min_sec = str(local_t.tm_hour)+str(local_t.tm_min)
    #create dir
    main_dirname = os.getcwd()+'/HP/results/'+year_mon_mday+'_'+hour_min_sec+'/'+str(args.t)+'_'+str(args.data_size)+'_'+str(args.neural_num)+'/'
    makedirs(main_dirname)

    sub_dirname = main_dirname + str(args.model)+'/'
    makedirs(sub_dirname)
    #Train_Loss = loss_iteration()

    for train_iter in range(args.iteration):
        # record result
        dirname = sub_dirname + 'Iter'+str(train_iter)+'/'
        makedirs(dirname)
        logger = get_logger(logpath=os.path.join(dirname, 'logs'), filepath=os.path.abspath(__file__))
        logger.info(args)

        # creat dataset parameters
        t, ext, true_y, true_y0 = Biolek( type   = args.input_type, 
                                    t      = torch.Tensor(np.linspace(0, args.t, args.data_size)), 
                                    Am     = torch.Tensor([args.Am]), 
                                    period = torch.Tensor([args.period]), 
                                    offset = torch.Tensor([args.offset]), 
                                    bias   = torch.Tensor([args.bias]), 
                                    w0     = torch.Tensor([args.w0]),
                                    device = device)
        # creat model
        if args.model == "odenet":
            model = ODEBlock(ODEFunc(dim=args.dim, ext_dim=args.ext_dim, neural_num=args.neural_num, 
                                     layer_num = 3, activation = nn.ReLU(),
                                     device=device, Am=args.Am, period=args.period, bias=args.bias))

        else:
            model = ResBlock(Resnet(dim=args.dim, ext_dim=args.ext_dim, neural_num = args.neural_num, 
                                    layer_num = 3, activation = nn.ReLU(), device=device))
        # record model
        logger.info(model)
        logger.info('++++++++++++++++++++++++++++++++++++')
        logger.info('Number of parameters: {}'.format(count_parameters(model)))
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
        global s
        train_loss = loss_manage()
        test_loss = loss_manage()
        itr = 0
        # train
        while (itr < args.epoch):
            itr = itr + 1
            optimizer.zero_grad()
            batch_y0, batch_t, batch_y, batch_ext, s = get_batch(args.data_size, args.batch_time, args.batch_size, t, true_y, ext)
            if args.model == "odenet":
                pred_y = model(batch_y0, batch_t, s, rtol=1e-3, atol=1e-3)
            else:
                pred_y = model(batch_y0, batch_ext)
            l1, mse, mrs, dtw = train_loss.calculate(pred_y, batch_y)
            # select loss function
            if args.loss_fn == "L1":
                l1.backward()
            elif args.loss_fn == "mse":
                mse.backward()
            elif args.loss_fn == "mre":
                mrs.backward()
            elif args.loss_fn == "dtw":
                dtw.backward()

            optimizer.step()
            # print loss
            if itr % args.plot_freq == 0:
                with torch.no_grad():
                    s = t[:1]
                    if args.model == "odenet":
                        pred_y = model(true_y0.unsqueeze(axis=-1).unsqueeze(axis=-1), t, s, rtol=1e-3, atol=1e-3)[:,0]
                    else:
                        pred_y = model(torch.unsqueeze(true_y0, dim=-1), torch.unsqueeze(ext, dim=-1))
                    # L1, MSE, MRE, DTW 
                    l1, mse, mre, dtw = test_loss.calculate(pred_y, torch.unsqueeze(true_y, dim=-1))

                    Test_Loss = mre.item()
                    plot(t, ext, pred_y, true_y, itr, Test_Loss, plot_name=args.model, dirname=dirname)
                    print('Iter {:04d}'.format(itr))
                    logger.info("Epoch {:04d} | L1 {:.4f} | MSE {:.4f} | MRE {:.4f} | DTW {:.4f}" .format(
                         itr, l1.item(), mse.item(), mre.item(), dtw.item())
                         )
        # plot loss
        plot_loss(test_loss,  dirname=dirname, name="Test_Loss")
        plot_loss(train_loss, dirname=dirname, name="Train_Loss") 
        test_loss.save(path = dirname, name="Test_Loss")   
        train_loss.save(path = dirname, name="Train_Loss")
        #save model
        save_model(dirname+'%s_model_dict.pth'%(args.model), args.epoch, optimizer, model)