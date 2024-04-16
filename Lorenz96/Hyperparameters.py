import argparse
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from Lorenz96.ODEnet import ODEnet
from Lorenz96.other_NN import other_model
parser = argparse.ArgumentParser('Lorenz 96')
# model parameter
parser.add_argument('--model_type', type=str, choices=['ODEnet', 'RNN', 'LSTM', 'GRU'], default='RNN')
parser.add_argument('--method', type=str, choices=['dopri5', 'rk4','euler','dopri8'], default='dopri5')
# Dataset parameter
parser.add_argument('--input_dim', type=int, default=6)
parser.add_argument('--F', type=np.float32, default=6)
parser.add_argument('--interval', type=int, default=2400)
parser.add_argument('--integrate_time', type=int, default=48)
parser.add_argument('--discard', type=np.float32, default=0)
# Noise parameter
parser.add_argument('--write_noise', type=np.float32, default=0)
parser.add_argument('--read_noise', type=np.float32, default=0)
#Proportion of Fit and Predication 
parser.add_argument('--for_fit', type=np.float32, default=0.75)
parser.add_argument('--for_test', type=np.float32, default=0)
parser.add_argument('--for_predication', type=np.float32, default=0.25)
#model
parser.add_argument('--depth', type=int, default=1)
parser.add_argument('--hn', type=int, default=128)
parser.add_argument('--activation', type=str, choices=['Sigmoid', 'Tanh', 'Relu'], default='Relu')
parser.add_argument('--learning_rate', type=np.float32, default=1e-3)
#Setting Training
parser.add_argument('--run_loop', type=int, default=10)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--loss_standard', type=np.float32, default=0.1)
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--batch_time', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--optimizer', type=str, choices=['Adam', 'RMSprop', 'SGD'], default='Adam')
parser.add_argument('--loss_fn', type=str, choices=['DTW', 'L1', 'MSE'], default='DTW')
parser.add_argument('--gpu', type=int, default=0)
#other
parser.add_argument('--comment', type=str, default='demo')
parser.add_argument('--path', type=str, default=os.getcwd()+"/Lorenz96/results/")
args = parser.parse_args()

if args.model_type == 'ODEnet':
      ODEnet(args)  
else:
      other_model(args)
