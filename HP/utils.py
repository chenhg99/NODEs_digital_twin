import logging
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_softdtw.soft_dtw import SoftDTW
sdtw = SoftDTW(gamma=0.1, normalize=False) # just like nn.MSELoss()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def plot(t, ext, pred_y, true_y, itr, loss, plot_name, dirname):
    fig = plt.figure(figsize=(6,6), facecolor='white')
    plt.subplot(2, 1, 1)
    plt.plot(t.cpu().numpy(), ext.cpu().numpy())
    plt.ylabel('Input signal')
    plt.subplot(2, 1, 2) 
    plt.plot(pred_y[:,0].cpu(), label='pred', color='blue')
    plt.plot(true_y.cpu(), label='true', color='orange')
    plt.title('%s  Iteration %d, Loss %.4f' % (plot_name, itr, loss))
    plt.legend()
    plt.show()
    plt.savefig(dirname+'pic-{}_{:.6f}.png'.format(itr, loss))
    plt.close(fig)

def plot_loss(loss, dirname, name):
    plt.figure(figsize=(10,10))
    #画图，使用log纵坐标
    plt.semilogy(loss.L1, c='r', label='L1')
    plt.semilogy(loss.mse_loss, c='b', label='mse_loss')
    plt.semilogy(loss.mre_loss, c='y', label='mre_loss')
    plt.semilogy(loss.dtw_loss, c='g', label='dtw_loss')
    plt.title(name)
    plt.legend()
    plt.savefig(dirname+'%s.png'%(name))

class loss_iteration(object):
    def __init__(self):
        self.L1  = []
        self.mse_loss = [] 
        self.mre_loss = []
        self.dtw_loss = []

    def append(self, L1, mse_loss):
        self.L1.append(L1)
        self.mse_loss.append(mse_loss)
        
    def save(self, path):
        np.save(path +'iteration_L1.npy', self.L1)
        np.save(path +'iteration_mse_loss.npy', self.mse_loss)
        np.save(path +'iteration_mre_loss.npy', self.mre_loss)
        np.save(path +'iteration_dtw_loss.npy', self.dtw_loss)

    def plot(self, path, name):
        fig = plt.figure(figsize=(10, 5))
        y_min = np.min(self.L1, axis=0)
        y_max = np.max(self.L1, axis=0)
        y_average = np.mean(self.L1, axis=0)
        x = np.arange(len(self.L1[0]))
        plt.fill_between(x, y_min, y_max, alpha=0.8, color='b')
        plt.plot(x, y_average, color='r')
        #刻度显示为log
        plt.yscale('log')
        plt.savefig(path+'%s.png'%(name))
        plt.close()

# return four error : L1, MSE, MRE, DTW 
class loss_manage(object):
    def __init__(self):
        self.L1       = []
        self.mse_loss = [] 
        self.mre_loss = [] 
        self.dtw_loss = []
      
    def calculate(self, pred, target):
        l1 = nn.L1Loss()(pred, target)
        mse = nn.MSELoss()(pred, target)
        mre = torch.mean(torch.abs(pred - target / target))
        if len(pred.shape) == 3:
            dtw = sdtw(pred.permute(1,0,2), target.permute(1,0,2)).mean()
        else:
            dtw = sdtw(pred.permute(1,0), target.permute(1,0)).mean()

        self.L1_update(l1)
        self.mes_update(mse)
        self.mre_update(mre)
        self.dtw_update(dtw)
        return l1, mse, mre, dtw
    
    def L1_update(self, L1):
        self.L1.append(np.round(L1.item(), 8))

    def mes_update(self, mse):
        self.mse_loss.append(np.round(mse.item(), 8))

    def mre_update(self, mre):
        self.mre_loss.append(np.round(mre.item(), 8))
      
    def dtw_update(self, dtw):
        self.dtw_loss.append(np.round(dtw.item(), 8))

    def save(self, path, name):
            np.save(path +  name +'_L1.npy',  self.L1)
            np.save(path +  name + '_mse_loss.npy', self.mse_loss)
            np.save(path +  name +'_mre_loss.npy', self.mre_loss)
            np.save(path +  name +'_dtw_loss.npy', self.dtw_loss)

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(save_path, epoch, optimizer, model):
    torch.save({'epoch': epoch+1,
                'optimizer_dict': optimizer.state_dict(),
                'model_dict': model.state_dict()},
                save_path)
    print("model save success")

def load_model(save_name, optimizer, model):
    model_data = torch.load(save_name)
    model.load_state_dict(model_data['model_dict'])
    # optimizer.load_state_dict(model_data['optimizer_dict'])
    print("model load success")
    return model

def plot(t, ext, pred_y, true_y, itr, loss, plot_name, dirname):
    fig = plt.figure(figsize=(6,6), facecolor='white')
    plt.subplot(2, 1, 1)
    plt.plot(t.cpu().numpy(), ext.cpu().numpy())
    plt.ylabel('Input signal')
    plt.subplot(2, 1, 2) 
    plt.plot(pred_y[:,0].cpu(), label='pred', color='blue')
    plt.plot(true_y.cpu(), label='true', color='orange')
    plt.title('%s  Iteration %d, Loss %.4f' % (plot_name, itr, loss))
    plt.legend()
    plt.show()
    plt.savefig(dirname+'pic-{}_{:.6f}.png'.format(itr, loss))
    plt.close(fig)
