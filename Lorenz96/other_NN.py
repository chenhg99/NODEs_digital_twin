import numpy as np
import os
import sys
import os
sys.path.append(os.getcwd())
from  Lorenz96.utility.utilities import *
from Lorenz96.utility.data_utils import odeint_lorenz96
import torch
import datetime

def other_model(args):
    # device
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    # dataset parameter
    N = torch.tensor(args.input_dim) # Number of variables
    F = torch.tensor(args.F) # Forcing
    y0 = torch.tensor([-1.2061,  0.0617,  1.1632, -1.5008, -1.5944, -0.0187]) # Initial conition
    t, true_y, y0, _, _ = odeint_lorenz96(N, F, y0, args, device) # Generate dataset
    feature_len = len(true_y[0])
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    args.batch_size = 1
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # args.for_fit used for training
    fit_y_true = torch.as_tensor(true_y[:int(args.for_fit*args.interval),:]).to(device)
    pred_y_true = torch.as_tensor(true_y[int(args.for_fit*args.interval):,:]).to(device)

    # slicing
    fit_slice = int(args.for_fit*args.interval)
    pred_slice = int((1 - args.for_fit)*args.interval) + 1

    # x as input data, y as true data
    x = fit_y_true[:-1, :].view(args.batch_size, fit_slice - 1, feature_len).to(torch.float64).to(device)
    y = fit_y_true[1:, :].view(args.batch_size, fit_slice - 1, feature_len).to(torch.float64).to(device)
    
    for i in range(args.run_loop):
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #define model
        model = Net(model_type = args.model_type, input_size=args.input_dim, hn=args.hn, depth=args.depth, batch_first=True, wr=0, rr=0, device=device).to(device)
        # Choose Loss function
        loss_fn = choose_loss_fn(args)
        # Choose optimizer
        optimizer = choose_optimizer(args, model)   
        # Create folder for saving 
        save_name = f'{str(args.depth)}_{str(args.hn)}_{str(args.activation)}_{str(args.interval)}_{str(args.integrate_time)}_{str(args.method)}_{str(args.optimizer)}_wr{str(args.write_noise)}_rr{str(args.read_noise)}'
        dirname = f'{args.path}{args.model_type}_{args.comment}/dim{str(args.input_dim)}_{str(args.loss_fn)}_{save_name}/{str(i)}'
        makedirs(dirname)
        log_location = '%s/0_log.txt'%(dirname)
        log = open(log_location,'a+')
        sys.stdout = log
        print_log(args, device, model)
        log.close()

        #Trainning
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Define some param
        ii = 0
        jj = 0
        fit_min_loss = 500
        Fit_loss_log = []
        pre_min_loss = 500
        Pred_loss_log = []
        # mark time
        starttime = datetime.datetime.now()
        hidden_prev_train = torch.zeros(args.depth, args.batch_size, args.hn).to(device)      
        c_train = torch.zeros(args.depth, args.batch_size, args.hn).to(device)
        for itr in range(args.niters):
            model.train()
            optimizer.zero_grad()
            #Fit lorenz time series
            if args.model_type == "LSTM":
                output, (hidden_prev_train, c_train) = model(x, (hidden_prev_train, c_train))
                c_train = c_train.detach()
            else:
                output, hidden_prev_train = model(x, hidden_prev_train)
            
            hidden_prev_train = hidden_prev_train.detach()
            loss = loss_fn(output, y)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            if itr % 1 == 0:
                log = open(log_location,'a+')
                sys.stdout = log
                endtime = datetime.datetime.now()
                print('------------------------------------------------')
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()), end='\r')
                print('Itr %s Runnig time : %s'%(itr, endtime-starttime))
                log.close()

                #Testing
                with torch.no_grad():
                    #initial input
                    input = x[:, 0, :]
                    input = input.view(1, 1, feature_len)   #input：[1,1,1]
                    #initial sorted data
                    fit_data = []
                    fit_data.append(input.cpu().detach().numpy().ravel())
                    #initial hidden state
                    hidden_prev = torch.zeros(args.depth, args.batch_size, args.hn).to(device)
                    c = torch.zeros(args.depth, args.batch_size, args.hn).to(device)
                    #0-1800 used for fitting
                    for _ in range(fit_slice - 1):             #迭代seq_len次
                            if args.model_type == "LSTM":
                                pred, (hidden_prev, c) = model(input, (hidden_prev, c))
                                c = c.detach()
                            else:
                                pred, hidden_prev = model(input, hidden_prev)
                            input = pred
                            fit_data.append(pred.cpu().detach().numpy().ravel())

                    fit_data = np.array(fit_data)
                    fit_data = torch.from_numpy(fit_data).to(device)
                    fit_loss = loss_fn(fit_data, fit_y_true)
                    Fit_loss_log.append(fit_loss.item())

                    #initial sorted data
                    pred_data = []
                    pred_data.append(input.cpu().detach().numpy().ravel())
                    #initialize hidden_prev
                    #1800-2000 used for prediction
                    for _ in range(pred_slice - 2):
                        if args.model_type == "LSTM":
                            pred, (hidden_prev, c) = model(input, (hidden_prev, c))
                            c = c.detach()
                        else:   
                            pred, hidden_prev = model(input, hidden_prev)
                            input = pred
                        pred_data.append(pred.cpu().detach().numpy().ravel())
                        
                    pred_data = np.array(pred_data)
                    pred_data = torch.from_numpy(np.array(pred_data)).to(device)
                    pre_loss = loss_fn(pred_data, pred_y_true)
                    Pred_loss_log.append(pre_loss.item())
                    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    #predicate 40s, 36s for fitting and 4s for predicate
                    #think about hidden state of RNN, LSTM, GRU
                    if fit_loss<= fit_min_loss:
                        #record log
                        log = open(log_location,'a+')
                        sys.stdout = log
                        #save model
                        load_path = dirname + '/'+'0_fit_params.npy'
                        torch.save(model.state_dict(), load_path)
       
                        #figure number +1
                        ii+=1
                        #update min fit loss
                        fit_min_loss = fit_loss

                        #save Fitting figure
                        fig = plt.figure(figsize=(10, 2*args.input_dim),  facecolor='white')
                        print(str(ii) + ' Itr {:04d} | Fit Loss {:.6f}'.format(itr, fit_loss.item()))
                        save_name = str(ii) + '_fit_' + '%.4f'%fit_loss + '.png'
                        fit_data = np.array(fit_data.cpu())
                        visualize(fit_y_true, fit_data, t[:fit_slice], dirname, save_name)
                        
                        #Record time
                        endtime = datetime.datetime.now()
                        print('------------------------------------------------')
                        print('Itr %s Runnig time : %s'%(itr, endtime-starttime))
                        print('------------------------------------------------')
                        log.close()

                    if pre_loss <= pre_min_loss:
                        #record log
                        log = open(log_location,'a+')
                        sys.stdout = log
                        #save model
                        load_path = dirname + '/'+'0_pred_params.npy'
                        torch.save(model.state_dict(), load_path)

                        #figure number +1
                        jj+=1

                        #update min fit loss
                        pre_min_loss = pre_loss

                        #save Predication figure
                        fig = plt.figure(figsize=(6, 2*args.input_dim),  facecolor='white')
                        print(str(jj) + ' Iter {:04d} | Predication Loss {:.6f}'.format(itr, pre_loss.item()))
                        save_name = str(jj) + '_predication_' + '%.4f'%pre_loss + '.png'
                        pred_data = np.array(pred_data.cpu())
                        visualize(pred_y_true, pred_data, t[-pred_slice+1:], dirname, save_name)

                        #Record time
                        endtime = datetime.datetime.now()
                        print('------------------------------------------------')
                        print('Itr %s Runnig time : %s'%(itr, endtime-starttime))
                        print('------------------------------------------------')
                        log.close()
                        
        endtime = datetime.datetime.now()
        log = open(log_location,'a+')
        sys.stdout = log
        print('------------------------------------------------')
        print('Fit Loss:')
        plot_loss(dirname, Fit_loss_log, name = 'fit')
        print('------------------------------------------------')
        print('Predication Loss:')
        plot_loss(dirname, Pred_loss_log, name = 'pred')
        print('------------------------------------------------')
        print('Total Runnig time : %s'%(endtime-starttime))
        print('------------------------------------------------')
        log.close()