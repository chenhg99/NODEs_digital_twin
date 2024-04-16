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
    # batch_size=1, discrete model
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
        # func = Net(model_type = args.model_type, input_size=args.input_dim, hn=args.hn, depth=args.depth, batch_first=True, wr=0, rr=0, device=device).to(device)
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
        hidden_prev_train = torch.zeros(args.depth, args.batch_size, args.hn).to(device)       #初始化记忆单元h0[args.batch_size,num_layer,hidden_len]
        c_train = torch.zeros(args.depth, args.batch_size, args.hn).to(device)          #初始化记忆单元h0[args.batch_size,num_layer,hidden_len]
        for itr in range(args.niters):
            model.train()
            optimizer.zero_grad()
            #Fit lorenz time series
            if args.model_type == "LSTM":
                output, (hidden_prev_train, c_train) = model(x, (hidden_prev_train, c_train))
                c_train = c_train.detach()
            else:
                output, hidden_prev_train = model(x, hidden_prev_train)       #喂入模型得到输出
            
            hidden_prev_train = hidden_prev_train.detach()
            loss = loss_fn(output, y)        #计算MSE损失
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
                    input = x[:, 0, :]             #取seq_len里面第0号数据
                    input = input.view(1, 1, feature_len)   #input：[1,1,1]
                    #initial sorted data
                    fit_data = []
                    fit_data.append(input.cpu().detach().numpy().ravel())
                    #initial hidden state
                    hidden_prev = torch.zeros(args.depth, args.batch_size, args.hn).to(device)       #初始化记忆单元h0[args.batch_size,num_layer,hidden_len]
                    c = torch.zeros(args.depth, args.batch_size, args.hn).to(device)          #初始化记忆单元h0[args.batch_size,num_layer,hidden_len]
                    #0-1800 used for fitting
                    for _ in range(fit_slice - 1):             #迭代seq_len次
                            if args.model_type == "LSTM":
                                pred, (hidden_prev, c) = model(input, (hidden_prev, c))
                                c = c.detach()
                            else:
                                pred, hidden_prev = model(input, hidden_prev)
                            input = pred           #预测出的(下一个点的)序列pred当成输入(或者直接写成input, hidden_prev = model(input, hidden_prev))
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
                    for _ in range(pred_slice - 2):             #迭代seq_len次
                        if args.model_type == "LSTM":
                            pred, (hidden_prev, c) = model(input, (hidden_prev, c))
                            c = c.detach()
                        else:   
                            pred, hidden_prev = model(input, hidden_prev)
                            input = pred           #预测出的(下一个点的)序列pred当成输入(或者直接写成input, hidden_prev = model(input, hidden_prev))
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
                        # save_model(dirname, func, name = '0_fit')
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
        # print(Fit_loss_log)
        plot_loss(dirname, Fit_loss_log, name = 'fit')
        print('------------------------------------------------')
        print('Predication Loss:')
        # print(Pred_loss_log)
        plot_loss(dirname, Pred_loss_log, name = 'pred')
        print('------------------------------------------------')
        print('Total Runnig time : %s'%(endtime-starttime))
        print('------------------------------------------------')
        log.close()

            # # Inference process including fitting and predicating
            # #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # for wr in [0]:
            #     for rr in [0]:
            #         #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #         # load model with different noise level
            #         func_inference = Net(model_type = args.model_type, input_size=args.input_dim, hn=args.hn, depth=args.depth, batch_first=True, wr=0, rr=0).to(device)
            #         func_inference.load_state_dict(torch.load(load_path))
            #         func_inference.to(device)
            #         #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #         #0-1800 used for fitting
            #         input = x[:, 0, :]             #取seq_len里面第0号数据
            #         input = input.view(1, 1, feature_len)   #input：[1,1,1]

            #         fit_data = []
            #         fit_data.append(input.cpu().detach().numpy().ravel()) 

            #         hidden_prev = torch.zeros(args.depth, args.batch_size, args.hn).to(device)       #初始化记忆单元h0[args.batch_size,num_layer,hidden_len]
            #         c = torch.zeros(args.depth, args.batch_size, args.hn).to(device)          #初始化记忆单元h0[args.batch_size,num_layer,hidden_len]
            #         for _ in range(fit_slice - 1):             #迭代seq_len次
            #                 if args.model_type == "LSTM":
            #                     pred, (hidden_prev, c) = func_inference(input, (hidden_prev, c))
            #                     c = c.detach()
            #                 else:
            #                     pred, hidden_prev = func_inference(input, hidden_prev)
            #                 input = pred           #预测出的(下一个点的)序列pred当成输入(或者直接写成input, hidden_prev = model(input, hidden_prev))
            #                 fit_data.append(pred.cpu().detach().numpy().ravel())
                    
            #         fit_data = np.array(fit_data)
            #         fit_data = torch.from_numpy(fit_data).to(device)
            #         fit_loss = loss_fn(fit_data, fit_y_true)
            #         fit_data = np.array(fit_data.cpu())
            #         save_name = '0_fit_data_wr%s_rr%s_loss_%.4f.png'%(wr, rr, fit_loss)
            #         visualize_nn(fit_y_true, fit_data, t[:fit_slice], dirname, save_name)
            #         #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #         #1800 - 2000 used for predcating
            #         input = torch.tensor(pred_y_true[0]).float().view(1, feature_len)             #取seq_len里面第0号数据
            #         input = input.view(1, 1, feature_len)   #input：[1,1,1]

            #         pred_data = []
            #         pred_data.append(input.cpu().detach().numpy().ravel())

            #         hidden_prev = torch.zeros(args.depth, args.batch_size, args.hn).to(device)       #初始化记忆单元h0[args.batch_size,num_layer,hidden_len]
            #         c = torch.zeros(args.depth, args.batch_size, args.hn).to(device)          #初始化记忆单元h0[args.batch_size,num_layer,hidden_len]
            #         for _ in range(pred_slice - 1):             #迭代seq_len次
            #                 if args.model_type == "LSTM":
            #                     pred, (hidden_prev, c) = func_inference(input, (hidden_prev, c))
            #                     c = c.detach()
            #                 else:
            #                     pred, hidden_prev = func_inference(input, hidden_prev)
            #                 input = pred           #预测出的(下一个点的)序列pred当成输入(或者直接写成input, hidden_prev = model(input, hidden_prev))
            #                 pred_data.append(pred.cpu().detach().numpy().ravel())
            #         pred_data = np.array(pred_data)
            #         pred_data = torch.from_numpy(pred_data).to(device)
            #         pre_loss = loss_fn(pred_data, pred_y_true)
            #         pred_data = np.array(pred_data.cpu())
            #         save_name = '0_pred_data_wr%s_rr%s_loss_%.4f.png'%(wr, rr, pre_loss)
            #         visualize_nn(pred_y_true, pred_data, t[:pred_slice], dirname, save_name)

            # #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # for wr in [0]:
            #     for rr in [0]:
            #         #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #         # load model with different noise level
            #         func_inference = Net(model_type = model_type, input_size=6, hn=1024, depth=2, batch_first=True, wr=wr, rr=rr).to(device)
            #         func_inference.load_state_dict(torch.load(load_path))
            #         func_inference.to(device)
            #         #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #         #0-1800 used for fitting
            #         #测试过程
            #         input = torch.tensor(pred_y_true[0]).float().view(1, feature_len)             #取seq_len里面第0号数据
            #         input = input.view(1, 1, feature_len)   #input：[1,1,1]

            #         pred_data = []
            #         pred_data.append(input.cpu().detach().numpy().ravel())

            #         hidden_prev = torch.zeros(depth, args.batch_size, hn).to(device)       #初始化记忆单元h0[args.batch_size,num_layer,hidden_len]
            #         c = torch.zeros(depth, args.batch_size, hn).to(device)          #初始化记忆单元h0[args.batch_size,num_layer,hidden_len]
            #         for _ in range(args.for_fit - 1):             #迭代seq_len次
            #                 if model_type == "LSTM":
            #                     pred, (hidden_prev, c) = func_inference(input, (hidden_prev, c))
            #                     c = c.detach()
            #                 else:
            #                     pred, hidden_prev = func_inference(input, hidden_prev)
            #                 input = pred           #预测出的(下一个点的)序列pred当成输入(或者直接写成input, hidden_prev = model(input, hidden_prev))
            #                 pred_data.append(pred.cpu().detach().numpy().ravel())
            #         pred_data = np.array(pred_data)
            #         save_name = '0_pred_data_wr%s_rr%s.png'%(wr, rr)
            #         visualize_nn(pred_y_true, pred_data, t[:args.for_fit], dirname, save_name)








            # hidden_prev = torch.zeros(depth, args.batch_size, hn)
            # #1800 - 2000 used for predication
            # #测试过程
            # input = torch.tensor(pred_y_true[0]).float().view(1, feature_len)             #取seq_len里面第0号数据
            # input = input.view(1, 1, feature_len)   #input：[1,1,1]

            # pred_data = []
            # pred_data.append(input.detach().numpy().ravel())
            # for _ in range(args.for_fit - 1):             #迭代seq_len次
            #     pred, hidden_prev = func_inference(input, hidden_prev)
            #     input = pred           #预测出的(下一个点的)序列pred当成输入(或者直接写成input, hidden_prev = model(input, hidden_prev))
            #     pred_data.append(pred.detach().numpy().ravel())
            # pred_data = np.array(pred_data)
            # np.save('/home/hegan/Lorenz96_odenet/RNN'+'/'+'pred_data.npy', pred_data)

            # fig = plt.figure(figsize=(6, 6), dpi=100)
            # plt.title('Predicated curve') 
            # for i in range(feature_len):
            #     plt.subplot(int(feature_len), 1, i+1)
            #     plt.plot(t[:args.for_fit], pred_y_true[:,i], color='b')
            #     plt.scatter(t[:args.for_fit], pred_data[:,i], s=4, c='darkred')   #y的预测值
            # fig.tight_layout(pad=1, h_pad=0.1)
            # plt.show()