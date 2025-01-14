import sys
import os
sys.path.append(os.getcwd())
from  Lorenz96.utility.utilities import *
from Lorenz96.utility.data_utils import odeint_lorenz96
import torch
import datetime

def ODEnet(args):
    def get_batch():
        global s
        s = torch.from_numpy(np.random.choice(np.arange(args.interval*args.for_fit-args.batch_time, dtype=np.int64), args.batch_size, replace=False))
        batch_y0 = fit_y_true[s]  # (M, D)
        batch_t = t[:args.batch_time]  # (T)
        batch_y = torch.stack([fit_y_true[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
        return batch_y0.to(device), batch_t.to(device), batch_y.to(device)
    # device
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    # dataset parameter
    N = torch.tensor(args.input_dim) # Number of variables
    F = torch.tensor(args.F) # Forcing
    y0 = torch.tensor([-1.2061,  0.0617,  1.1632, -1.5008, -1.5944, -0.0187]) # Initial conition
    t, true_y, y0, fit_y_true, predicate_y_true = odeint_lorenz96(N, F, y0, args, device) # Generate dataset

    for i in range(args.run_loop):
        func = ODEBlock(ODEFunc_noise(args.input_dim, args.depth, args.hn, args.activation, wr=args.write_noise, rr=args.read_noise, device=device))
        # Choose Loss function
        loss_fn = choose_loss_fn(args)
        # Choose optimizer
        optimizer = choose_optimizer(args, func) 
        #Define log
        save_name = f'{str(args.depth)}_{str(args.hn)}_{str(args.activation)}_{str(args.interval)}_{str(args.integrate_time)}_{str(args.method)}_{str(args.optimizer)}_wr{str(args.write_noise)}_rr{str(args.read_noise)}'
        dirname = f'{args.path}{args.model_type}_{args.comment}/dim{str(args.input_dim)}_{str(args.loss_fn)}_{save_name}/{str(i)}'
        makedirs(dirname)
        log_location = '%s/0_log.txt'%(dirname)
        log = open(log_location,'a+')
        sys.stdout = log

        starttime = datetime.datetime.now()
        print_log(args, device, func)
        log.close()
        
        # Define some param
        ii = 0
        jj = 0
        fit_min_loss = 500
        Fit_loss_log = []
        pre_min_loss = 500
        Pred_loss_log = []
        for itr in range(1, args.niters + 1):
            #Training
            func.train()
            optimizer.zero_grad()
            batch_y0, batch_t, batch_y = get_batch()
            pred_y = func(batch_y0, batch_t, args.method)
            loss = loss_fn(pred_y, batch_y)
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
                    #Testing Fitting effect
                    #Fit lorenz time series
                    fit_tmax = args.integrate_time*(1-args.for_predication)
                    fit_steps = int(args.interval*(1-args.for_predication))
                    fit_t = torch.linspace(0, fit_tmax, fit_steps).to(device)
                    fit_y0 = fit_y_true[0].to(device)
                    fit_y = func(fit_y0, fit_t, args.method)
                    # loss
                    fit_loss = loss_fn(fit_y, fit_y_true)
                    Fit_loss_log.append(np.float32(fit_loss.cpu()))

                    #Testing Predication effect
                    #Predicate lorenz time series
                    pre_tmax = args.integrate_time*args.for_predication
                    pre_steps = int(args.interval*args.for_predication)
                    pred_t = torch.linspace(0, pre_tmax, pre_steps).to(device)
                    pred_y0 = predicate_y_true[0].to(device)
                    pred_y = func(pred_y0, pred_t, args.method)
                    pre_loss = loss_fn(pred_y, predicate_y_true) 
                    Pred_loss_log.append(np.float32(pre_loss.cpu()))

                    if fit_loss <= fit_min_loss:
                        #record log
                        log = open(log_location,'a+')
                        sys.stdout = log
                        save_model(dirname=dirname, func=func.odefunc, name=f'{ii}_fit_{fit_loss:.4f}') #save model
                        #figure number +1
                        ii+=1
                        #update min fit loss
                        fit_min_loss = fit_loss
                        #save Fitting figure
                        fig = plt.figure(figsize=(10, 2*args.input_dim), facecolor='white')
                        print(f'{ii} Iter {itr:04d} | Predication Loss {fit_loss.item():.6f}')
                        visualize(true_y = fit_y_true, pred_y = fit_y.cpu().numpy(), t = fit_t, 
                                  load_path = dirname, name=f'{ii}_fit_{fit_loss:.4f}')
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
                        save_model(dirname=dirname, func=func.odefunc, name = f'{jj}_pre_{pre_loss:4f}')
                        #figure number +1
                        jj+=1
                        #update min pred loss
                        pre_min_loss = pre_loss
                        #save Predication figure
                        fig = plt.figure(figsize=(6, 2*args.input_dim),  facecolor='white')
                        print(f'{jj} Iter {itr:04d} | Predication Loss {pre_loss.item():.6f}')
                        visualize(true_y=predicate_y_true, pred_y=pred_y.cpu().numpy(), t=pred_t, 
                                  load_path = dirname, name = f'{jj}_pre_{pre_loss:4f}')
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