import os
import sys
sys.path.append(os.getcwd())
from Lorenz96.utility.utilities import get_current_time
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Different size on ODEnet
# # # create path
year_mon_mday, hour_min_sec = get_current_time()
path = "Lorenz96/Model_size_ODEnet__%s_%s/"%(year_mon_mday, hour_min_sec)
# run for loop
for depth in [1]:
      for hn in [512]:
            # commnet = variable
            os.system(f'python Lorenz96/Hyperparameters.py --path {path} --comment {str(depth)+"_"+str(hn)}  --depth {depth} --hn {hn} --niters {3000}')

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Different noise on ODEnet
# #create path
# year_mon_mday, hour_min_sec = get_current_time()
# path = "Different_noise_on_ODEnet_%s_%s/"%(year_mon_mday, hour_min_sec)
# # path = "/home/hegan/Lorenz96_odenet/Fig4/Different_noise_on_ODEnet_2023711_1917/"
# #run for loop
# for wr in np.linspace(0, 0.03, 4):
#       for rr in np.linspace(0, 0.03, 4):
#             os.system(f'python Hyperparameters.py --path {path} --comment {str(wr)+"_"+str(rr)} --loss_fn {"DTW"} --write_noise {wr} --read_noise {rr} --niters {2000} --gpu {1} --run_loop {10}')


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Different noise on RNN/GRU/LSTM
#create path
# year_mon_mday, hour_min_sec = get_current_time()
# path = "Different_noise_on_other_NN_%s_%s/"%(year_mon_mday, hour_min_sec)

# #run for loop
# for model_type in ['RNN', 'GRU', 'LSTM']:
#       for wr in np.linspace(0, 0.03, 4):
#             for rr in np.linspace(0, 0.03, 4):
#                   os.system(f'python Hyperparameters.py --path {path} --comment {str(wr)+"_"+str(rr)} --model_type {model_type} --loss_fn {"DTW"} --write_noise {wr} --read_noise {rr}  --niters {2000} --batch_size {1} --gpu {1} --run_loop {10}')