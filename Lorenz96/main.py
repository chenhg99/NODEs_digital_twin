import os
import sys
sys.path.append(os.getcwd())
from Lorenz96.utility.utilities import get_current_time
year_mon_mday, hour_min_sec = get_current_time()
path = "Lorenz96/Model_size_ODEnet__%s_%s/"%(year_mon_mday, hour_min_sec)
for depth in [1]:
      for hn in [512]:
            os.system(f'python Lorenz96/Hyperparameters.py --path {path} --comment {str(depth)+"_"+str(hn)}  --depth {depth} --hn {hn} --niters {3000}')