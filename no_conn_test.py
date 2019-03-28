#!/usr/bin/python
# Event-based adaptive LIF netowrk 
from func.ED_no_conn import *
np.random.seed(12345)
random.seed(12345)
if __name__ == "__main__":
    # Set params
    params = {'Vthr': 20,
              'Vres': 10,
              'V0':0,
              'tau_ref': 2.0,
              'tau_m':40,
              'd':3.5,
              'N':5000,
              'epsilon':0.2,#0.95,
              'p':0.1,
              'g':0,#5/495,
              'J':0.0,#0.94868329805051377,
              'eta':0.0,
              'tau_w':20000,
              'a':0.0,
              'b':2}
    sim_time =10000
    run(params, sim_time)
