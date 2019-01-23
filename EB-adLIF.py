#!/usr/bin/python
# Event-based adaptive LIF netowrk 
from func.ED_sim_ad import *
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
              'N':1000,
              'epsilon':0.95,
              'p':0.1,
              'g':1.0,#50/950,#200/800,#50/950,
              'J':16.21,
              'eta':0.5,
              'tau_w':20000,
              'a':0.0,
              'b':2}

    sim_time =500000
    run(params, sim_time)
