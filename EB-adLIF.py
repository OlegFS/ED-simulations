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
              'g':5/95,
              'J':(15.5*np.sqrt(5))/np.sqrt(80),#09.94868329805051377,#(3*np.sqrt(25))/np.sqrt(250),#25/np.sqrt(25),#09.94868329805051377,
              'J_ext':1.7,
              'input':'NE+NI',
              'ex_rate':(0.09615384615384616,0.09615384615384616),
              'eta':0.0,
              'tau_w':2000,
              'a':0.0,
              #'w':'E',
              'b':2}
    print(params)
    sim_time =350000
    run(params, sim_time)
