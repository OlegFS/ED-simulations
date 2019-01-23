#!/usr/bin/python
# Event-based Brunel network
from func.ED_sim_ad import *

import multiprocessing
np.random.seed()
random.seed()

if __name__ == "__main__":
    # Set params

    Js =[2.55]# [1.28,1.55,1.5,1.6,1.75,1.9,3.55]
    Gs = [500/500]#[950/50,900/100,700/300,1.0,300/700,200/800,50/950]
    for i,epsilon in enumerate([0.5]):#[0.05,0.1,0.3,0.5,0.7,0.8,0.95]):
        params = {'Vthr': 20,
                  'Vres': 10,
                  'V0':0,
                  'tau_ref': 2.0,
                  'tau_m':40,
                  'd':3.5,
                  'N':1000,
                  'epsilon':epsilon,
                  'p':0.1,
                  'g':Gs[i],
                  'J':Js[i],
                  'eta':0.4,
                  'tau_w':90000.,
                  'a':0.0,
                  'b':0.015}
        g = Gs[i]
        all_params =[]# params.copy()

        for k in [0.8,0.7,0.5,0.3,0.2,0.05]:
            params['g']=g*k
            all_params.append(params.copy())#

        proc_n = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(6)
        print(all_params[0])

        result = pool.map(run,all_params)#[pool.apply_async(run,args = (par)) for par in all_params]
