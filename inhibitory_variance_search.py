# Look for higher variance in network with asynchornous input (only to inh)
# Input only to Excitatory population
# External rate = 0.09615384615384616
from func.ED_sim_ad import *
import multiprocessing
np.random.seed()
random.seed()
if __name__ == "__main__":
        # loop over all inh. perc
    params = {'Vthr': 20,
              'Vres': 10,
              'V0':0,
              'tau_ref': 2.0,
              'tau_m':40,
              'd':3.5,
              'N':5000,
              'epsilon':0.5,
              'p':0.1,
              'g':1.0,
              'J':4.0,#25/np.sqrt(25),#0.94868329805051377,
              'J_ext':1.5,
              'input':'NE+NI',
              'eta':0.0,
              'tau_w':0,
              'a':0.0,
              'b':2}
    all_params = []
    
    Eps =[0.5]#[0.05,0.1,0.2,0.5,0.7,0.8]# [0.3,0.5,0.8]
    Gs =[1.0]# [95/5,9/1,8/2,1.0,3/7,2/8]#[7/3,1.0,2/8]

    #Etas = [0.1,0.2,0.3,0.4]
    Taus = [10000,20000]#[2000,5000,10000]
    Jexts = [1.5,1.8,2.0,3.0,4.0]
    Js = [2.0,3.0,4.0,5.0,6.0]
    for e_index,epsilon in enumerate(Eps):
        for tau_index, tau in enumerate(Taus):
            for j in Js:
                for Jext in Jexts:
                    for g in [1]:
                        params['epsilon']  =  epsilon
                        params['J'] = j
                        params['J_ext'] = Jext
                        params['tau_w'] = tau
                        if g==1:
                            params['g'] = Gs[e_index]
                        else:
                            params['g'] = 0

                        all_params.append(params.copy())#

    proc_n = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(len(all_params))
    print(len(all_params))
    result = pool.map(run,all_params)#[pool.apply_async(run,args = (par)) for par in all_params]

