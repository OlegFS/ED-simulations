# Look for the best fit in the network
# Input only to EI
# Varibility is given by E/I 
# The first rought gues for the EI model with N=5001
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
              'N':500,
              'epsilon':0.0,
              'p':0.1,
              'g':0.0,
              'J':0.,#25/np.sqrt(25),#0.94868329805051377,
              'J_ext':0.,
              'input':'NE+NI',
              'ex_rate':0,
              'eta':0.0,
              'tau_w':0,
              'a':0.0,
              'conn':'rand',
              #'leader':1,
              'b':2}
    all_params = []
    Eps=[0.5]#,0.1,0.2,0.3,0.5,0.7,0.8,0.93]# [0.3,0.5,0.8]
    Gs =[1.0]#,9.0,4.0,7/3,1.0,3/7,0.25,7/93]#[8/2,1.0,3/7,2/8,5/95]#[7/3,1.0,2/8]
    #Etas = [0.1,0.2,0.3,0.4]
    Taus = [10000]#[5000,20000]#[2000,5000,10000]
    Js =[13.0]#[8.5,9.0,10.]# [12.,13.,15.,16.,17.,18.,19.,22.0]#[25.0,15.5,10.0]#[1.0,2.0,2.5,2.8,3.0,4.0]
    ex_rates =[0.09615384615384616/5]#/5]#[0.09615384615384616/10,0.09615384615384616/5]
    #[0.09615384615384616,0.09615384615384616/20,0.09615384615384616/40, 0.09615384615384616/80, 0.09615384615384616/100]#,0.09615384615384616/2]#[0.03,0.04,0.05,0.06,0.07,0.08,0.09]
    Jexts = [0.5]#[2.0,3.0]#[3.0]
    N = params['N']
    for e_index,epsilon in enumerate(Eps):
        for tau_index, tau in enumerate(Taus):
            for j in Js:
                for ex_rate in ex_rates:
                    for jex in Jexts:
                        for g in [1,0]:
                            j_=j#(1.79*np.sqrt(250))/np.sqrt((params['N']-(epsilon*params['N']))*0.1)
                            params['epsilon']  =  epsilon
                            params['J']=(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1)
                            params['J_ext'] = jex
                            params['ex_rate'] = 0.2#0.2#(ex_rate/80)*((N-(epsilon*N))*0.1)
                            params['tau_w'] = tau
                            if g==1:
                                params['g'] = Gs[e_index]
                            else:
                                params['g'] = Gs[e_index]*g
                            all_params.append(params.copy())#
    proc_n = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(len(all_params))
    print(len(all_params))
    p = pool.map_async(run,all_params)
    #res = [pool.apply_async(run,args = (par)) for par in all_params]
    #output = [pool.get() for p in res]
    res = p.get()
    print(res)
    pool.close()
