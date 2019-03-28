#Search for static model with parameters similar to EI model
from func.ED_sim_ad import *
import multiprocessing
#np.random.seed()
#random.seed()
np.random.seed(12345)
random.seed(12345)
if __name__ == "__main__":
        # loop over all inh. perc
    params = {'Vthr': 20,
              'Vres': 10,
              'V0':0,
              'tau_ref': 2.0,
              'tau_m':40,
              'd':3.5,
              'N':1000,
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
              #'w':'E',
              'b':0}
    all_params = []
    Eps =[0.5]#[0.5,0.75,0.8,0.95,0.99]#0.5,0.8,0.95]#[0.5,0.8,0.95]#[0.3,0.8,0.95]#[0.3,0.5,0.7,0.8,0.95,0.99]# [0.3,0.5,0.8]
    #Etas = [0.1,0.2,0.3,0.4]
    Taus = [20000]#[5000,20000]#[2000,5000,10000]
    #Jexts = [2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,15.0]
    Js =[7,8]#np.arange(15,25)#[16.,17.,18.]#[25.0,15.5,10.0]#[1.0,2.0,2.5,2.8,3.0,4.0]
    ex_rates =[0.0007]#,0.0000002]#[0.0016025641025641027]#[0.09615384615384616/60]#[0.09615384615384616/90,0.09615384615384616/100,0.09615384615384616/70,0.09615384615384616/60]#0.09615384615384616/80]#,0.09615384615384616/2]#[0.03,0.04,0.05,0.06,0.07,0.08,0.09]
    Jexts =[10.0]# [15.,16.,17.,18.0,19.0]#[2.0,3.0]#[3.0]
    N = params['N']
    for e_index,epsilon in enumerate(Eps):
        for tau_index, tau in enumerate(Taus):
            for j in Js:
                for ex_rate in ex_rates:
                    for jex in Jexts:
                        for g in [1.]:
                           # j_=#(1.79*np.sqrt(250))/np.sqrt((params['N']-(epsilon*params['N']))*0.1)
                            params['epsilon']  =  epsilon
                            params['J']=(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1)#(j*5)/((N-(epsilon*N))*0.1)
                            print(params['J'])
                            #(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1)
                            params['J_ext'] =jex#(jex*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1)
                            params['ex_rate'] =(ex_rate/5)*((N-(epsilon*N))*0.1)# (ex_rate/80)*((N-(0.8*N))*0.1)
                            params['tau_w'] = tau
                            params['g'] =(((N-(epsilon*N))/(N*epsilon)))*g
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
