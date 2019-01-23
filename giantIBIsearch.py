from func.ED_sim_ad import *
import multiprocessing
np.random.seed()
random.seed()
if __name__ == "__main__":
    # Set params
    #Js = [1.28,1.55,1.5,1.55,1.75,1.9,3.65]
    for eta in [0.1,0.35,0.4,0.45,0.5]:
        for tau_w in [25000.,50000,80000,90000,95000,100000,150000]
            for j in [1,1.5,2,2.5,3.0,3.5,4.0,5.]:
                Gs = [900/100,800/200,700/300,1.0,300/700,200/800,50/950]
                #for i,epsilon in enumerate([0.05,0.1,0.3,0.5,0.7,0.8,0.95]):
                Es =[0.1,0.2,0.5,0.8,0.95]# [0.05,0.1,0.3,0.5,0.7,0.8,0.95]
                    # loop over all inh. perc
                params = {'Vthr': 20,
                          'Vres': 10,
                          'V0':0,
                          'tau_ref': 2.0,
                          'tau_m':40,
                          'd':3.5,
                          'N':1000,
                          'epsilon':0,
                          'p':0.1,
                          'g':0.0,#Gs[i],
                          'J':j,
                          'eta':eta,
                          'tau_w':tau_w,
                          'a':0.0,
                          'b':0.01}
                #sim_time=500000
                #run(params,sim_time)#[pool.apply_async(run,args = (par)) for par in all_params]
                all_params = []
                for i,k in enumerate(Es):
                    params['epsilon']=k
                    params['g']  = Gs[i]
                    all_params.append(params.copy())#
                proc_n = multiprocessing.cpu_count()
                pool = multiprocessing.Pool(5)
                print(all_params[0])

                result = pool.map(run,all_params)#[pool.apply_async(run,args = (par)) for par in all_params]

