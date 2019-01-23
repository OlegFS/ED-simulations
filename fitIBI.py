from func.ED_sim_ad import *
import multiprocessing
np.random.seed(12345)
random.seed(12345)
if __name__ == "__main__":
    # Set params
    for b in [2]:#[0.35,0.4,0.5]:
        for tau_w in [20000]:#[20000,40000,60000]:#,40000,60000,80000]:#[60000,80000]:
            for j in [7.]:#[2.5,3.0,3.5,4.]:#[1.5,2.0,2.5,2.75,3.0,3.25,3.5]:
                #Js =[2.9,3.2,3.3,3.8,5.0,8.6]#[2.9,3.2,3.3,5.0,8.0]#[1.28,1.55,1.5,1.55,1.75,1.9,3.65]
                Gs =[200/800,50/950] 
                #for i,epsilon in enumerate([0.05,0.1,0.3,0.5,0.7,0.8,0.95]):
                Es =[0.8,0.95]
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
                          'eta':0.5,
                          'tau_w':tau_w,
                          'a':0.0,
                          'b':b}
                #sim_time=500000
                #run(params,sim_time)#[pool.apply_async(run,args = (par)) for par in all_params]
                all_params = []
                for i,k in enumerate(Es):
                    params['epsilon']=k
                    params['g']  = Gs[i]
                   # params['J'] = Js[i]
                    all_params.append(params.copy())#

                proc_n = multiprocessing.cpu_count()
                pool = multiprocessing.Pool(len(Gs))
                print(all_params[0])

                result = pool.map(run,all_params)#[pool.apply_async(run,args = (par)) for par in all_params]

