from func.ED_sim_ad import *
import multiprocessing
np.random.seed(12345)
random.seed(12345)
if __name__ == "__main__":
    # Set params
    tau_w = 20000#[20000,40000,60000]:#,40000,60000,80000]:#[60000,80000]:
    b = 2
    Js= [7.0]#[4.0,4.0,4.0,4.0,4.0,6.0,7.0,9.25]
    Gs =[200/800]#[9.0,900/100,800/200,700/300,1.0,300/700,200/800,50/950]
    Es =[0.8]#,0.1,0.2,0.3,0.5,0.7,0.8,0.95]
    for epsilon_index,epsilon in enumerate(Es):
        #for i,epsilon in enumerate([0.05,0.1,0.3,0.5,0.7,0.8,0.95]):
            # loop over all inh. perc
        params = {'Vthr': 20,
                  'Vres': 10,
                  'V0':0,
                  'tau_ref': 2.0,
                  'tau_m':40,
                  'd':3.5,
                  'N':1000,
                  'epsilon':epsilon,
                  'p':0.1,
                  'g':Gs[epsilon_index],
                  'J':Js[epsilon_index],
                  'eta':0.5,
                  'tau_w':tau_w,
                  'a':0.0,
                  'b':b}
        #sim_time=500000
        #run(params,sim_time)#[pool.apply_async(run,args = (par)) for par in all_params]
        all_params = []
        for i,bic in enumerate([0.8,0.7,0.5,0.3,0.2,0.0]):
            params['g']  = Gs[epsilon_index]*bic
            all_params.append(params.copy())#

        proc_n = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(6)
        print(all_params[0])

        result = pool.map(run,all_params)#[pool.apply_async(run,args = (par)) for par in all_params]

