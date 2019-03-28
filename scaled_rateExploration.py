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
              'epsilon':0.0,
              'p':0.1,
              'g':0.0,#Gs[i],
              'J':0,
              'eta':999.0, #2.403846153846154/kE
              'tau_w':20000,
              'a':0.0,
              'b':2}

    all_params = []
    Js = [3.0]#,3.1,3.2,3.5]
    Eps =[0.5,0.7,0.95]# [0.3,0.5,0.8]
    Gs = [1.0,3/7,5/95]#[7/3,1.0,2/8]
    #Etas = [0.1,0.2,0.3,0.4]
    #Taus = [1000,3000]

    for j_index,j in enumerate(Js):
        for e_index,epsilon in enumerate(Eps):
            for g in [1]:
                params['epsilon']  =  epsilon
                params['J'] = j
                if g==1:
                    params['g'] = Gs[e_index]
                else:
                    params['g'] = 0

                all_params.append(params.copy())#

    proc_n = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(len(all_params))
    print(len(all_params))

    result = pool.map(run,all_params)#[pool.apply_async(run,args = (par)) for par in all_params]

