#!/usr/bin/python
# Event-based Brunel network
import numpy as np
import hashlib
from numba import jit
from sortedcontainers import SortedDict, SortedList
na = np.array
import pickle
import h5py as h5py
import json
import random
import collections
import multiprocessing

np.random.seed()
random.seed()
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# generate connectivity 
def fixed_in_conn(NE,NI,p):
    """Helper funvtion to generate fixed indegree"""
    conn = []
    kE = int(p*NE)
    kI  = int(p*NI)
    for n in range(NE+NI):
        # Set fixed in-degree
        projections = np.hstack([np.random.randint(0,NE,kE),np.random.randint(NE,NE+NI,kI)])
        for pp in projections:
            conn.append((pp,n)) # out-in
    conn_sorted = sorted (conn)
    conn_dict = {}
    conn_sorted = na(conn_sorted)
    for n in range(NE+NI):
        conn_sorted[conn_sorted[:,0]==n]
        conn_dict.update({n:conn_sorted[conn_sorted[:,0]==n][:,1]})
    return conn_dict

@jit
def V_update(t,t0,I,tau):
    """t0 - time of the event
     t - current time """
    return I*np.exp(-(t-t0)/tau)
# Intializtion 
#@jit

def run(params,sim_time=200000):
    """Event-driven simualtion of the brunel network i
        TODO: class? """
    # Set params
    Vthr = params['Vthr']
    Vres = params['Vres']
    V0 = params['V0'] # 
    tau_ref = params['tau_ref'] #ms
    tau_m = params['tau_m']#ms
    d = params['d'] 
    N = params['N']
    epsilon = params['epsilon']#inhibitory fraction
    NI = int(N*epsilon)
    NE = N-NI # NE
    p =params['p']# connection probability
    Nindetity = np.ones(N)
    g = params['g']
    Nindetity[NE:]= -params['g'] # set g
    Jext =params['J'] # Jext = J by default
    J= params['J']
    J_inh = g*J
    b = params['b'] #adaptation increment
    a = params['a'] # subthreshold adaptation
    tau_w = params['tau_w']#timescale of adaptation
    k = int(N*p)
    kI = int(NI*p)
    kE = int(NE*p)
    eta = params['eta'] #input in nu_ext/nu_thr
    # Process the parameters
    nu_th = Vthr / (Jext*kE*tau_m)
    nu_ex = eta * nu_th
    p_rate = nu_ex
    # mean of exponential dist
    expR=(p_rate*kE)# it's Beta in numpy implementation
    print(kE)
    sim_time = sim_time#ms
    # init conn 
    try:
       # connectivity generation is a weak stop of this implemetnation at the moment
       conn_dict = load_obj('conn/%s_fixed_in.pkl'%N)
    except:
        print('Generating Connectivity')
        conn_dict = fixed_in_conn(NE,NI,p)
        save_obj(conn_dict, 'conn/%s_fixed_in.pkl'%N)
    #init volatege
    Vm =np.random.normal(0,0.1,N)
    W = np.random.normal(0,0.1,N)
    wt = np.zeros([N])#time of previous adaptation event
    spikes0 = Vm>Vthr
    #Q - is the main queue of events
    Q = SortedList()#np.zeros(N)
    et = np.zeros(N) #time of the previous event for each neuron 
    # Init previous spike vector 
    previous_st = np.zeros(N)*-1000

    st = [] # list of spikes
    gid = []# list of unit id's #consider storing to memory for big simualtions
    print(nu_ex,expR)
    # Initialize 
    np.random.seed()
    for n in range(N):
    #         np.random.seed() # if conn is generated on the go
        event_t = random.expovariate(expR)
        Q.add((event_t, n,'I'))
    # Main loop 
    t=0
    print(NE,NI)
    while t<sim_time:
        t,n,event = Q[0]
        Q.pop(index=0)

        if (t-previous_st[n])>tau_ref: # Refactory period
            if event=='I':
                # Evolve voltage from previos event and V to current
                #update adaptation
                V_= V_update(t,et[n],Vm[n],tau_m)
                W[n] = a*V_+V_update(t,et[n],W[n],tau_w)
                #update voltage
                Vm[n] =V_+Jext-W[n]
              #  if n ==0:
              #      print(Vm[n])
                et[n] = t
            if event =='S':
                # record
                st.append(t)
                gid.append(n)
                previous_st[n]=t

                V_= V_update(t,et[n],Vm[n],tau_m)
                W[n] = a*V_+V_update(t,et[n],W[n],tau_w)+b
                #update voltage
                Vm[n] = Vres#-W[n]# set to reset V
                et[n] = t
                # Send spikes
                # Code for generating conn on the go (fast, but so far only fixed out-degree is possible)
                #np.random.seed(N+n)
                #out_deg = np.random.randint(0,NE,kE) # Exciatory out
                #out_deg = np.hstack([out_deg, np.random.randint(NE+1,NE+NI,kI)]) # Inhibitory out
                out_deg = conn_dict.get(n)
                sign = Nindetity[n]

                for n_out in out_deg:
                    # Input events
                    if sign>0:
                        Q.add((t+d,n_out,'P'))#EPSP
                    else:
                        Q.add((t+d,n_out,'N'))#IPSP

            if event =='N':
                V_= V_update(t,et[n],Vm[n],tau_m)
                W[n] =  a*V_+V_update(t,et[n],W[n],tau_w)
                #update voltage
                Vm[n] =V_-(g*J)-W[n]
                et[n] = t

            if event == 'P':
                V_= V_update(t,et[n],Vm[n],tau_m)
                W[n] = a*V_+V_update(t,et[n],W[n],tau_w)
                #update voltage
                Vm[n] =V_+J-W[n]
                et[n] = t
            if Vm[n]>=Vthr:
                #Handle spontaneous spikes
                Q.add((t,n,'S'))
        if event=='I':
                #next input 
            #np.random.seed() # if conn is generated on the go
            #make sure to insert noise even if event is ignored ! 
            if n<800:
                event_t = random.expovariate(expR)
                Q.add((t+event_t,n,'I'))

    params_json = json.dumps(params)
    hash_name = hashlib.sha256(params_json.encode('utf-8')).hexdigest()
    print(hash_name)
    print(params)
    print(sim_time)
    np.save('sim/%s'%hash_name,[st,gid,params])


    with open("sim/log.txt", "a") as myfile:
        myfile.write(params_json)

    #with h5py.File('sim/'+str(hash_name),'w',libver='latest') as f:
    #    f['st']  =st
    #    f['uid']  =gid
    #    f['params'] =params_json
    print(len(st)/N/(sim_time/1000))
    return st, gid


if __name__ == "__main__":
    # Set params
    params = {'Vthr': 20,
              'Vres': 10,
              'V0':0,
              'tau_ref': 2.0,
              'tau_m':40,
              'd':3.5,
              'N':1000,
              'epsilon':0.2,
              'p':0.1,
              'g':800/200,
              'J':1.5,
              'eta':0.5,
              'tau_w':25000.,
              'a':0.0,
              'b':0.01}
    g = 800/200
    all_params =[]# params.copy()

    for k in [0.8,0.7,0.5,0.3,0.2]:
        params['g']=g*k
        all_params.append(params.copy())#

    proc_n = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(5)
    print(all_params[0])

    result = pool.map(run,all_params)#[pool.apply_async(run,args = (par)) for par in all_params]
