import numpy as np
import hashlib
from numba import jit
from sortedcontainers import SortedDict, SortedList
na = np.array
import pickle
import h5py as h5py
import json
import random
np.random.seed(1234)
random.seed(1234)
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# generate connectivity 
def fixed_in_conn(NE,NI,p):
    """Helper funvtion to generate fixed indegreei
     Not excluding the double and self-connections"""
    conn = []
    kE = int(p*NE) # excitatory degrees
    kI  =int(p*NI) # inhibitory degrees 
    for n in range(NE+NI):
        # Set fixed in-degree
        projections = np.hstack([np.random.randint(0,NE,kE),np.random.randint(NE,NE+NI,kI)])
        # TODO: change to permutation
        for pp in projections:
            conn.append((pp,n)) # out-in
    conn_sorted = sorted(conn) # TODO: delete 
    conn_dict = {}
    conn_sorted = na(conn_sorted)
    for n in range(NE+NI):
        conn_sorted[conn_sorted[:,0]==n]
        conn_dict.update({n:conn_sorted[conn_sorted[:,0]==n][:,1]})
    return conn_dict

# generate connectivity 
def random_conn(NE,NI,p):
    """Helper funvtion to generate fixed indegreei
     Not excluding the double and self-connections"""
    conn = []
    kE = int(p*NE) # excitatory degrees
    kI  =int(p*NI) # inhibitory degrees 
    for n in range(NE+NI):
        ke = np.int(np.abs(np.random.normal(kE,kE/3)))
        ki = np.int(np.abs(np.random.normal(kI,kI/3)))
        # Set fixed in-degree
        projections =np.hstack([np.random.randint(0,NE,ke),np.random.randint(NE,NE+NI,ki)])
        # TODO: change to permutation
        for pp in projections:
            conn.append((pp,n)) # out-in
    conn_sorted = sorted(conn) # TODO: delete 
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

def run(params,sim_time=200000):
    """Event-driven simualtion of adLIF network
    Args:
            [1] params (dict): network parameters
                kwrds:
                Vthr - firing threshold 
                Vres
                V0 TODO: Make sure that V0 is a part of the equation
                tau_ref
                tau_m
                d - delay
                N - Number of neurons
                epsilon - inhibitory fraction
                p - connection probability
                J - coupling strength
                J_ext - strength of external input [not ther in eaerly version]
                g - relative inhibitory coupling
                b - adaptation increment
                a - subthreshold adaptation
                tau_w - adaptation timeconstant
            [2] sim_tim (str): Simulation time
    Returns:
            ts,gid (list,list): spike times, unit ids
        """

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
    J= params['J']
    J_inh = -g*J
    b = params['b'] #adaptation increment
    a = params['a'] # subthreshold adaptation
    tau_w = params['tau_w']#timescale of adaptation
    # Parse the Jext
    if 'leader' in params.keys():
        Jext = np.abs(np.random.normal(Jext,0.3,N))
        Jext[np.random.randint(0,NE,10)]=5
    else:
        Jext = np.zeros(N)
        Jext[:] = params['J_ext']#
    #conn_type =params['conn']
    try:
        #catch the varaibility in params
        if params['w']=='E':
            print('only ex adaptation')
            # set adaptation only for exciatatory population
            Nw  =NE
        else:
            Nw = N
    except:
        Nw= N
    if params['input']=='NE':
        # set the input to N neurons
        N_input = NE
    else:
        print('E and I input')
        N_input = NE+NI
    k = int(N*p) # degrees
    kI = int(NI*p) # inhibitory indegrees
    kE = int(NE*p) # ex indegrees
    eta = params['eta'] #input in nu_ext/nu_thr
    # Process the parameters
    expR =np.zeros(N_input)
    nu_th = 0
    nu_ex = 0
    p_rate = 0
    if params['ex_rate']=='eta':
        nu_th = Vthr / (Jext*kE*tau_m) # Brunel 2000
        nu_ex = eta * nu_th
        p_rate = nu_ex
        expR[:] = p_rate*kE
    elif type(params['ex_rate'])==tuple:
        #managing inhomogeneous input rate to E and I
        # works only if input is set to 'E+I'
        expR[:NE]=params['ex_rate'][0]
        expR[NE:] = params['ex_rate'][1]
    else:
        # set input rate explicitry
        expR[:]=params['ex_rate']#Good rate = 0.09615384615384616#0.4807692307692308/np.sqrt(kE)
    #2.403846153846154/kE#0.09615384615384616#(p_rate*kE)# it's Beta in numpy implementation
    # init conn 
    try:
        conn_type = params['conn']
    except:
        conn_type = 'fixed_in'

    try:
       # connectivity generation is a weak stop of this implemetnation at the moment
       #conn_dict = load_obj('conn/%s_of_%s_fixed_in_p=%s.pkl'%(NI,N,p))
       conn_dict = load_obj('conn/%s_of_%s_%s_p=%s.pkl'%(NI,N,conn_type,p))
    except:
        print('Generating Connectivity')
        #conn_dict = fixed_in_conn(NE,NI,p)
        if 'rand':
            conn_dict = random_conn(NE,NI,p)
        else:
            conn_dict = fixed_in_conn(NE,NI,p)

        #save_obj(conn_dict, 'conn/%s_of_%s_fixed_in_p=%s.pkl'%(NI,N,p))
        save_obj(conn_dict,'conn/%s_of_%s_%s_p=%s.pkl'%(NI,N,conn_type,p))
    #init volatege
    Vm =np.random.normal(17,0.1,N)
    W =np.abs(np.random.normal(10,3,N))
    Vm[:]=0.
    if b==0:
        W[:]=0
    W[Nw:] = 0
    wt = np.zeros([N])#time of previous adaptation event #Kill? 
    spikes0 = Vm>Vthr
    #Q - is the main queue of events
    Q = SortedList()#np.zeros(N)
    et = np.zeros(N) #time of the previous event for each neuron 
    # Init previous spike vector 
    previous_st = np.zeros(N)*-1000

    st = [] # list of spikes
    gid= []# list of unit id's #consider storing to memory for big simualtions
    V_ = 0
    print(J_inh)
    print('Ext rate:',nu_ex,params['ex_rate'])
    # Initialize 
    np.random.seed()
    for n in range(N_input):
    #         np.random.seed() # if conn is generated on the go
        event_t = random.expovariate(expR[n])
        Q.add((event_t, n,'I'))
    # Main loop 
    t=0
    print(NE,NI)
    while t<sim_time:
        t,n,event = Q[0]
        Q.pop(index=0)
        if (t-previous_st[n])>tau_ref:#Refactory period
            if event=='I':
                # Evolve voltage from previos event and V to current
                #update adaptation
                if n<Nw:
                    W[n] = a*V_+V_update(t,et[n],W[n],tau_w)
                V_= V_update(t,et[n],Vm[n]+W[n],tau_m)
                #update voltage
                Vm[n] =V_ + Jext[n] -W[n]
              #  if n ==0:
              #      print(Vm[n])
                et[n] = t
            if event =='S':
                # record
                st.append(t)
                gid.append(n)
                previous_st[n]=t
                if n<Nw:
                    W[n] = a*V_+V_update(t,et[n],W[n],tau_w)+b
                Vm[n] = Vres-W[n]
                #V_= V_update(t,et[n],Vm[n]-W[n],tau_m)
                #update voltage
                # set to reset V
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
                 if n<Nw:
                     W[n] = a*V_+V_update(t,et[n],W[n],tau_w)
                 V_= V_update(t,et[n],Vm[n]+W[n],tau_m)
 #               update voltage
                 Vm[n] =V_
                 Vm[n]+=J_inh
                 Vm[n]-=W[n]
                 et[n] = t

            if event == 'P':
                if n<Nw:
                    W[n] = a*V_+V_update(t,et[n],W[n],tau_w)
                V_= V_update(t,et[n],Vm[n]+W[n],tau_m)
                #update voltage
                Vm[n] =V_
                Vm[n]+=J
                Vm[n]-=W[n]
                et[n] = t

            if t!=Q[0][-3] or n!=Q[0][-2]:
                if Vm[n] >= Vthr:
                #Handle spikes only if there are no further events at this time
                #for this neuron
                #Handle spontaneous spikes
                    Q.add((t,n,'S'))

        if event=='I':
            #next input 
            #np.random.seed() # if conn is generated on the go
            #make sure to insert noise even if event is ignored ! 
            if n<N_input:
                event_t = random.expovariate(expR[n])
                Q.add((t+event_t,n,'I'))
        if t%1000==0:
            # terminate for static model search
            if len(st)/N/(t/1000)>20:
                print('terminated since the fr was too high')
                break

    params_json = json.dumps(params)
    hash_name = hashlib.sha256(params_json.encode('utf-8')).hexdigest()
    print(hash_name)
    print(params)
    print(sim_time)
    np.save('sim/%s'%hash_name,[st,gid,params])
    with open("sim/log.txt", "a") as myfile:
        myfile.write(params_json)
    print(len(st)/N/(sim_time/1000))
    return 'Done'
