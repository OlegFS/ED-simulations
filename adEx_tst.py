# %load EB-adLIF.py
#!/usr/bin/python
# Event-based Brunel network
# Compare EDN with NEST output

import numpy as np
import hashlib
import matplotlib.pyplot as plt
from numba import jit
from sortedcontainers import SortedDict, SortedList
na = np.array
import pickle
import time
import sys
import seaborn as sns
import h5py as h5py
import json
import hashlib
sys.path.append('../N_balance/')
from func.helpers import *
na = np.array
sns.set_style('darkgrid')
sys.path.append('../N_balance/')
from func.AdEx_meta import *
from func.helpers import *
# Load ED 
    # Set params
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
          'g':5,
          'J':0.5,
          'eta':2.8,
          'tau_w':0.,
          'b':0.}


# sim_time = 200
J =params['J']#,700.,700.,700.,700.,500.,510.,510.,450.]#400.,450.,450.,450.,600.]#[3822.]#[392.,427.,540.,854.,1709.,3822.]#[ 450., 450., 450.,450.]#[400.,450.,450.,450.,600.]#[400.,400.,400.,400.,400.]
eta =params['eta']#np.float(sys.argv[2])#0.5

d =  [params['d']]
NI=200#,100,200,250,300,500,700,800,950]#,200,500,800,990]#[40,200,800,3200,16000]#[50,200,500,800,950,990]#[200,500,800,950]#[200,800,2000,3200,3800]
sim_time=100#110
directory   = 'sim/AdEx_balance/'
NE =int(1000-NI)#int(5000-NI)

jA = params['J']
j = jA
a =0.
b =np.float(params['b'])
tau_w = params['tau_w']

g =params['g']#np.float(sys.argv[1])#4.0#(NE/NI)
simulation='AdEx_J=%s_g=%s_eta=%s_a=%s_b=%s_tau_w=%s_NI=%s_N_tot=%s_sim_time=%s'%(J,g,eta,a,b,tau_w, NI,NI+NE,sim_time)
#simulation='tsodyks_j=%s_g=%s_eta=%s_NI=%s_u=%s_tau_rec=%s_tau_fac=%s_Nt=%s'%(j,g,eta,NI,U,tau_rec,tau_fac,NI+NE)
A = AdExModel(directory= directory,
                        simulation = simulation,
                        g=np.round(g,decimals=3), # inhibitor0y strenght
                        eta = np.round(eta,decimals=3),
                        d=d, # synaptic delay
                        J=jA, #synaptic strengthd
                        NE =NE, # fraction of inh neurons
                        NI= NI,
                        N_rec = NE+NI,
                        epsilon = 0.1,
                        tauMem =40.,
                        simtime=sim_time,
                        master_seed = 1000,
                        verbose = True,
                        chunk= False,
                        chunk_size= 50000,
                        voltage = False)
A.build(tau_w = tau_w,Delta_T=0.0, a=a,b =b, constant_k=[0.1,0.1])#rate = 180.)#196
A.connect(j_cnqx=False)
A.run()
# conn = A.get_connectivity()
