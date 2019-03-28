%matplotlib inline
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
from scipy.signal import find_peaks
# sys.path.append('../N_balance/')
# from func.helpers import *
na = np.array
sns.set_style('darkgrid')

# from tqdm import tqdm_notebook as tqdm

# from IPython.display import clear_output
def read(params,path = 'sim/'):
    params_json = json.dumps(params)
    name = hashlib.sha256(params_json.encode('utf-8')).hexdigest()
#     print(name)
    S = np.load(path+name+'.npy')
    st = S[0]
    gid = S[1]
    #with h5py.File('sim/'+str(name),'r',libver='latest') as f:
    #    st = list(f['st']  )
    #    gid = list(f['uid'] )
    return st, gid

def IBIstat(sc,thr):
    """
    return mean IBI, CV, and R
    """
    sc= sc[100:]
    peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)
    ibis = np.diff(peakTimes_*20/1000)
    return np.mean(ibis),np.std(ibis)/np.mean(ibis),np.corrcoef(ibis,peakAmp['peak_heights'][:-1])[0,1]

def get_properties(params,sim_time,plt_t= (0,'max')):
    st,gid = read(params)
    st= na(st)
    gid= na(gid)
    sim_time = np.max(st)
    if plt_t[1]=='max':
        plt_t = (0,sim_time)
    sc,_ = np.histogram(st,np.arange(0,sim_time,20))
    # sc = sc[1000:]
    bin_size = 20
    NE = params['N']-(params['epsilon']*params['N'])
    thr = NE
    sc= sc[100:]
    peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)#4000-NI_[i]
    print(peakTimes_)
    ibis = np.diff(peakTimes_*20/1000)
    print(ibis)
    print('meanIBI',np.mean(ibis))
    print('CV of IBI',np.std(ibis)/np.mean(ibis))
    print('R', np.corrcoef(ibis,peakAmp['peak_heights'][:-1])[0,1])
    plt.figure();
    plt.plot(ibis,peakAmp['peak_heights'][:-1],'o')
    plt.figure(figsize = (15,3))

    #tau =2000/bin_size
    #signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')

    #plt.plot(signal)
    # plt.title(params)
    plt.plot(sc)
    
    plt.plot(peakTimes_,peakAmp['peak_heights'],'*r')
    plt.ylabel('spikes')
    plt.xlabel('time (ms)')
    plt.yscale('log')
    plt.figure(figsize = (15,3))
    plt.plot(st,gid,'.',markersize = 1.5)
    plt.ylabel('rate')
    plt.xlabel('time (ms)')
    
    plt.xlim(plt_t)
    plt.figure(figsize = (15,3))
    #plt.plot(signal)
    
    print(len(st[gid[np.where(gid<NE)]])/(NE*(sim_time/1000)))
    return peakTimes_

def plot_eirates(params,sim_time):
    st,gid = read(params)
    st= na(st)
    gid= na(gid)
    NE = params['N']-(params['N']*params['epsilon'])
    bin_size = 20
    e_sc,_ = np.histogram(st[gid<NE],np.arange(0,sim_time,bin_size))
    i_sc,_ = np.histogram(st[gid>NE],np.arange(0,sim_time,bin_size))
    e_fr = []
    i_fr = []
    for n in range(params['N']):
        if n<NE:
            e_fr.append(len(st[gid==n])/(sim_time/1000))
        else:
            i_fr.append(len(st[gid==n])/(sim_time/1000))
    # sc = sc[1000:]
    #sc= sc[200:]
    plt.figure(figsize = (15,3))
    plt.plot(i_sc,'-r')
    plt.plot(e_sc)
    plt.ylabel('spikes')
    plt.xlabel('time (ms)')
    plt.yscale('log')
    plt.figure()
    plt.hist(e_fr,density=True,color='r',alpha = 0.4)
    plt.hist(i_fr,density =True)

    print(len(st[gid[np.where(gid<NE)]])/(NE*(sim_time/1000)))
    return peakTimes_   

Vthr = 20
Jext = 4
kE =np.arange(1,475)#800
tau_m = 40

rate = [(kE*Vthr*0.5) / (J*tau_m) for J in [1,2,3,4,5]]

min_waiting = []
for n in [25,100,250,300,475]:
    y_ = np.random.lognormal(10,3,size = [n,100000]);
    min_waiting.append(np.min(np.diff(y_,0)));
    
    
print(np.mean(y_));
np.std(y_)/np.mean(y_)
#_____________________________________95%_______________________________________
#_____________________________________95%_______________________________________
#_____________________________________95%_______________________________________
#_____________________________________95%_______________________________________

params = {'Vthr': 20,
          'Vres': 10,
          'V0':0,
          'tau_ref': 2.0,
          'tau_m':40,
          'd':3.5,
          'N':1000,
          'epsilon':0.95,
          'p':0.1,
          'g':5/95,
          'J':(15.5*np.sqrt(5))/np.sqrt(80),#(3*np.sqrt(25))/np.sqrt(25),# 4.0,#j,#Js[i],
          'J_ext':1.7,
          'input':'NE+NI',
          'ex_rate':(0.09615384615384616,0.09615384615384616),#615384615384616,#615384615384616,
          'eta':0.0,
          'tau_w':2000,
          'a':0.0,
          'b':2}
    #                 try:
sim_time =250000#np.max(st)

peakTimes = get_properties(params,sim_time,plt_t =(25000,26000))
peakTimes = get_properties(params,sim_time)
plot_eirates(params,sim_time)

#_____________________________________80%_______________________________________
#_____________________________________80%_______________________________________
#_____________________________________80%_______________________________________
#_____________________________________80%_______________________________________
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
          'J':2.5,#0.94868329805051377,#(3*np.sqrt(25))/np.sqrt(25),# 4.0,#j,#Js[i],
          'J_ext': 3.0,
          'input':'NE',
          'ex_rate':0.07,#0.09615384615384616,
          'eta':0.0,
          'tau_w':20000,
          'a':0.0,
          'b':2}
    #                 try:
sim_time =250000#np.max(st)
peakTimes = get_properties(params,sim_time)
plot_eirates(params,sim_time)
#_____________________________________50%_______________________________________
#_____________________________________50%_______________________________________
#_____________________________________50%_______________________________________
#_____________________________________50%_______________________________________

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
          'J':2.5,#0.94868329805051377, #25/np.sqrt(250.0),#j,#Js[i],
          'J_ext':9.0,
          'input':'NE',
          'ex_rate':0.002,#0.09615384615384616/1,#(0.09615384615384616/1.2,0.096/3),#0.09615384615384616/1.2,#3
          'eta':0.0,
          'tau_w':20000,
          'a':0.0,
          #'w':'E',
          'b':2}
    #                 try:
    

sim_time =250000#np.max(st)
peakTimes = get_properties(params,sim_time,plt_t=(0,50000))
peakTimes = get_properties(params,sim_time)
plot_eirates(params,sim_time)

#_____________________________________50%_______________________________________
#_____________________________________50%_______________________________________
#_____________________________________50%_______________________________________
#_____________________________________50%_______________________________________

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
          'J':#1.78, #25/np.sqrt(250.0),#j,#Js[i],
          'J_ext': 1.78,
          'input':'NE+NI',
          'ex_rate':'eta',
          'eta':0.5,
          'tau_w':20000,
          'a':0.0,
          'b':2}
    #                 try:
    

sim_time =250000#np.max(st)
peakTimes = get_properties(params,sim_time,plt_t=(32500,33500))
peakTimes = get_properties(params,sim_time)
#_____________________________________20%_______________________________________
#_____________________________________20%_______________________________________
#_____________________________________20%_______________________________________
#_____________________________________20%_______________________________________


params = {'Vthr': 20,
          'Vres': 10,
          'V0':0,
          'tau_ref': 2.0,
          'tau_m':40,
          'd':3.5,
          'N':5000,
          'epsilon':0.2,
          'p':0.1,
          'g':0.0,
          'J':5.0, #25/np.sqrt(250.0),#j,#Js[i],
          'J_ext': 1.8,
          'input':'NE+NI',
          'eta':0.0,
          'tau_w':10000,
          'a':0.0,
          'b':2}
    #                 try:

sim_time =250000#np.max(st)
peakTimes = get_properties(params,sim_time)

params = {'Vthr': 20,
          'Vres': 10,
          'V0':0,
          'tau_ref': 2.0,
          'tau_m':40,
          'd':3.5,
          'N':5000,
          'epsilon':0.5,
          'p':0.1,
          'g':1.0,#4.0,#950/50,
          'J':0.94868329805051377,#j,#Js[i],
          'eta':0.0,
          'tau_w':20000,
          'a':0.0,
          'b':2}
    #                 try:
sim_time =50000#np.max(st)
peakTimes = get_properties(params,sim_time)

#_____________________________________5%_______________________________________
#_____________________________________5%_______________________________________
#_____________________________________5%_______________________________________
#_____________________________________5%_______________________________________
params = {'Vthr': 20,
          'Vres': 10,
          'V0':0,
          'tau_ref': 2.0,
          'tau_m':40,
          'd':3.5,
          'N':5000,
          'epsilon':0.05,#0.95,
          'p':0.1,
          'g':9.0,
          'J':0.68824720161168529,#0.94868329805051377,
          'eta':0.0,
          'tau_w':20000,
          'a':0.0,
          'b':2}
sim_time =250000

    #                 try:
     #                 try:
params = {'Vthr': 20,
          'Vres': 10,
          'V0':0,
          'tau_ref': 2.0,
          'tau_m':40,
          'd':3.5,
          'N':5000,
          'epsilon':0.2,#0.95,
          'p':0.1,
          'g':0,#5/495,
          'J':0.0,#0.94868329805051377,
          'eta':0.0,
          'tau_w':20000,
          'a':0.0,
          'b':2}
sim_time =10000
peakTimes = get_properties(params,sim_time)   


#______________________Process a batch ______________________________________
#______________________ei model  search_________________________________
#______________________First attempt (ssearch_eiModel)______________________________________
#____________________________________________________________________________
params = {'Vthr': 20,
          'Vres': 10,
          'V0':0,
          'tau_ref': 2.0,
          'tau_m':40,
          'd':3.5,
          'N':5000,
          'epsilon':0.0,
          'p':0.1,
          'g':0.0,
          'J':0.,#25/np.sqrt(25),#0.94868329805051377,
          'J_ext':0.,
          'input':'NE+NI',
          'ex_rate':'eta',
          'eta':0.5,
          'tau_w':0,
          'a':0.0,
          'b':2}

Eps =[0.05,0.1,0.2,0.5,0.7,0.8,0.95]# [0.3,0.5,0.8]
Gs =[95/5,9/1,8/2,1.0,3/7,2/8,5/95]#[7/3,1.0,2/8]
#Etas = [0.1,0.2,0.3,0.4]
Taus = [20000]#[2000,5000,10000]
#Jexts = [2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,15.0]
Js = [1.9]#[1.0,2.0,2.5,2.8,3.0,4.0]
ex_rates = ['eta']#[0.03,0.04,0.05,0.06,0.07,0.08,0.09]
Jexts = [1.9]#[3.0]
burst_map = np.zeros([len(Eps),2])*np.nan
burstCV_map = np.zeros([len(Eps),2])*np.nan
burstR_map = np.zeros([len(Eps),2])*np.nan
for e_index,epsilon in enumerate(Eps):
    for tau_index, tau in enumerate(Taus):
        for j in Js:
            for ex_rate in ex_rates:
                for jex in Jexts:
                    for g in [1]:
                        j_=j#(1.79*np.sqrt(250))/np.sqrt((5000-(epsilon*5000))*0.1)
                        params['epsilon']  =  epsilon
                        params['J'] =j_
                        params['J_ext'] = j_
                        params['ex_rate'] = ex_rate
                        params['tau_w'] = tau
                        if g==1:
                            params['g'] = Gs[e_index]
                        else:
                            params['g'] = 0
                        try:

                            st,gid = read(params)
                            st= na(st)
                            gid= na(gid)
                            sim_time = 250000#np.max(st)
                            sc,_ = np.histogram(st,np.arange(0,sim_time,20))
                            bin_size = 20
                            NE = 5000-(epsilon*5000)
                            thr = NE
                            plt.figure(figsize=(15,5))
                #            tau =1000/bin_size
                #            signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')
                            plt.plot(st,gid,'.',markersize = 1.5)

                #            plt.plot(signal)

                            plt.title(str([j, epsilon]))
                            peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)#4000-NI_[i]
                            ibis = np.diff((peakTimes_*20)/1000)
                            burst_map[e_index,g] = np.mean(ibis)
                            burstCV_map[e_index,g] = np.std(ibis)/np.mean(ibis)
                            burstR_map[e_index, g] = np.corrcoef(ibis,peakAmp['peak_heights'][:-1])[0,1]
                        except:
                            continue

                                
                                
#______________________Process a batch ______________________________________
#______________________simple model search_________________________________
#______________________Only ex input______________________________________
#____________________________________________________________________________
params = {'Vthr': 20,
          'Vres': 10,
          'V0':0,
          'tau_ref': 2.0,
          'tau_m':40,
          'd':3.5,
          'N':5000,
          'epsilon':0.95,#0.5
          'p':0.1,
          'g':95/5,
          'J':0.,#25/np.sqrt(25),#0.94868329805051377,
          'J_ext':0.,
          'input':'NE',
          'ex_rate':0.0,
          'eta':0.0,
          'tau_w':0,
          'a':0.0,
          'b':2}
all_params = []

Eps =[0.05,0.1,0.2,0.5,0.7,0.8,0.95]# [0.3,0.5,0.8]
Gs =[95/5,9/1,8/2,1.0,3/7,2/8,5/95]#[7/3,1.0,2/8]
#Etas = [0.1,0.2,0.3,0.4]
Taus = [20000]#[2000,5000,10000]
#Jexts = [2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,15.0]
Js = [6.9]
#7.9
#[1.0,2.0,2.5,2.8,3.0,4.0]
ex_rates = [0.0002]#[0.00002,0.0002,0.002,0.02,0.05,0.07,0.09]
#[0.03,0.04,0.05,0.06,0.07,0.08,0.09]
Jexts = [20.0]
#[3.0,5.0]
burst_map = np.zeros([len(Eps)])
burstCV_map = np.zeros([len(Eps)])
for e_index,epsilon in enumerate(Eps):
    for tau_index, tau in enumerate(Taus):
        for j_index,j in enumerate(Js):
            for rate_index,ex_rate in enumerate(ex_rates):
                for jex_index,jex in enumerate(Jexts):
                    for g in [1]:
                        params['epsilon']  =  epsilon
                        params['J']=(6.9*np.sqrt(25))/np.sqrt((5000-(5000*epsilon))*0.1)
                        params['J_ext'] = jex
                        params['ex_rate'] = ex_rate
                        params['tau_w'] = tau
                        if g==1:
                            params['g'] = Gs[e_index]
                        else:
                            params['g'] = 0

                        try:

                            st,gid = read(params)

                            st= na(st)
                            gid= na(gid)
                            sim_time = 300000#np.max(st)
                            sc,_ = np.histogram(st,np.arange(0,sim_time,20))
                            bin_size = 20
                            NE = 5000-(epsilon*5000)
                            thr = NE
                            plt.figure(figsize=(15,5))
                            tau =1000/bin_size
                            signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')
                            plt.plot(st,gid,'.',markersize = 0.5)
                            sc = sc[1000:]
                            #plt.xlim([10000,20000])
                            peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=100)#4000-NI_[i]
                            plt.title(str([params['J'], epsilon,g, len(peakTimes_)]))
                            ibis = np.diff((peakTimes_*20)/1000)
                            burst_map[e_index] = np.mean(ibis)
                            burstCV_map[e_index] = np.std(ibis)/np.mean(ibis)
                        except:
                            continue


sns.heatmap(burst_map[0,:,:],annot=True, xticklabels=Jexts, yticklabels=ex_rates)
sns.heatmap(burstCV_map[0,:,:],annot=True, xticklabels=Jexts, yticklabels=ex_rates)
sns.heatmap(burstCV_map[:,:,0],annot=True, xticklabels=ex_rates, yticklabels=Js)

#______________________Process a batch ______________________________________
#______________________50% simple model search_________________________________
#______________________Only ex input______________________________________
#____________________________________________________________________________
params = {'Vthr': 20,
          'Vres': 10,
          'V0':0,
          'tau_ref': 2.0,
          'tau_m':40,
          'd':3.5,
          'N':5000,
          'epsilon':0.95,#0.5
          'p':0.1,
          'g':95/5,
          'J':0.,#25/np.sqrt(25),#0.94868329805051377,
          'J_ext':0.,
          'input':'NE',
          'ex_rate':0.0,
          'eta':0.0,
          'tau_w':0,
          'a':0.0,
          'b':2}


Eps =[0.5]#[0.5]#[0.05,0.1,0.2,0.5,0.7,0.8]# [0.3,0.5,0.8]
#options: [0.5]
Gs =[1.0]#[5/95]# [95/5,9/1,8/2,1.0,3/7,2/8]#[7/3,1.0,2/8]
#Etas = [0.1,0.2,0.3,0.4]
Taus = [20000]#[2000,5000,10000]
#Jexts = [2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,15.0]
Js = [1.0,2.0,2.5,2.8,4.0,4.0]
#[8.0][8.0,9.0]#
ex_rates =[0.00002,0.0002,0.002,0.02,0.05,0.07,0.09]#[0.03,0.04,0.05,0.06,0.07,0.08,0.09]
# [0.00002,0.0002,0.002,0.02,0.05,0.07,0.09]
#
Jexts = [3.0,5.0]#[2.0,3.0,5.0,8.0,9.0,10.0]
#[3.0,5.0]
burst_map = np.zeros([len(Js),len(ex_rates),len(Jexts)])*np.nan
burstCV_map = np.zeros([len(Js),len(ex_rates),len(Jexts)])*np.nan
for e_index,epsilon in enumerate(Eps):
    for tau_index, tau in enumerate(Taus):
        for j_index,j in enumerate(Js):
            for rate_index,ex_rate in enumerate(ex_rates):
                for jex_index,jex in enumerate(Jexts):
                    for g in [1]:
                        params['epsilon']  =  epsilon
                        params['J'] = j
                        params['J_ext'] = jex
                        params['ex_rate'] = ex_rate
                        params['tau_w'] = tau
                        if g==1:
                            params['g'] = Gs[e_index]
                        else:
                            params['g'] = 0

                        try:

                            st,gid = read(params)

                            st= na(st)
                            gid= na(gid)
                            sim_time = 250000#np.max(st)
                            sc,_ = np.histogram(st,np.arange(0,sim_time,20))
                            bin_size = 20
                            NE = 5000-(epsilon*5000)
                            thr = NE
                           # plt.figure(figsize=(15,5))
                           # plt.plot(st,gid,'.')
                #            tau =1000/bin_size
                #            signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')

                #            plt.plot(signal)

                           # plt.title(str([j,jex,ex_rate, epsilon,inh]))
                            peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)#4000-NI_[i]
                            ibis = np.diff((peakTimes_*20)/1000)
                            burst_map[j_index,rate_index,jex_index] = np.mean(ibis)
                            burstCV_map[j_index,rate_index,jex_index] = np.std(ibis)/np.mean(ibis)
                        except:
                            continue


sns.heatmap(burst_map[0,:,:],annot=True, xticklabels=Jexts, yticklabels=ex_rates)
sns.heatmap(burstCV_map[0,:,:],annot=True, xticklabels=Jexts, yticklabels=ex_rates)
sns.heatmap(burstCV_map[:,:,0],annot=True, xticklabels=ex_rates, yticklabels=Js)

#______________________Process a batch ______________________________________
#______________________Fixed Rate Search_____________________________________
#______________________Only ex input______________________________________
#____________________________________________________________________________

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
          'eta':0.0,
          'tau_w':20000,
          'a':0.0,
          'b':2}

Js = [3.0,3.1,3.2,3.5]#,3.0,4.0]
Eps = [0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.95]
Gs = [19.0,9.0,4.0,7/3,1.0,3/7,2/8,5/95]

burst_map = np.zeros([len(Js),len(Eps),2])
burstCV_map = np.zeros([len(Js),len(Eps),2])

for j_index,j in enumerate(Js):
    for e_index,epsilon in enumerate(Eps):
        for inh in [0]:#[0,1]:
            params['J'] = j
            params['epsilon'] = epsilon
            if inh ==1:
                params['g'] = 0
            else:
                params['g'] = Gs[e_index]
            try:
                st,gid = read(params)
                #
                #plt.title(str([j, epsilon,inh]))
                #plt.plot(st,gid,'.',markersize =3.5)
                #plt.xlim([0,50000])
                #plt.ylim([0,5000])
            except:
                st,gid = [0],[0]
                
            st= na(st)
            gid= na(gid)
            sim_time = 250000#np.max(st)
            sc,_ = np.histogram(st,np.arange(0,sim_time,20))
            bin_size = 20
            NE = 5000-(epsilon*5000)
            thr = NE
#            plt.figure(figsize=(15,5))
#            tau =1000/bin_size
#            signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')

#            plt.plot(signal)
    
#            plt.title(str([j, epsilon,inh]))
            peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)#4000-NI_[i]
            ibis = np.diff((peakTimes_*20)/1000)
            burst_map[j_index,e_index, inh] = np.mean(ibis)
            burstCV_map[j_index,e_index, inh] = np.std(ibis)/np.mean(ibis)
            
sns.heatmap(burst_map[:,:,1],annot=True, xticklabels=Eps, yticklabels=Js)


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
          'eta':0.0,
          'tau_w':20000,
          'a':0.0,
          'b':2}

Js = [3.0]#,3.0,4.0]
Eps = [0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.95]
Gs = [19.0,9.0,4.0,7/3,1.0,3/7,28,5/95]
plt.figure(figsize=(10,10))
for i,epsilon in enumerate(Eps):
    params['epsilon'] = epsilon
    params['J'] = Js[0]
    params['g'] = Gs[i]
    st,gid = read(params)
    plt.plot(na(st),na(gid)-i*5000,'.',markersize = 1.5,label = epsilon*100);
    plt.xlim([50000,250000])
plt.legend(title='% of inh.' ,bbox_to_anchor=(1, 0.5))    
plt.xlabel('time (ms)')
plt.yticks([])


sns.heatmap(burstCV_map[:,:,0],annot=True, xticklabels=Eps,yticklabels=Js); plt.xlabel('% inh.'); plt.ylabel('J'); 

#______________________Process a batch ______________________________________
#______________________J~1/sqrt(K) sclaing___________________________________
#_______________________Only excitatory input________________________________
#__________________________________1/sqrtpy__________________________________

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
      'J_ext':3.0,
      'input':'NE',
      'ex_rate':0.09615384615384616,
      'eta':0.0,
      'tau_w':20000,
      'a':0.0,
      'b':2}

Eps = [0.05,0.1,0.2,0.5,0.7,0.8,0.95] 

Gs =  [9.0,9/1,8/2,1.0,3/7,2/8,5/95]#

burst_map = np.zeros([len(Eps),2])
burstCV_map = np.zeros([len(Eps),2])
burstR_map = np.zeros([len(Eps),2])


for e_index,epsilon in enumerate(Eps):
    for inh in [0,1]:
        j =  15/np.sqrt((5000-(5000*epsilon))*0.1) 
        params['J'] = j
        params['epsilon'] = epsilon
        if inh ==1:
            params['g'] = 0
        else:
            params['g'] = Gs[e_index]
            
        try:
            st,gid = read(params)
            #
            plt.figure()
            plt.title(str([j, epsilon,params['g'],inh]))
            plt.plot(st,gid,'.',markersize =3.5)
#            tau =1000/bin_size
#            signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')

#            plt.plot(signal)
            #plt.xlim([0,50000])
#            plt.ylim([0,5000])
        except:
            st,gid = [0],[0]

        st= na(st)
        gid= na(gid)
        sim_time = 600000#np.max(st)
        sc,_ = np.histogram(st,np.arange(0,sim_time,20))
        sc = sc[500:]
        bin_size = 20
        NE = 5000-(epsilon*5000)
        thr = NE
        #plt.figure()
        #plt.title(str([j, epsilon,params['g'],inh]))
#            plt.plot(st,gid,'.',markersize =3.5)
        tau =1000/bin_size
        #signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')

        #plt.plot(signal)
            #plt.xlim([0,50000])
#            plt.figure(figsize=(15,5))
#            tau =1000/bin_size
#            signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')

#            plt.plot(signal)

#            plt.title(str([j, epsilon,inh]))
        peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)#4000-NI_[i]
        ibis = np.diff((peakTimes_*20)/1000)
        burst_map[e_index, inh] = np.mean(ibis)
        burstCV_map[e_index, inh] = np.std(ibis)/np.mean(ibis)
        burstR_map[e_index, inh] = np.corrcoef(ibis,peakAmp['peak_heights'][:-1])[0,1]
        
#### plot the IBIs
plt.figure()
plt.plot(na(Eps)*100,burst_map[:,0],'-o',label='J~1/sqrt(K) model'); plt.errorbar([5,10,20,30,50,70,80,95],origIBI,IBIstd, label ='data'); plt.xticks(na(Eps)*100); plt.ylabel('IBI (s)'); plt.xlabel('%'); plt.legend()
plt.figure()
plt.plot([0,10,20,50,70,80,95],burstCV_map[:,0],'-o',label = 'Fixed J model'); plt.plot([0,10,20,25,30,50,70,80,95],CVdata,'o-',label = 'data'); plt.legend(); plt.xlabel('%'); plt.ylabel('CV')
Js = [3.0]#,3.0,4.0]
Eps = [0.05,0.1,0.2,0.5,0.7,0.8,0.95]
Gs = [9.0,9.0,4.0,1.0,3/7,2/8,5/95]
plt.figure(figsize=(10,10))
for i,epsilon in enumerate(Eps):
    params['epsilon'] = epsilon
    params['J'] = 15/np.sqrt((5000-(5000*epsilon))*0.1) 
    params['g'] = Gs[i]
    st,gid = read(params)
    plt.plot(na(st),na(gid)-i*5000,'.',markersize = 1.5,label = epsilon*100);
    plt.xlim([50000,250000])
plt.legend(title='% of inh.' ,bbox_to_anchor=(1, 0.5))    
plt.xlabel('time (ms)')
plt.yticks([])


#______________________Process a batch ______________________________________
#______________Exploring variance in Asynchornous inhibition networks________
#____________________________________________________________________________
#____________________________________________________________________________
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
              'input':'NE',
              'eta':0.0,
              'tau_w':0,
              'a':0.0,
              'b':2}

Eps = [0.5]
Gs =  [1.0]

Taus = [2000,5000,10000]
Jexts = [1.5,1.8,2.0,3.0,4.0]
Js = [2.0,3.0,4.0,5.0,6.0]
burst_map = np.zeros([len(Js),len(Jexts),len(Taus)])*np.nan
burstCV_map = np.zeros([len(Js),len(Jexts),len(Taus)])*np.nan
for e_index,epsilon in enumerate(Eps):
    for tau_index, tau in enumerate(Taus):
        for j_index,j in enumerate(Js):
            for je_index,Jext in enumerate(Jexts):

                for g in [1]:
                    params['epsilon']  =  epsilon
                    params['J'] = j
                    params['J_ext'] = Jext
                    params['tau_w'] = tau
                    if g==1:
                        params['g'] = Gs[e_index]
                    else:
                        params['g'] = 0
                    try:
                        st,gid = read(params)

                
                        st= na(st)
                        gid= na(gid)
                        sim_time = 250000#np.max(st)
                        sc,_ = np.histogram(st,np.arange(0,sim_time,20))
                        bin_size = 20
                        NE = 5000-(epsilon*5000)
                        thr = NE
                #            plt.figure(figsize=(15,5))
                #            tau =1000/bin_size
                #            signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')

                #            plt.plot(signal)

                #            plt.title(str([j, epsilon,inh]))
                        peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)#4000-NI_[i]
                        ibis = np.diff((peakTimes_*20)/1000)
                        burst_map[j_index,je_index, tau_index] = np.mean(ibis)
                        burstCV_map[j_index,je_index, tau_index] = np.std(ibis)/np.mean(ibis)
                    except:
                        continue
                        
sns.heatmap(burstCV_map[:,:,2],xticklabels=Jexts,yticklabels=Js,annot=True); plt.xlabel('J_ext'); plt.ylabel('J'); plt.title('tau_w = 10s, 50% inh.')




#______________________Process a batch ______________________________________
#______________Exploring variance in inhibitory input networks________
#____________________________________________________________________________
#____________________________________________________________________________
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

Eps = [0.5]
Gs =  [1.0]

Taus = [10000,20000]
Jexts = [1.5,1.8,2.0,3.0,4.0]
Js = [2.0,3.0,4.0,5.0,6.0]
burst_map = np.zeros([len(Js),len(Jexts),len(Taus)])*np.nan
burstCV_map = np.zeros([len(Js),len(Jexts),len(Taus)])*np.nan
burstR_map = np.zeros([len(Js),len(Jexts),len(Taus)])*np.nan
for e_index,epsilon in enumerate(Eps):
    for tau_index, tau in enumerate(Taus):
        for j_index,j in enumerate(Js):
            for je_index,Jext in enumerate(Jexts):

                for g in [1]:
                    params['epsilon']  =  epsilon
                    params['J'] = j
                    params['J_ext'] = Jext
                    params['tau_w'] = tau
                    if g==1:
                        params['g'] = Gs[e_index]
                    else:
                        params['g'] = 0
                    try:
                        st,gid = read(params)

                
                        st= na(st)
                        gid= na(gid)
                        sim_time = 250000#np.max(st)
                        sc,_ = np.histogram(st,np.arange(0,sim_time,20))
                        bin_size = 20
                        NE = 5000-(epsilon*5000)
                        thr = NE
                #            plt.figure(figsize=(15,5))
                #            tau =1000/bin_size
                #            signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')

                #            plt.plot(signal)

                #            plt.title(str([j, epsilon,inh]))
                        peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)#4000-NI_[i]
                        ibis = np.diff((peakTimes_*20)/1000)
                        burst_map[j_index,je_index, tau_index] = np.mean(ibis)
                        burstCV_map[j_index,je_index, tau_index] = np.std(ibis)/np.mean(ibis)
                        burstR_map[j_index,je_index, tau_index] = np.corrcoef(ibis,peakAmp['peak_heights'][:-1])[0,1]
                    except:
                        continue
                        
sns.heatmap(burstCV_map[:,:,2],xticklabels=Jexts,yticklabels=Js,annot=True); plt.xlabel('J_ext'); plt.ylabel('J'); plt.title('tau_w = 10s, 50% inh.')




#______________________Process a batch ______________________________________
#______________General fit in the netwrok with inh. input________
#_______________Inh. Drive search______________________
#____________________________________________________________________________
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
          'J':0.,#25/np.sqrt(25),#0.94868329805051377,
          'J_ext':0.,
          'input':'NE+NI',
          'eta':0.0,
          'tau_w':0,
          'a':0.0,
          'b':2}

Eps =[0.2,0.5,0.8,0.95]#[0.05,0.1,0.2,0.5,0.7,0.8]# [0.3,0.5,0.8]
Gs =[4.0,1.0,2/8,5/95]# [95/5,9/1,8/2,1.0,3/7,2/8]#[7/3,1.0,2/8]

#Etas = [0.1,0.2,0.3,0.4]
Taus = [10000]#[2000,5000,10000]
Jexts = [2.0]
Js = [2.0,3.0,4.0,5.0,6.0]
burst_map = np.zeros([len(Eps),len(Js),2])*np.nan
burstCV_map = np.zeros([len(Eps),len(Js),2])*np.nan
burstR_map = np.zeros([len(Eps),len(Js),2])*np.nan
for e_index,epsilon in enumerate(Eps):
    for tau_index, tau in enumerate(Taus):
        for j_index,j in enumerate(Js):
            for je_index,Jext in enumerate(Jexts):
                for g in [1,0]:
                    params['epsilon']  =  epsilon
                    params['J'] = j
                    params['J_ext'] = Jext
                    params['tau_w'] = tau
                    if g==1:
                        params['g'] = Gs[e_index]
                    else:
                        params['g'] = 0
                    try:
                        st,gid = read(params)
                        st= na(st)
                        gid= na(gid)
                        sim_time = 250000#np.max(st)
                        sc,_ = np.histogram(st,np.arange(0,sim_time,20))
                        bin_size = 20
                        NE = 5000-(epsilon*5000)
                        thr = NE
                        peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)#4000-NI_[i]
                        ibis = np.diff((peakTimes_*20)/1000)
                        burst_map[e_index,j_index, g] = np.mean(ibis)
                        burstCV_map[e_index,j_index, g] = np.std(ibis)/np.mean(ibis)
                        burstR_map[e_index,j_index, g] = np.corrcoef(ibis,peakAmp['peak_heights'][:-1])[0,1]
                    except:
                        continue
                        
                        
                
sns.heatmap(burstCV_map[:,:,0],xticklabels=Jexts,yticklabels=Js,annot=True); plt.xlabel('J_ext'); plt.ylabel('J'); plt.title('tau_w = 10s, 50% inh.')

#______________________Process a batch ______________________________________
#______________________1/Ke Scaled Input Search_____________________________________
#____________________________________________________________________________
#____________________________________________________________________________

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
          'eta':0.0,
          'tau_w':20000,
          'a':0.0,
          'b':2}

Js = [3.0]#,3.0,4.0]
Eps = [0.5,0.7,0.95]#[0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.95]
Gs = [1.0,3/7,5/95]#[19.0,9.0,4.0,7/3,1.0,3/7,2/8,5/95]

burst_map = np.zeros([len(Js),len(Eps),2])
burstCV_map = np.zeros([len(Js),len(Eps),2])

for j_index,j in enumerate(Js):
    for e_index,epsilon in enumerate(Eps):
        for inh in [0,1]:
            params['J'] = j
            params['epsilon'] = epsilon
            if inh ==1:
                params['g'] = 0.0
            else:
                params['g'] = Gs[e_index]
            try:
                st,gid = read(params)
                plt.figure()
                plt.title(str([j, epsilon,inh]))
                plt.plot(st,gid,'.',markersize =0.5)
            except:
                st,gid = [0],[0]
                
            st= na(st)
            gid= na(gid)
            sim_time = 250000#np.max(st)
            sc,_ = np.histogram(st,np.arange(0,sim_time,20))
            bin_size = 20
            NE = 5000-(epsilon*5000)
            thr = NE

            peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)#4000-NI_[i]
            ibis = np.diff((peakTimes_*20)/1000)
            burst_map[j_index,e_index, inh] = np.mean(ibis)
            burstCV_map[j_index,e_index, inh] = np.std(ibis)/np.mean(ibis)
            
    

#______________________Process a batch ______________________________________
#______________________Small Tau Search______________________________________
#____________________________________________________________________________
#____________________________________________________________________________
params = {'Vthr': 20,
          'Vres': 10,
          'V0':0,
          'tau_ref': 2.0,
          'tau_m':40,
          'd':3.5,
          'N':1000,
          'epsilon':0.05,
          'p':0.1,
          'g':0.0,#Gs[i],
          'J':0,
          'eta':0,
          'tau_w':0,
          'a':0.0,
          'b':2}

Js = [2.0,3.0,4.0]#,3.0,4.0]
Etas = [0.1,0.2,0.3,0.4]
Taus = [1000,3000]
burst_map = np.zeros([len(Js),len(Etas),len(Taus)])
burstCV_map = np.zeros_like(burst_map)
for j_index,j in enumerate(Js):
    for e_index,eta in enumerate(Etas):
        for t_index,tau in enumerate(Taus):
            params['tau_w']  = tau
            params['J'] = j
            params['eta'] = eta
            try:
                st,gid = read(params)
            except:
                st,gid = [0],[0]
            
            st= na(st)
            gid= na(gid)
            sim_time = 1000000#np.max(st)
            sc,_ = np.histogram(st,np.arange(0,sim_time,20))
            bin_size = 20
            NE = 1000-(0.05*1000)
            thr = NE

            peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)#4000-NI_[i]
            ibis = np.diff((peakTimes_*20)/1000)
            burst_map[j_index,e_index, t_index] = np.mean(ibis)
            burstCV_map[j_index,e_index, t_index] = np.std(ibis)/np.mean(ibis)
            tau =1000/bin_size
            # signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'full')


Js = [4.0]#[2.0,3.0]#,3.0,4.0]
Etas = [0.1,0.2,0.3]#,0.4]
Taus = [1000]#[1000,3000]
# burst_map = np.zeros([len(Js),len(Etas),len(Taus)])
for j_index,j in enumerate(Js):
    for e_index,eta in enumerate(Etas):
        for t_index,tau in enumerate(Taus):
            params['tau_w']  = tau
            params['J'] = j
            params['eta'] = eta
            try:
                st,gid = read(params)
            except:
                st,gid = [0],[0]
            st= na(st)
            gid= na(gid)
            sim_time = 1000000#np.max(st)
            sc,_ = np.histogram(st,np.arange(0,sim_time,20))
            bin_size = 20
            NE = 1000-(0.05*1000)
            thr = NE
            plt.plot(sc-e_index*10000)
            plt.xlim([0,5000])
            # signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'full')

origIBI = [65.467486666666659,
 12.760416666666666,
 10.365777777777778,
 12.699990909090911,
 17.38184,
 16.656661538461542,
 20.871327272727271,
 67.648690000000002]
IBIstd = [5.3267141099138744,
 1.6721213659101821,
 1.1630823658614113,
 1.5865836726256906,
 2.4175298579335065,
 2.7513295254551071,
 4.778779721010312,
 19.244195723840129]
cvs =[1.3926423814352991,
 0.86069716712158839,
 0.82681015882400044,
 1.0341554952132499,
 0.91389182679149228,
 0.89143558038044346,
 0.78179435111134454]
CVdata = [0.301212628,0.483898774,0.374725971,0.366898665,0.451753893,0.449703728,0.577622938,0.556537065,0.735208728]
CVstd = [0.132827791,0.099404768,0.065518815,0.132088579,0.099347754,0.084593953,0.081237032,0.061673205,0.119522766]
CVsem = [0.034295988,0.028695685,0.021839605,0.038130688,0.029954475,0.026750957,0.022531099,0.018595171,0.037796417]