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





def colapserise(A):
    #See what it does
    for t in range(len(A)-1):
        x = t+1
        while x<=len(A) and A[x]>A[t]:
            A[t]=A[x]
            x=x+1
            
    return A 

def derivative_repartition(trace):
    """Based on burst detection matlab script"""
    thr = 0
    d_signal = np.diff(trace)
    d_signal = d_signal[d_signal>0]
    bins = np.arange(np.min(d_signal),np.max(d_signal),np.std(d_signal))
    print(np.min(d_signal))
    B,_ = np.histogram(d_signal,bins)
    
    i=0
    for j in range(len(B)-1): 
        if B[j+1]>=B[j]:
            i+=1
        else:
            break
            
    for j in range(len(B)-1): 
        if B[j+1]<B[j]:
            i+=1
        else:
            break
    ind1 = i
    print(ind1)
    for j in range(len(B)-1): 
        if B[j+1]==B[j]:
            i+=1
        else:
            break
    ind2 = i
    print(ind2)
        
        
    if ind2<len(B):
        thr=bins[1]+(bins[ind2]-bins[ind1])*0.7;
    return thr 

def giveBurstInd(trace):
    
    col_trace = colapserise(trace.copy())
    thr = derivative_repartition(col_trace)
    lastindex = 0
    burst_indices= []
    for t in range(2,len(trace)):
        if col_trace[t]-col_trace[t-2] > thr:
            if t!=lastindex+1:
                burst_indices.append(t)
            lastindex = t
        
    for i in range(len(burst_indices)):
        max_i = np.max(np.hstack([1,burst_indices[i]-1]))
        min_i = np.min(np.hstack([burst_indices[i]+10,len(trace)]))
        interval = np.arange(max_i,min_i)
        O= trace[interval]
        O = O[1:]-O[0:-1]
        b = np.argmax(O)
        burst_indices[i] = interval[b]
        
    ###delete bursts with min distance###
    # 40 samples is good for 20ms bins 
    d_I = np.diff(burst_indices)
    burst_indices  =na(burst_indices)
    burst_indices = np.hstack([burst_indices[0],burst_indices[1:][d_I>40]])
    
    return burst_indices
        





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

def get_properties(params,
                   sim_time,
                   plt_t= (0,'max'),
                   thr ='NE'):
    """thr: NE or std"""
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
    if thr=='NE':
        thr = NE
    elif thr =='std':
        thr = 5*np.std(sc)
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

    tau =2000/bin_size
    signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')

    plt.plot(signal)
    # plt.title(params)
    plt.figure(figsize = (15,3))
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

    print('E FR '+np.mean(e_fr))
    print('I FR '+np.mean(i_fr))
    return 'Done' 