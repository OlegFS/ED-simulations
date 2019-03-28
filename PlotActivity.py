%matplotlib inline
import pandas as pd 
from func.processing_helpers import *


### Model with low variance, check if I can scale it to 1000 neurons 
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
          'tau_w':16000,
          'a':0.0,
#           'conn':'rand',
          #'w':'E',
          'b':1}

Eps =[0.2]#,0.1,0.2,0.3,0.5,0.7,0.8,0.93]#,0.99,0.994]
#[0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.93]#,0.99,0.994]
#Eps =[0.05,0.2,0.5,0.8,0.95,]

#
#Gs =[9.0,8/2,1.0,2/8,5/95,]
Gs =[4.]#,9.0,4.0,7/3,1.0,3/7,0.25,7/93]

#{'Vthr': 20,
    
Js = [16.0]#[7.2,9.0,14.4] ,14.4]# 6.7,6.8,7.0, 7.2
ex_rates =[0.09615384615384616/5]#,0.09615384615384616/2]#
Jexts = [8.0]#,3.0]#,6.7 - is not very nice
N = params['N']


g_modifier=  [1.0]#[0,0.5,1.0,1.5,2.0,4.0]#4.0]#[0,0.5,1,1.5]#,2.0,4.0]#[0.5,1.0,1.5]
burst_map = np.zeros([len(Eps),len(g_modifier)])*np.nan
burstCV_map = np.zeros([len(Eps),len(g_modifier)])*np.nan
burstR_map =  np.zeros([len(Eps),len(g_modifier)])*np.nan
burstA_map =  np.zeros([len(Eps),len(g_modifier)])*np.nan
sns.set_style('ticks')
sns.set_context('poster')
plt.figure(figsize=(25,20))
S = []
Taus = [ 0]
for e_index,epsilon in enumerate(Eps):
    for tau_index, tau in enumerate(Taus):
        for j_index,j in enumerate(Js):
            for er_index,ex_rate in enumerate(ex_rates):
                for jex_index,jex in enumerate(Jexts):
                    for g_i,g in enumerate(g_modifier):
                        params['input'] = 'NE+NI'
                        if epsilon == 0.05:
                            params['intput']= 'NE'
                            j = 16
                            g = 0.6
                        
                        params['epsilon']  = epsilon
                        params['J'] =(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1)
                        params['J_ext'] = jex#params['J']
                        params['ex_rate'] = (ex_rate/80)*((N-(epsilon*N))*0.1)#ex_rate
                        params['g'] =Gs[e_index]*g
                        try:
                            st,gid = read(params)
                            st= na(st)
                            gid= na(gid)
                            sim_time = 2000000#np.max(st)
                            bin_size = 20
                            #plt.figure(figsize=(25,3))
                            sc,_ = np.histogram(st,np.arange(0,sim_time,bin_size))
                            sc = sc[1000:]
                            
                            bin_size = 20
                            NE = N-(epsilon*N)
                            if NE >50:
                                thr = NE
                            else:
                                thr =50
                            #thr = np.std(sc)*5
                            tau =2000/bin_size
                            signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')
                            bInd= giveBurstInd(signal)
                            
                            peakAmp = [np.max(signal[bI:bI+100]) for bI in bInd]
                            S.append(signal)
                            #peakAmp = signal[na(bInd)+3]

                           #peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)
                            #plt.plot(peakTimes_,peakAmp['peak_heights'],'*')
                            ibis = np.diff(bInd)*20/1000
                            
                            burst_map[e_index,g_i] = np.mean(ibis)
                            burstCV_map[e_index,g_i] = np.std(ibis)/np.mean(ibis)
                            burstR_map[e_index,g_i] =np.corrcoef(ibis,peakAmp[:-1])[0,1]
                            burstA_map[e_index,g_i] =np.mean(peakAmp)
                            
                            if True:
                                
                                plt.figure(figsize=(25,5))
                                plt.plot(st,gid,'.',markersize = 1.); 
                                sns.despine()
#                                 plt.plot(bInd,peakAmp,'*')
                                #plt.subplot(len(g_modifier),1,g_i+1)
                                #plt.plot(st,gid,'.',markersize = 2.1)
                                plt.xlim([100000,350000])
                                plt.xlim([100000,110000])
                                #plt.plot(np.arange(0,250000-20,20)[1000:],sc)#-g_i*5)#-np.mean(sc))/np.std(sc))-(g_i*40))
                                #plt.xlabel('time (ms)')
                                #plt.ylabel('spikes')
                                #plt.title([epsilon+'IPSP='+str(params['g'])])
#                                 plt.xticks(np.arange(0,250000/20,))
                                #plt.plot(((sc-np.mean(sc))/np.std(sc))-(g_i*40))
                                #plt.axhline(thr)
                            #plt.yscale('log')
                            #tau =1000/bin_size
                            #signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')
                            #F0=np.percentile(signal,[55])
                            #plt.plot(((signal-np.median(signal))/np.median(signal))-(8*e_index))
                            #plt.plot((signal/np.max(signal))-e_index*1)
                            #plt.plot(((signal-np.mean(signal))/np.std(signal))-e_index*20)
                            #((signal-np.mean(signal))/np.std(signal))
                        except:
                            continue
                            

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

import matplotlib
# matplotlib.use('PDF')
import matplotlib.pylab as plt
from matplotlib import rc
plt.rcParams['ps.useafm'] = True
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rcParams['pdf.fonttype'] = 42


plt.figure(figsize=(15,5))
plt.plot(st,gid,'k|',markersize = 2.5); 
sns.despine()
#                                 plt.plot(bInd,peakAmp,'*')

#plt.subplot(len(g_modifier),1,g_i+1)
#plt.plot(st,gid,'.',markersize = 2.1)
# plt.xlim([100000,350000])
x1,x2= 100000,300000
plt.xlim([x1,x2])
bin_size = 20
# sc,_ = np.histogram(st,np.arange(x1,x2,bin_size))
# plt.plot(np.arange(x1,x2-bin_size,bin_size),(sc/np.max(sc))*1000,'k',label = 'mean firing rate')
plt.axvspan( 111500,113000,0,1000,alpha=0.7,color= 'r')
plt.xticks(np.arange(x1,x2+1,50000),np.arange(x1,x2+1,50000)/1000)
plt.ylabel('Neuron ID')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.savefig('figs/model_activity.pdf',fmt='pdf')
plt.savefig('figs/model_activity.png',fmt='png',dpi=300)



plt.figure(figsize=(7,5))
plt.plot(st,gid,'k|',markersize = 2.,alpha =0.7); 
sns.despine()
#                                 plt.plot(bInd,peakAmp,'*')

#plt.subplot(len(g_modifier),1,g_i+1)
#plt.plot(st,gid,'.',markersize = 2.1)
# plt.xlim([100000,350000])
x1,x2= 111500,112000#109500,109750
plt.xlim([x1,x2])
bin_size = 20
sc,_ = np.histogram(st,np.arange(x1,x2,bin_size))
plt.plot(np.arange(x1,x2-bin_size,bin_size),(sc/np.max(sc))*1000,'r',label = 'mean firing rate',alpha =0.7)
plt.xticks(np.arange(x1,x2+1,100),np.arange(x1,x2+1,100)-x1)
plt.ylabel('Neuron ID')
plt.xlabel('Time (ms)')
plt.tight_layout()
plt.savefig('figs/burst_zoom.pdf',fmt='pdf')
plt.savefig('figs/burst_zoom.png',fmt='png',dpi=300)