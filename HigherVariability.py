%matplotlib inline
import pandas as pd 
from func.processing_helpers import *




                           
origEps = [0.05,0.1,0.2,0.3,0.5,0.7,0.8,1]
origEps2 = [0.05,0.1,0.2,0.25,0.3,0.5,0.7,0.8,1]
origIBI = [65.467486666666659,
 12.760416666666666,
 10.365777777777778,
 12.699990909090911,
 17.38184,
 16.656661538461542,
 20.871327272727271,
 67.648690000000002]
stds = [5.3267141099138744,
 1.6721213659101821,
 1.1630823658614113,
 1.5865836726256906,
 2.4175298579335065,
 2.7513295254551071,
 4.778779721010312,
 19.244195723840129]
CVdata = [0.301212628,0.483898774,0.374725971,0.366898665,0.451753893,0.449703728,0.577622938,0.556537065,0.735208728]
CVstd = [0.132827791,0.099404768,0.065518815,0.132088579,0.099347754,0.084593953,0.081237032,0.061673205,0.119522766]
CVsem = [0.034295988,0.028695685,0.021839605,0.038130688,0.029954475,0.026750957,0.022531099,0.018595171,0.037796417]
BIC_sat = pd.read_csv('data/bic_saturation.csv')
BIC_sat = na(BIC_sat)

plt.plot(na(Eps)*100,burst_map[:,0],'-o',label='model'); plt.errorbar(na(origEps)*100,origIBI,stds,label='data')
sns.despine()
plt.ylabel('IBI(s)')
plt.xlabel('%')
plt.legend()
# plt.yscale('log')

plt.plot(na(Eps)*100,burstCV_map[:,0],'-o',label='model')
plt.errorbar(na(origEps2)*100,CVdata,CVsem,label='data')
sns.despine()
plt.ylabel('CV')
plt.xlabel('%')
plt.legend()


plt.plot(na(Eps)*100,burstA_map[:,0],'-o',label='model')
plt.ylabel('Amplitude (spikes)')
plt.xlabel('%')
plt.legend()
plt.yscale('log')

plt.errorbar(na([5,10,20,30,50,70,80]),np.nanmean(na(BIC_sat[:,0:-1]),0),
             np.nanstd(na(BIC_sat[:,0:-1]),0),fmt= 'o--',color= 'darkorange',label='data: max [BIC]')

plt.errorbar(na(origEps[0:-1])*100,origIBI[0:-1],stds[0:-1],color = 'darkorange',label='data: no BIC')
plt.plot(na(Eps[0:-1])*100,burst_map[0:-1,0],'-o',color='C0',label='model'); 
plt.plot(na(Eps[0:-1])*100,burst_map[0:-1,-1],'--o',color='C0',label='model: no inh'); 
plt.legend(bbox_to_anchor=(1.1, 0.9))
plt.yscale('log')
plt.xlabel('% inh.')
plt.ylabel('IBI (s)')

# Varios Gs
with sns.color_palette("Oranges"):

    [plt.plot(na(Eps)*100,burst_map[:,i],'-o',label=['g*'+str(g_modifier[i])]) for i in range(len(g_modifier))]; 
    plt.errorbar(na(origEps)*100,origIBI,stds,label='data',color= 'k')
    sns.despine()
    plt.ylabel('IBI(s)')
    plt.xlabel('%')
    plt.legend()
    plt.legend(bbox_to_anchor=(1.1, 0.9))
    plt.yscale('log')



# Varios Gs
with sns.color_palette("Oranges"):

    [plt.plot(na(Eps)*100,burstCV_map[:,i],'-o',label=['g*'+str(g_modifier[i])]) for i in range(len(g_modifier))]; 
    plt.errorbar(na(origEps2)*100,CVdata,CVsem,color='k',label='data')
    sns.despine()
    plt.ylabel('CV')
    plt.xlabel('%')
    plt.legend()
    plt.legend(bbox_to_anchor=(1.1, 0.9))
    #plt.yscale('log')



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
          'input':'NE',
          'ex_rate':0,
          'eta':0.0,
          'tau_w':20000,
          'a':0.0,
          #'w':'E',
          'conn':'rand',
          'b':2}

Eps =[0.5]#,0.1,0.2,0.3,0.5,0.7,0.8,0.93]#,0.99,0.994]
#[0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.93]#,0.99,0.994]
#Eps =[0.05,0.2,0.5,0.8,0.95,]

#
#Gs =[9.0,8/2,1.0,2/8,5/95,]
Gs =[1.0]#,9.0,4.0,7/3,1.0,3/7,0.25,7/93]

#{'Vthr': 20,
    
Js = [15.0,17.]#[7.2,9.0,14.4] ,14.4]# 6.7,6.8,7.0, 7.2
ex_rates =[0.09615384615384616/5]#,0.09615384615384616/2]#
Jexts = [1.0]#,3.0]#,6.7 - is not very nice
N = params['N']

g_modifier=  [0,1]#[0,0.5,1.0,1.5,2.0,4.0]#4.0]#[0,0.5,1,1.5]#,2.0,4.0]#[0.5,1.0,1.5]
burst_map = np.zeros([len(Eps),len(g_modifier)])*np.nan
burstCV_map = np.zeros([len(Eps),len(g_modifier)])*np.nan
burstR_map =  np.zeros([len(Eps),len(g_modifier)])*np.nan
burstA_map =  np.zeros([len(Eps),len(g_modifier)])*np.nan
sns.set_style('ticks')
sns.set_context('poster')
plt.figure(figsize=(25,20))
S = []
Taus = [0]
for e_index,epsilon in enumerate(Eps):
    for tau_index, tau in enumerate(Taus):
        for j_index,j in enumerate(Js):
            for er_index,ex_rate in enumerate(ex_rates):
                for jex_index,jex in enumerate(Jexts):
                    for g_i,g in enumerate(g_modifier):
                        params['input'] = 'NE+NI'
                        params['epsilon']  = epsilon
                        params['J'] =(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1)
                        params['J_ext'] = jex#params['J']
                        params['ex_rate'] = 0.2#(ex_rate/80)*((N-(epsilon*N))*0.1)#ex_rate
                        params['g'] =Gs[e_index]*g
                        try:
                            st,gid = read(params)
                            st= na(st)
                            gid= na(gid)
                            sim_time = 100000#np.max(st)
                            bin_size = 20
                            #plt.figure(figsize=(25,3))
                            sc,_ = np.histogram(st,np.arange(0,sim_time,bin_size))
                           # sc = sc[1000:]
                            
                            bin_size = 20
                            NE = N-(epsilon*N)
                            if NE >50:
                                thr = NE
                            else:
                                thr =50
                            #thr = np.std(sc)*5
                            tau =2000/bin_size
                            signal  = np.convolve(sc,np.exp(-np.arange(100000/bin_size)/tau),'full')
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
                                plt.subplot(121)
                                plt.plot(signal); 
                                plt.plot(bInd,peakAmp,'*')
                                #plt.subplot(len(g_modifier),1,g_i+1)
                                plt.title([j,g])
                                plt.subplot(122)
                                plt.plot(st,gid,'o',markersize = 1)            
                            
                        except:
                            print('0')

#                           plt.xlim([1000,300000])
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

                        
                        
params = {'Vthr': 20,
          'Vres': 10,
          'V0':0,
          'tau_ref': 2.0,
          'tau_m':40,
          'd':3.5,
          'N':500,
          'epsilon':0.0,
          'p':0.1,
          'g':0.0,
          'J':0.,#25/np.sqrt(25),#0.94868329805051377,
          'J_ext':0.,
          'input':'NE',
          'ex_rate':0,
          'eta':0.0,
          'tau_w':8000,
          'a':0.0,
          'conn':'rand',
#           'w':'E',
#           'leader':1,
          'b':0.5}

Eps =[0.5]#,0.1,0.2,0.3,0.5,0.7,0.8,0.93]#,0.99,0.994]
#[0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.93]#,0.99,0.994]
#Eps =[0.05,0.2,0.5,0.8,0.95,]

#
#Gs =[9.0,8/2,1.0,2/8,5/95,]
Gs =[1.0]#,9.0,4.0,7/3,1.0,3/7,0.25,7/93]

#{'Vthr': 20,
    
Js = [20.]#,10.,12.,17.,19.,21]#[7.2,9.0,14.4] ,14.4]# 6.7,6.8,7.0, 7.2
ex_rates =[0.09615384615384616/5]#,0.09615384615384616/2]#
Jexts = [10.0]#,3.0]#,6.7 - is not very nice
N = params['N']

g_modifier=  [0,1]#[0,0.5,1.0,1.5,2.0,4.0]#4.0]#[0,0.5,1,1.5]#,2.0,4.0]#[0.5,1.0,1.5]
burst_map = np.zeros([len(Js),len(g_modifier)])*np.nan
burstCV_map = np.zeros([len(Js),len(g_modifier)])*np.nan
burstR_map =  np.zeros([len(Js),len(g_modifier)])*np.nan
burstA_map =  np.zeros([len(Js),len(g_modifier)])*np.nan
sns.set_style('ticks')
sns.set_context('poster')
plt.figure(figsize=(25,20))
S = []
Taus = [0]
for e_index,epsilon in enumerate(Eps):
    for tau_index, tau in enumerate(Taus):
        for j_index,j in enumerate(Js):
            for er_index,ex_rate in enumerate(ex_rates):
                for jex_index,jex in enumerate(Jexts):
                    for g_i,g in enumerate(g_modifier):
                        params['input'] = 'NE+NI'
                        params['epsilon']  = epsilon
                        params['J'] =(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1)
                        params['J_ext'] = jex#params['J']
                        params['ex_rate'] = 0.002#(ex_rate/80)*((N-(epsilon*N))*0.1)#ex_rate
                        params['g'] =Gs[e_index]*g
                        try:
                            st,gid = read(params)
                            st= na(st)
                            gid= na(gid)
                            sim_time = 800000#np.max(st)
                            bin_size = 20
                            #plt.figure(figsize=(25,3))
                            sc,_ = np.histogram(st,np.arange(0,sim_time,bin_size))
                           # sc = sc[1000:]
                            
                            bin_size = 20
                            NE = N-(epsilon*N)
                            if NE >50:
                                thr = NE
                            else:
                                thr =50
                            #thr = np.std(sc)*5
                            tau =2000/bin_size
                            signal  = np.convolve(sc,np.exp(-np.arange(100000/bin_size)/tau),'full')
                            bInd= giveBurstInd(signal)
                            
                            peakAmp = [np.max(signal[bI:bI+100]) for bI in bInd]
                            S.append(signal)
                            #peakAmp = signal[na(bInd)+3]

                           #peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)
                            #plt.plot(peakTimes_,peakAmp['peak_heights'],'*')
                            ibis = np.diff(bInd)*20/1000
                            
                            burst_map[j_index,g_i] = np.mean(ibis)
                            burstCV_map[j_index,g_i] = np.std(ibis)/np.mean(ibis)
                            burstR_map[j_index,g_i] =np.corrcoef(ibis,peakAmp[:-1])[0,1]
                            burstA_map[j_index,g_i] =np.mean(peakAmp)
                            if True:

                                plt.figure(figsize=(25,5))
                                plt.subplot(121)
                                plt.plot(signal); 
                                plt.plot(bInd,peakAmp,'*')
                                #plt.subplot(len(g_modifier),1,g_i+1)
                                plt.title([j,g])
                                plt.subplot(122)
                                plt.plot(st,gid,'o',markersize = 1)            
                            
                        except:
                            print('0')

#                           plt.xlim([1000,300000])
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
                            