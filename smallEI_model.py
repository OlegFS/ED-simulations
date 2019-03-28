%matplotlib inline
import pandas as pd 
from func.processing_helpers import *


### Load and plot all % simulations 
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
          'ex_rate':,
          'eta':0.5,
          'tau_w':20000,
          'a':0.0,
          'b':2}



Eps =[0.05,0.2,0.5,0.8,0.95]
Gs =[19.0,8/2,1.0,0.25,5/95]
Taus = [2000]
Js = [20.0,20.0]
ex_rates =[0.09615384615384616]
Jexts = [1.5]
ib_fit = np.zeros([len(Eps),len(Js),2,2])*np.nan
# check the g-on for E and E+I models
for e_index,epsilon in enumerate(Eps):
    for tau_index, tau in enumerate(Taus):
        for j_index,j in enumerate(Js):
            for ex_rate in ex_rates:
                for jex_index,jex in enumerate(Jexts):
                    for model_i,model_type in enumerate(['NE','NE+NI']):
                        for g in [0,1]:#,0]:
                            j_=j#(1.79*np.sqrt(250))/np.sqrt((5000-(epsilon*5000))*0.1)
                            params['epsilon']  = epsilon
                            params['J'] =(j*np.sqrt(5))/np.sqrt((1000-(epsilon*1000))*0.1)
                            params['J_ext'] = jex
                            params['ex_rate'] = ex_rate
                            params['tau_w'] = tau
                            params['input'] = model_type
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
                                NE = 1000-(epsilon*1000)
                                thr = NE
                                peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)
                                ibis = np.diff((peakTimes_*20)/1000)
                                plt.figure(figsize=(15,5))
                                plt.title([epsilon, j,g, np.mean(ibis)])# np.std(ibis)/np.mean(ibis))
                                plt.plot(st,gid,'.',markersize = 1.5)
                                ib_fit[e_index,j_index,g,model_i] =np.mean(ibis)
                            except:
                                continue

                            
                            
                            
### Load and plot all % simulations 
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
          'J':0.,
          'J_ext':0.,
          'input':'NE+NI',
          'ex_rate':'eta',
          'eta':0.5,
          'tau_w':20000,
          'a':0.0,
          'b':2}

Eps =[0.05,0.2,0.5,0.8,0.95]
Gs =[95/5,8/2,1.0,2/8,5/95]
Taus = [2000,5000,20000]
Js = [25.0,20.0,15.5,10.0,5.0]#
ex_rates =[0.09615384615384616]
Jexts = [1.5,1.7,2.0,3.0]

burst_map = np.zeros([len(Eps),len(Js),len(Jexts),len(Taus)])*np.nan
burstCV_map = np.zeros([len(Eps),len(Js),len(Jexts),len(Taus)])*np.nan
burstR_map =  np.zeros([len(Eps),len(Js),len(Jexts),len(Taus)])*np.nan
for e_index,epsilon in enumerate(Eps):
    for tau_index, tau in enumerate(Taus):
        for j_index,j in enumerate(Js):
            for ex_rate in ex_rates:
                for jex_index,jex in enumerate(Jexts):
                    for g in [0]:
                        params['epsilon']  = epsilon
                        params['J'] =(j*np.sqrt(5))/np.sqrt((1000-(epsilon*1000))*0.1)
                        params['J_ext'] = jex
                        params['ex_rate'] = ex_rate
                        params['tau_w'] = tau
                        if g==1:
                            params['g'] = Gs[e_index]
                        else:
                            params['g'] = 0
                        try:
                            try:
                                st,gid = read(params)
                            except:
                                params['input']='NE'
                                st,gid = read(params)
                                
                            st= na(st)
                            gid= na(gid)
                           # plt.figure()
                            sim_time = 300000#np.max(st)
                            sc,_ = np.histogram(st,np.arange(0,sim_time,20))
                            #plt.plot(sc,'-')
                            #plt.title(params)
                            bin_size = 20
                            NE = 1000-(epsilon*1000)
                            thr = NE
                            peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)
                            ibis = np.diff((peakTimes_*20)/1000)
                            burst_map[e_index,j_index,jex_index,tau_index] = np.mean(ibis)
                            burstCV_map[e_index,j_index,jex_index,tau_index] = np.std(ibis)/np.mean(ibis)
                            burstR_map[e_index,j_index,jex_index,tau_index] =np.corrcoef(ibis,peakAmp['peak_heights'][:-1])[0,1]
                        except:
                            continue
                            

# plot
plt.figure(figsize=(15,5))
jex_index = 2
for i in range(len(Taus)):
    plt.subplot(1,3,i+1)
    #sns.heatmap(burst_map[:,:,i].T,annot=True, yticklabels=Js,xticklabels=Eps)
    [plt.plot(Eps,burst_map[:,k,jex_index,i],'o-',label = Js[k]) for k in range(len(Js))] 
    plt.xlim([0,1])
    plt.title([Taus[i],Jexts[jex_index]])
    plt.legend()

# plot
for i in range(len(Taus)):
    plt.figure()
    sns.heatmap(burst_map[:,:,0,i].T,annot=True, yticklabels=Js,xticklabels=Eps)
    plt.title(Taus[i])

    
    
    
                            
                            
### Model with low variance, check if I can scale it to 1000 neurons 
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
          'input':'NE',
          'ex_rate':0,
          'eta':0.0,
          'tau_w':20000,
          'a':0.0,
          'b':2}
Eps =[0.05,0.05,0.2,0.5,0.8,0.95]
Gs =[9.0,95/5,8/2,1.0,2/8,5/95]
Taus = [20000]
Js = [7.2]# 6.7,6.8,7.0, 7.2
ex_rates =[0.09615384615384616/70]
Jexts = [20.0]#,6.7 - is not very nice

burst_map = np.zeros([len(Eps),len(Jexts)])*np.nan
burstCV_map = np.zeros([len(Eps),len(Jexts)])*np.nan
burstR_map =  np.zeros([len(Eps),len(Jexts)])*np.nan
for e_index,epsilon in enumerate(Eps):
    for tau_index, tau in enumerate(Taus):
        for j_index,j in enumerate(Js):
            for ex_rate in ex_rates:
                for jex_index,jex in enumerate(Jexts):
                    for g in [1]:
                        j_=j#(1.79*np.sqrt(250))/np.sqrt((5000-(epsilon*5000))*0.1)
                        params['epsilon']  = epsilon
                        params['J'] =(j*np.sqrt(5))/np.sqrt((1000-(epsilon*1000))*0.1)
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
                            plt.figure(figsize=(15,3))
                            sim_time = 800000#np.max(st)
                            sc,_ = np.histogram(st,np.arange(0,sim_time,20))
                            #plt.plot(st,gid,'.',markersize = 0.5)
                            plt.plot(sc)
                            plt.title(params)
                            bin_size = 20
                            NE = 1000-(epsilon*1000)
                            if NE>50:
                                thr = NE
                            else:
                                thr = 50
                            peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)
                            ibis = np.diff((peakTimes_*20)/1000)
                            burst_map[e_index,jex_index] = np.mean(ibis)
                            burstCV_map[e_index,jex_index] = np.std(ibis)/np.mean(ibis)
                            burstR_map[e_index,j_index,jex_index,tau_index] =np.corrcoef(ibis,peakAmp['peak_heights'][:-1])[0,1]
                        except:
                            continue

                            
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
          'input':'NE+NI',
          'ex_rate':0,
          'eta':0.0,
          'tau_w':16000,
          'a':0.0,
#           'conn':'rand',
          #'w':'E',
          'b':2}

Eps =[0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.93]#,0.99,0.994]
#[0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.93]#,0.99,0.994]
#Eps =[0.05,0.2,0.5,0.8,0.95,]

#
#Gs =[9.0,8/2,1.0,2/8,5/95,]
Gs =[9.0,9.0,4.0,7/3,1.0,3/7,0.25,7/93]

#{'Vthr': 20,
    
Js = [16.0]#[7.2,9.0,14.4] ,14.4]# 6.7,6.8,7.0, 7.2
ex_rates =[0.09615384615384616/5]#,0.09615384615384616/2]#
Jexts = [8.0]#,3.0]#,6.7 - is not very nice
N = params['N']

g_modifier=  [1]#[0,0.5,1.0,1.5,2.0,4.0]#4.0]#[0,0.5,1,1.5]#,2.0,4.0]#[0.5,1.0,1.5]
burst_map = np.zeros([len(Eps),len(g_modifier)])*np.nan
burstCV_map = np.zeros([len(Eps),len(g_modifier)])*np.nan
burstR_map =  np.zeros([len(Eps),len(g_modifier)])*np.nan
burstA_map =  np.zeros([len(Eps),len(g_modifier)])*np.nan
sns.set_style('ticks')
sns.set_context('poster')
plt.figure(figsize=(25,20))
S = []
for e_index,epsilon in enumerate(Eps):
    for tau_index, tau in enumerate(Taus):
        for j_index,j in enumerate(Js):
            for er_index,ex_rate in enumerate(ex_rates):
                for jex_index,jex in enumerate(Jexts):
                    for g_i,g in enumerate(g_modifier):
                        params['input'] = 'NE+NI'
                        #if epsilon == 0.05:
                        #    params['intput']= 'NE'
                        #    j = 16
                        #    g = 0.6
                        
                        params['epsilon']  = epsilon
                        params['J'] =(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1)
                        params['J_ext'] = jex#params['J']
                        params['ex_rate'] = (ex_rate/80)*((N-(epsilon*N))*0.1)#ex_rate
                        params['g'] =Gs[e_index]*g
                        try:
                            st,gid = read(params)
                            st= na(st)
                            gid= na(gid)
                            sim_time = 3000000#np.max(st)
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
                                plt.plot(signal); 
                                plt.plot(bInd,peakAmp,'*')
                                #plt.subplot(len(g_modifier),1,g_i+1)
                                #plt.plot(st,gid,'.',markersize = 2.1)
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
                        except:
                            continue
                            

# Export data
from func.ED_sim_ad import save_obj, load_obj
save_obj({'perc':Eps,
                         'mIBI':burst_map[:,0],
                         'cvIBI':burstCV_map[:,0],
                         'aIBI':burstA_map[:,0],
                         'rIBI':burstR_map[:,0],
                         'Signal':S},
                     'Figures/data/EI_modelN=1000_16')

# sns.despine()
# plt.yticks(np.arange(-len(Eps),1,1)+1,Eps[::-1])
# plt.yticks(np.arange(-len(g_modifier),1,1)+1,Eps[::-1])
# plt.ylabel('% inh')
# plt.xlabel('time (ms)')(
                            

    
    
#============================================ Bicuculline ====================================

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
          #'w':'E',
          'b':2}

Eps =[0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.93]#,0.94,0.95]#0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.93,
#
Gs =[9.0,9.0,8/2,7/3,1.0,3/7,2/8,7/93]#,6/94,5/95,4/96]#7/93,



#{'Vthr': 20,
    
Js = [16.0]#[7.2,9.0,14.4] ,14.4]# 6.7,6.8,7.0, 7.2
ex_rates =[0.09615384615384616/5]#,0.09615384615384616/2]#
Jexts = [8.0]#,3.0]#,6.7 - is not very nice
N = params['N']
g_modifier=[0,0.14285714,0.2,0.25,0.3,0.33333333,0.4,0.5,0.6,0.76923077,0.8,0.85,0.93023256,1]
#[1,0.85,0.8,0.6,0.4,0.3,0.2,0]#[0,0.1,0.25,0.35,0.5,0.75,0.8,0.9,1][::-1]# [0,0.1,0.25,0.35,0.5,0.75,0.9,1][::-1]
burst_map = np.zeros([len(Eps),len(g_modifier)])*np.nan
burstCV_map = np.zeros([len(Eps),len(g_modifier)])*np.nan
burstR_map =  np.zeros([len(Eps),len(g_modifier)])*np.nan
burstA_map =  np.zeros([len(Eps),len(g_modifier)])*np.nan
sns.set_style('ticks')
sns.set_context('poster')
plt.figure(figsize=(25,20))
S = []
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
                        params['ex_rate'] = (ex_rate/80)*((N-(epsilon*N))*0.1)#ex_rate
                        params['g'] =Gs[e_index]*g
                        try:
                            st,gid = read(params)
                            st= na(st)
                            gid= na(gid)
                            sim_time = 10000000#np.max(st)
                            bin_size = 20
                            #plt.figure(figsize=(25,3))
                            sc,_ = np.histogram(st,np.arange(0,sim_time,bin_size))
                            sc = sc[1:]
                            
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
                            
                            peakAmp = [np.max(signal[bI:bI+50]) for bI in bInd]
                            S.append(signal)
                            #peakAmp = signal[na(bInd)+3]

                           #peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)
                            #plt.plot(peakTimes_,peakAmp['peak_heights'],'*')
                            ibis = np.diff(bInd)*20/1000
                            if np.mean(peakAmp)<100:
                                # do not deletect single spikes
                                ibis = np.nan
                                peakAmp = np.nan
                            burst_map[e_index,g_i] = np.mean(ibis)
                            burstCV_map[e_index,g_i] = np.std(ibis)/np.mean(ibis)
                            burstR_map[e_index,g_i] =np.corrcoef(ibis,peakAmp[:-1])[0,1]
                            burstA_map[e_index,g_i] =np.mean(peakAmp)
                            
                            if False:
                                
                                plt.figure(figsize=(25,5))
                                plt.plot(signal); 
                                plt.plot(bInd,peakAmp,'*')
                                plt.title(g)
                                #plt.subplot(len(g_modifier),1,g_i+1)
                                #plt.plot(st,gid,'.',markersize = 2.1)
#                           plt.xlim([1000,300000])
                                #plt.plot(np.arange(0,250000-20,20)[1000:],sc)#-g_i*5)#-np.mean(sc))/np.std(sc))-(g_i*40))
                                plt.xlabel('time (ms)')
                                plt.ylabel('spikes')
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
                            

 # Export data
from func.ED_sim_ad import save_obj, load_obj
save_obj({'perc':Eps,
           'g_mod':g_modifier,
            'mIBI':burst_map,
            'cvIBI':burstCV_map,
            'aIBI':burstA_map,
            'rIBI':burstR_map,
            'Signal':S},
            'Figures/data/EI_modelN=1000_bic_t16s')




from func.ED_sim_ad import save_obj, load_obj
save_obj({'perc':Eps,
           'g_mod':g_modifier,
            'mIBI':burst_map,
            'cvIBI':burstCV_map,
            'aIBI':burstA_map,
            'rIBI':burstR_map,
            'Signal':S},
            'Figures/data/EI_modelN=1000_bic_t16s_full')




#====================================================================================================================



#============================================ Bicuculline 90% ====================================

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
          #'w':'E',
#           'seed':1234,
          'b':2}

Eps =[0.93]#,0.93]#[0.93,0.91,0.92,0.93,0.94,0.95,0.96]#0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.93,
#
Gs =[7/93]#,7/93]#[7/93,9/91,8/92,7/93,6/94,5/95,4/96]#9.0,9.0,8/2,7/3,1.0,3/7,2/8,7/93,



#{'Vthr': 20,
    
Js = [16.0]#[7.2,9.0,14.4] ,14.4]# 6.7,6.8,7.0, 7.2
ex_rates =[0.09615384615384616/5]#,0.09615384615384616/2]#
Jexts = [8.0]#,3.0]#,6.7 - is not very nice
N = params['N']
Taus = [0]

g_modifier=[1,0.85,0.8,0.6,0.4,0.3,0.2,0]#[0,0.1,0.25,0.35,0.5,0.75,0.8,0.9,1][::-1]# [0,0.1,0.25,0.35,0.5,0.75,0.9,1][::-1]
burst_map = np.zeros([len(Eps),len(g_modifier)])*np.nan
burstCV_map = np.zeros([len(Eps),len(g_modifier)])*np.nan
burstR_map =  np.zeros([len(Eps),len(g_modifier)])*np.nan
burstA_map =  np.zeros([len(Eps),len(g_modifier)])*np.nan
sns.set_style('ticks')
sns.set_context('poster')
plt.figure(figsize=(25,20))
S = []
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
                        params['ex_rate'] = (ex_rate/80)*((N-(epsilon*N))*0.1)#ex_rate
                        params['g'] =Gs[e_index]*g
                        try:
                            st,gid = read(params)
                            st= na(st)
                            gid= na(gid)
                            sim_time = 10000000#np.max(st)
                            bin_size = 20
                            #plt.figure(figsize=(25,3))
                            sc,_ = np.histogram(st,np.arange(0,sim_time,bin_size))
                            sc = sc[1:]
                            
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
                            
                            peakAmp = [np.max(signal[bI:bI+50]) for bI in bInd]
                            
                            S.append(signal)
                            #peakAmp = signal[na(bInd)+3]

                           #peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)
                            #plt.plot(peakTimes_,peakAmp['peak_heights'],'*')
                            ibis = np.diff(bInd)*20/1000
#                             if np.mean(peakAmp)<100:
                                # do not deletect single spikes
#                                 ibis = np.nan
#                                 peakAmp = np.nan
                            burst_map[e_index,g_i] = np.mean(ibis)
                            burstCV_map[e_index,g_i] = np.std(ibis)/np.mean(ibis)
                            burstR_map[e_index,g_i] =np.corrcoef(ibis,peakAmp[:-1])[0,1]
                            burstA_map[e_index,g_i] =np.mean(peakAmp)
                            
                            if True:
                                
                                plt.figure(figsize=(25,5))
                                plt.plot(signal); 
                                plt.plot(bInd,peakAmp,'*')
                                plt.title(g)
                                #plt.subplot(len(g_modifier),1,g_i+1)
                                #plt.plot(st,gid,'.',markersize = 2.1)
#                           plt.xlim([1000,300000])
                                #plt.plot(np.arange(0,250000-20,20)[1000:],sc)#-g_i*5)#-np.mean(sc))/np.std(sc))-(g_i*40))
                                plt.xlabel('time (ms)')
                                plt.ylabel('spikes')
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
                            

# # Export data
# from func.ED_sim_ad import save_obj, load_obj
# save_obj({'perc':Eps,
#           'g_mod':g_modifier,
#            'mIBI':burst_map,
#            'cvIBI':burstCV_map,
#            'aIBI':burstA_map,
#            'rIBI':burstR_map,
#            'Signal':S},
#            'Figures/data/EI_modelN=1000_bic_t16s')





#====================================================================================================================





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
    #plt.yscale('log')
    plt.figure()
    plt.hist(e_fr,density=True,color='r',alpha = 0.4)
    plt.hist(i_fr,density =True)

    print(len(st[gid[np.where(gid<NE)]])/(NE*(sim_time/1000)))
    return 'Done' 

j = 20.0
N=1000
ex_rate = 0.09615384615384616/80
epsilon = 0.2
params = {'Vthr': 20,
          'Vres': 10,
          'V0':0,
          'tau_ref': 2.0,
          'tau_m':40,
          'd':3.5,
          'N':1000,
          'epsilon':epsilon,
          'p':0.1,
          'g':((N-(epsilon*N))/(N*epsilon))*0,
          'J':(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1),#25/np.sqrt(25),#0.94868329805051377,
          'J_ext':20.0,
          'input':'NE+NI',
          'ex_rate':(ex_rate/80)*((N-(epsilon*N))*0.1),
          'eta':0.0,
          'tau_w':20000,
          'a':0.0,
          #'w':'E',
          'b':2}                    
sim_time= 1
get_properties(params,sim_time,thr='std')#,#plt_t=(17000,18000))
plot_eirates(params,sim_time)




# SIMPLE MODEL

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
burstA_map= np.zeros([len(Eps),2])

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
            #plt.figure()
            #plt.title(str([j, epsilon,params['g'],inh]))
            #plt.plot(st,gid,'.',markersize =3.5)
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
        burstA_map[e_index, inh] = np.mean(peakAmp['peak_heights'])

        
        
        
## G-stability 


# Inhibiton dominated networks
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
          'tau_w':20000,
          'a':0.0,
          #'w':'E',
          'b':2}

Eps =[0.2,0.8]#[0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.95,0.99]
#Eps =[0.05,0.2,0.5,0.8,0.95,]

#
#Gs =[9.0,8/2,1.0,2/8,5/95,]
Gs =[4.0,0.25]#[9.0,9.0,4.0,7/3,1.0,3/7,0.25,5/95,1/99]#

#{'Vthr': 20,
    
Taus = [20000]
Js =[12.,13.,15.,16.,17.,18.,19.,20.0,22.0]
# [15.0,16.0,17.,20.0,]#[7.2,9.0,14.4] ,14.4]# 6.7,6.8,7.0, 7.2
ex_rates =[0.09615384615384616/80]#,0.09615384615384616/2]#
Jexts = [20.0]#,3.0]#,6.7 - is not very nice
N = params['N']

g_modifier=[0,0.8,1.,1.1,1.25,1.5,2.0,3.0,4.0] #[1,4.0]#[0,0.5,1.0,1.5,2.0,4.0]#4.0]#[0,0.5,1,1.5]#,2.0,4.0]#[0.5,1.0,1.5]
burst_map = np.zeros([len(Eps),len(Js),len(g_modifier)])*np.nan
burstCV_map = np.zeros([len(Eps),len(Js),len(g_modifier)])*np.nan
burstR_map =  np.zeros([len(Eps),len(Js),len(g_modifier)])*np.nan
burstA_map =  np.zeros([len(Eps),len(Js),len(g_modifier)])*np.nan
sns.set_style('ticks')
sns.set_context('poster')
plt.figure(figsize=(25,20))

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
                        params['ex_rate'] = (ex_rate/80)*((N-(epsilon*N))*0.1)#ex_rate
                        params['g'] =Gs[e_index]*g
                        try:
                            st,gid = read(params)
                            st= na(st)
                            gid= na(gid)
                            sim_time = 250000#np.max(st)
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
                            peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)
                            #plt.plot(peakTimes_,peakAmp['peak_heights'],'*')
                            ibis = np.diff((peakTimes_*20)/1000)
                            burst_map[e_index,j_index,g_i] = np.mean(ibis)
                            burstCV_map[e_index,j_index,g_i] = np.std(ibis)/np.mean(ibis)
                            burstR_map[e_index,j_index,g_i] =np.corrcoef(ibis,peakAmp['peak_heights'][:-1])[0,1]
                            burstA_map[e_index,j_index,g_i] =np.mean(peakAmp['peak_heights'])
                            
                            if False:
                                
#                                 plt.figure(figsize=(25,3))
                                plt.subplot(len(g_modifier),1,g_i+1)
                                #plt.plot(st,gid,'.',markersize = 2.1)
#                           plt.xlim([1000,300000])
                                plt.plot(np.arange(0,250000-20,20)[1000:],sc)#-g_i*5)#-np.mean(sc))/np.std(sc))-(g_i*40))
                                plt.xlabel('time (ms)')
                                plt.ylabel('spikes')
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
                            
                                plt.title(['IPSP='+str(params['g'])+', J=%s'%params['J']])
                                
                        except:
                            continue
# sns.despine()
# plt.yticks(np.arange(-len(Eps),1,1)+1,Eps[::-1])
# plt.yticks(np.arange(-len(g_modifier),1,1)+1,Eps[::-1])
# plt.ylabel('% inh')
# plt.xlabel('time (ms)')(
plt.tight_layout()

for i in range(len(g_modifier)):
    plt.plot((na(Js)*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1),burstA_map[0,:,i],'o-',label='g=%s*NE/NI'%(g_modifier[i]));
# plt.plot((na(Js)*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1),burstA_map[0,:,0],'or',label='g=NE/NI');
#plt.yscale('log')
plt.xlabel('J'); plt.ylabel('A (spikes)'); sns.despine(); plt.legend(bbox_to_anchor=(1.1, 1.2))



for i in range(len(g_modifier)):
    plt.plot((na(Js)*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1),burstCV_map[0,:,i],'o-',label='g=%s*NE/NI'%(g_modifier[i]));
plt.axhline(CVdata[2],color='k',label = '20% inh. data')
plt.xlabel('J'); plt.ylabel('CV'); sns.despine(); plt.legend(bbox_to_anchor=(1.1, 0.9))



for i in range(len(g_modifier)):
    plt.plot((na(Js)*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1),burst_map[0,:,i],'o-',label='g=%s*NE/NI'%(g_modifier[i]));
plt.axhline(origIBI[2],color='k',label = '20% inh. data')
plt.yscale('log')
plt.xlabel('J'); plt.ylabel('IBI'); sns.despine(); plt.legend(bbox_to_anchor=(1., 1.0))





        
## ExInput-stability 


# Inhibiton dominated networks
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
          'tau_w':20000,
          'a':0.0,
          #'w':'E',
          'b':2}

Eps =[0.2]#[0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.95,0.99]
#Eps =[0.05,0.2,0.5,0.8,0.95,]

#
#Gs =[9.0,8/2,1.0,2/8,5/95,]
Gs =[4.]#[9.0,9.0,4.0,7/3,1.0,3/7,0.25,5/95,1/99]#

#{'Vthr': 20,
    
Taus = [20000]
Js =[20.]#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,15.0]
#[20.]
#[20.]
# [15.0,16.0,17.,20.0,]#[7.2,9.0,14.4] ,14.4]# 6.7,6.8,7.0, 7.2
ex_rates =[0.09615384615384616,0.09615384615384616/5,0.09615384615384616/10,0.09615384615384616/20,0.09615384615384616/40,0.09615384615384616/80, 0.09615384615384616/100]#,0.09615384615384616/2]#
Jexts =[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.,15.,20.]
#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,15.0,20.0,25.0]
#[20.0]#,3.0]#,6.7 - is not very nice
N = params['N']

g_modifier=[1.0]#[0,0.8,1.,1.1,1.25,1.5,2.0,3.0,4.0] #[1,4.0]#[0,0.5,1.0,1.5,2.0,4.0]#4.0]#[0,0.5,1,1.5]#,2.0,4.0]#[0.5,1.0,1.5]
burst_map = np.zeros([len(Eps),len(ex_rates),len(Jexts)])*np.nan
burstCV_map = np.zeros([len(Eps),len(ex_rates),len(Jexts)])*np.nan
burstR_map =  np.zeros([len(Eps),len(ex_rates),len(Jexts)])*np.nan
burstA_map =  np.zeros([len(Eps),len(ex_rates),len(Jexts)])*np.nan
sns.set_style('ticks')
sns.set_context('poster')
plt.figure(figsize=(25,20))

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
                        params['ex_rate'] = (ex_rate/80)*((N-(epsilon*N))*0.1)#ex_rate
                        params['g'] =Gs[e_index]*g
                        try:
                            st,gid = read(params)
                            st= na(st)
                            gid= na(gid)
                            sim_time = 800000#np.max(st)
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
                            peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)
                            #plt.plot(peakTimes_,peakAmp['peak_heights'],'*')
                            ibis = np.diff((peakTimes_*20)/1000)
                            burst_map[e_index,er_index,jex_index] = np.mean(ibis)
                            burstCV_map[e_index,er_index,jex_index] = np.std(ibis)/np.mean(ibis)
                            burstR_map[e_index,er_index,jex_index] =np.corrcoef(ibis,peakAmp['peak_heights'][:-1])[0,1]
                            burstA_map[e_index,er_index,jex_index] =np.mean(peakAmp['peak_heights'])
                        except:
                            continue
                        if False:

#                                 plt.figure(figsize=(25,3))
                            plt.subplot(len(ex_rates),1,er_index+1)
                            #plt.plot(st,gid,'.',markersize = 2.1)
#                           plt.xlim([1000,300000])
                            plt.plot(np.arange(0,250000-20,20)[1000:],sc)#-g_i*5)#-np.mean(sc))/np.std(sc))-(g_i*40))
                            plt.xlabel('time (ms)')
                            plt.ylabel('spikes')
#                                 plt.xticks(np.arange(0,250000/20,))
                            #plt.plot(((sc-np.mean(sc))/np.std(sc))-(g_i*40))
                            plt.axhline(thr)
                        #plt.yscale('log')
                        #tau =1000/bin_size
                        #signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')
                        #F0=np.percentile(signal,[55])
                        #plt.plot(((signal-np.median(signal))/np.median(signal))-(8*e_index))
                        #plt.plot((signal/np.max(signal))-e_index*1)
                        #plt.plot(((signal-np.mean(signal))/np.std(signal))-e_index*20)
                        #((signal-np.mean(signal))/np.std(signal))

                            plt.title(['rate='+str(params['ex_rate']*1000)+', Jex=%s'%params['J_ext']])

                            plt.tight_layout()


# sns.despine()
# plt.yticks(np.arange(-len(Eps),1,1)+1,Eps[::-1])
# plt.yticks(np.arange(-len(g_modifier),1,1)+1,Eps[::-1])
# plt.ylabel('% inh')
# plt.xlabel('time (ms)')(

sns.set_context('paper')
sns.heatmap(burst_map[0,:,:],xticklabels=Jexts,yticklabels=na(ex_rates)*1000,annot=True,
           cbar_kws={'label': 'IBI(s)'})
plt.xlabel('J_ex');
plt.ylabel('ex rate (Hz)')

sns.set_context('paper')
sns.heatmap(burstCV_map[0,:,:],xticklabels=Jexts,yticklabels=na(ex_rates)*1000,annot=True,
           cbar_kws={'label': 'CV'})
plt.xlabel('J_ex');
plt.ylabel('ex rate (Hz)')

sns.set_context('paper')
sns.heatmap(np.log(burstA_map[0,:,:]),xticklabels=Jexts,yticklabels=na(ex_rates)*1000,annot=True,
            
           cbar_kws={'label': 'log Amplitude'})
plt.xlabel('J_ex');
plt.ylabel('ex rate (Hz)')

sns.heatmap(burst_map[1,:,:],xticklabels=Jexts,yticklabels=na(ex_rates)*1000,annot=True)

sns.heatmap(burstCV_map[0,:,:],xticklabels=Jexts,yticklabels=na(ex_rates)*1000,annot=True)
sns.heatmap(burstCV_map[1,:,:],xticklabels=Jexts,yticklabels=na(ex_rates)*1000,annot=True)

sns.heatmap(np.log(burstA_map[0,:,:]),xticklabels=Jexts,yticklabels=na(ex_rates)*1000,annot=True)


for i in range(len(ex_rates)):
    plt.plot(Jexts,burst_map[0,i,:],'o-',label='rate=%s'%(ex_rates[i]));
# plt.plot((na(Js)*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1),burstA_map[0,:,0],'or',label='g=NE/NI');
#plt.yscale('log')
plt.xlabel('J'); plt.ylabel('A (spikes)'); sns.despine(); plt.legend(bbox_to_anchor=(1.1, 1.2))


for i in range(len(ex_rates)):
    plt.plot(Jexts,burst_map[0,i,:],'o-',label='rate=%s'%(ex_rates[i]));
# plt.plot((na(Js)*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1),burstA_map[0,:,0],'or',label='g=NE/NI');
#plt.yscale('log')
plt.xlabel('J'); plt.ylabel('IBI (spikes)'); sns.despine(); plt.legend(bbox_to_anchor=(1.1, 1.2))



for i in range(len(g_modifier)):
    plt.plot((na(Js)*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1),burstCV_map[0,:,i],'o-',label='g=%s*NE/NI'%(g_modifier[i]));
plt.axhline(CVdata[2],color='k',label = '20% inh. data')
plt.xlabel('J'); plt.ylabel('CV'); sns.despine(); plt.legend(bbox_to_anchor=(1.1, 0.9))



for i in range(len(g_modifier)):
    plt.plot((na(Js)*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1),burst_map[0,:,i],'o-',label='g=%s*NE/NI'%(g_modifier[i]));
plt.axhline(origIBI[2],color='k',label = '20% inh. data')
plt.yscale('log')
plt.xlabel('J'); plt.ylabel('IBI'); sns.despine(); plt.legend(bbox_to_anchor=(1., 1.0))



# Chech the fit
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
          'ex_rate':0,
          'eta':0.0,
          'tau_w':20000,
          'a':0.0,
          #'w':'E',
          'b':2}

Eps =[]#[0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.95,0.99]
#Eps =[0.05,0.2,0.5,0.8,0.95,]

#
#Gs =[9.0,8/2,1.0,2/8,5/95,]
Gs =[9.0,9.0,4.0,7/3,1.0,3/7,0.25,5/95,1/99]#

#{'Vthr': 20,
    
Taus = [20000]
Js =[10.]#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,15.0]
#[20.]
#[20.]
# [15.0,16.0,17.,20.0,]#[7.2,9.0,14.4] ,14.4]# 6.7,6.8,7.0, 7.2
ex_rates =[0.09615384615384616/5]#,0.09615384615384616/5,0.09615384615384616/10,0.09615384615384616/20,0.09615384615384616/40,0.09615384615384616/80, 0.09615384615384616/100]#,0.09615384615384616/2]#
Jexts =[8.]#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.,15.,20.]
#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,15.0,20.0,25.0]
#[20.0]#,3.0]#,6.7 - is not very nice
N = params['N']

g_modifier=[0,1.0]#[0,0.8,1.,1.1,1.25,1.5,2.0,3.0,4.0] #[1,4.0]#[0,0.5,1.0,1.5,2.0,4.0]#4.0]#[0,0.5,1,1.5]#,2.0,4.0]#[0.5,1.0,1.5]
burst_map = np.zeros([len(Eps),2])*np.nan
burstCV_map = np.zeros([len(Eps),2])*np.nan
burstR_map =  np.zeros([len(Eps),2])*np.nan
burstA_map =  np.zeros([len(Eps),2])*np.nan
sns.set_style('ticks')
sns.set_context('poster')
#plt.figure(figsize=(25,20))

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
                        params['ex_rate'] = (ex_rate/80)*((N-(epsilon*N))*0.1)#ex_rate
                        params['g'] =Gs[e_index]*g
                        try:
                            st,gid = read(params)
                            st= na(st)
                            gid= na(gid)
                            sim_time = 800000#np.max(st)
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
                            peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)
                            #plt.plot(peakTimes_,peakAmp['peak_heights'],'*')
                            ibis = np.diff((peakTimes_*20)/1000)
                            burst_map[e_index,g_i] = np.mean(ibis)
                            
                            burstCV_map[e_index,g_i] = np.std(ibis)/np.mean(ibis)
                            burstR_map[e_index,g_i] =np.corrcoef(ibis,peakAmp['peak_heights'][:-1])[0,1]
                            burstA_map[e_index,g_i] =np.mean(peakAmp['peak_heights'])
                        except:
                            continue
                        if True:

                            plt.figure(figsize=(25,4))
                            #plt.subplot(len(Eps),1,e_index+1)
                            #plt.plot(st,gid,'.',markersize = 2.1)
#                           plt.xlim([1000,300000])
                            #plt.plot(np.arange(0,sim_time-20,20)[1000:],sc)#-g_i*5)#-np.mean(sc))/np.std(sc))-(g_i*40))
                            plt.xlabel('time (ms)')
                            plt.ylabel('spikes')
#                                 plt.xticks(np.arange(0,250000/20,))
                            #plt.plot(((sc-np.mean(sc))/np.std(sc))-(g_i*40))
                            plt.axhline(thr)
                        #plt.yscale('log')
                        #tau =1000/bin_size
                        #signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')
                        #F0=np.percentile(signal,[55])
                        #plt.plot(((signal-np.median(signal))/np.median(signal))-(8*e_index))
                        #plt.plot((signal/np.max(signal))-e_index*1)
                        #plt.plot(((signal-np.mean(signal))/np.std(signal))-e_index*20)
                        #((signal-np.mean(signal))/np.std(signal))

                            plt.title([g,'%s'%epsilon +'rate='+str(params['ex_rate']*1000)+', Jex=%s'%params['J_ext']])

                            plt.tight_layout()


# sns.despine()
# plt.yticks(np.arange(-len(Eps),1,1)+1,Eps[::-1])
# plt.yticks(np.arange(-len(g_modifier),1,1)+1,Eps[::-1])
# plt.ylabel('% inh')
# plt.xlabel('time (ms)')(

# BICUCULLINE BETTER FOT 04/03/19
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
          'tau_w':18000,
          'a':0.0,
          #'w':'E',
          'b':2}

Eps =[0.05]#,0.1,0.2,0.3,0.5,0.7,0.8,0.95]#,0.99,0.994]

#[0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.93]#,0.99,0.994]
#Eps =[0.05,0.2,0.5,0.8,0.95,]

#
#Gs =[9.0,8/2,1.0,2/8,5/95,]
Gs =[19.0]#,9.0,4.0,7/3,1.0,3/7,0.25,5/95]

#{'Vthr': 20,
    
Taus = [15000]
Js = [20.]#[7.2,9.0,14.4] ,14.4]# 6.7,6.8,7.0, 7.2
ex_rates =[0.09615384615384616/5]#,0.09615384615384616/2]#
Jexts = [2.]#,3.0]#,6.7 - is not very nice
N = params['N']
#array([[ 12.85728814,  45.61125   ,  60.81      ,  68.878     ,
#         74.94666667]])
g_modifier=  [1,0]#[1,0.75 , 0.5, 0.25, 0]#,0.75 , 0.5, 0.25, 0]#[0,0.5,1.0,1.5,2.0,4.0]#4.0]#[0,0.5,1,1.5]#,2.0,4.0]#[0.5,1.0,1.5]
burst_map = np.zeros([len(Eps),len(g_modifier)])*np.nan
burstCV_map = np.zeros([len(Eps),len(g_modifier)])*np.nan
burstR_map =  np.zeros([len(Eps),len(g_modifier)])*np.nan
burstA_map =  np.zeros([len(Eps),len(g_modifier)])*np.nan
sns.set_style('ticks')
sns.set_context('poster')
plt.figure(figsize=(25,20))
S = []
for e_index,epsilon in enumerate(Eps):
    for tau_index, tau in enumerate(Taus):
        for j_index,j in enumerate(Js):
            for er_index,ex_rate in enumerate(ex_rates):
                for jex_index,jex in enumerate(Jexts):
                    for g_i,g in enumerate(g_modifier):
                        #if epsilon == 0.05:
                        #    j = 20
                        params['input'] = 'NE+NI'
                        params['epsilon']  = epsilon
                        params['J'] =(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1)
                        params['J_ext'] = jex#params['J']
                        params['ex_rate'] = (ex_rate/80)*((N-(epsilon*N))*0.1)#ex_rate
                        params['g'] =Gs[e_index]*g
                        try:
                            st,gid = read(params)
                            st= na(st)
                            gid= na(gid)
                            sim_time = 800000#np.max(st)
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
                            
                        except:
                            continue
                            
                        if True:

                            plt.figure(figsize=(25,5))
                            #plt.plot(signal); 
                            #plt.plot(sc)
                            #plt.plot(bInd,peakAmp,'*')
                            #plt.subplot(len(g_modifier),1,g_i+1)
                            plt.plot(st,gid,'.',markersize = 5.1)
#                           plt.xlim([1000,300000])
                            #plt.plot(np.arange(0,250000-20,20)[1000:],sc)#-g_i*5)#-np.mean(sc))/np.std(sc))-(g_i*40))
                            plt.xlabel('time (ms)')
                            plt.ylabel('spikes')
                            plt.title([str(epsilon)+'IPSP='+str(params['g'])])
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

# Export data
meanBIC_50 = np.array([ 13.95616667,  12.6476    ,  24.72323333,  34.9623    ,
        42.86503333,  54.607     ,  53.00023333])

bic50_ful = np.array([[ 13.6413,  17.464 ,  10.7632],
       [ 11.4962,  13.5155,  12.9311],
       [ 19.4773,  37.8755,  16.8169],
       [ 30.3645,  46.9571,  27.5653],
       [ 35.8989,  41.0701,  51.6261],
       [ 30.1722,  80.0995,  53.5493],
       [ 30.1402,  82.9932,  45.8673]])
good_fit50BIC = np.array([[ 12.85728814,  45.61125   ,  60.81      ,  68.878     ,
        74.94666667]])

c = [0,0.5,1,1.5,3.,10.,40.]
# bic_conc = [0.0,0.08,0.16,0.25,0.5,0.7,1]
Kd =3.0
bic_conc = 1- (1/(1+(na(c)/Kd)))
plt.figure()
plt.plot(bic_conc,meanBIC_50,'-k'); plt.plot(bic_conc,bic50_ful,'o-')

plt.plot(g_modifier[::-1],good_fit50BIC[0,:],'--k')
plt.plot(g_modifier[::-1],burst_map[0,:],'or')



#---Panel A----------
# Prepare the data 
plt.figure(figsize=(15,5))
meanBic = []
sem_bic = []
EpsData = [0.2,0.25,0.3,0.5,0.7,0.8]
mean_bic=[]
for i,epsilon in enumerate(EpsData):
    bic_ = na(pd.read_excel('Figures/data/bic_IBI_raw_corr.xlsx',str(epsilon)))[:7,1:]
    rel_bic = na(bic_)/bic_[0,:]
    n = rel_bic.shape[1]
    mean_bic.append(np.nanmean(rel_bic,1))
    sem_bic.append([np.std(rel_bic[k,:]/np.sqrt(n)) for k in range(0,len(rel_bic))])
popMean_bic = np.nanmean(mean_bic,0)

# Simulations 

EIData = load_obj('Figures/data/EI_modelN=1000_bic')

mIBI = EIData['mIBI']
Eps = EIData['perc']
g_mod = EIData['g_mod']
#mIBI = burst_map
#g_mod = g_modifier
#relIBI = mIBI/mIBI[0,:]
EIpopMean_bic= np.nanmean(relIBI[2:-1],0)
StaticBIC = na([8.9066666666666663, 2.6400000000000001, 1.46875])

#plot
# plt.subplot(grid[0, 0])
# bic_conc = [1, 0.8,0.7,0.5,0.3,0.2,0.0][::-1]
c = [0,0.5,1,1.5,3.,10.,40.]
# bic_conc = [0.0,0.08,0.16,0.25,0.5,0.7,1]
Kd =3.
bic_conc = 1- (1/(1+(na(c)/Kd)))
# bic_conc = (((1+na(g_mod)/Kd))/1)
#with sns.color_palette("Blues_d",n_colors=12):
col = sns.color_palette("Blues_d",n_colors=7)
[plt.errorbar(bic_conc,mean_bic[i],sem_bic[i],fmt='o-',color = col[i],capsize=4, elinewidth=3,label=EpsData[i]) for i in range(len(EpsData))]
plt.errorbar(bic_conc,mean_bic[1],sem_bic[1],fmt='o-',color = 'k',capsize=4, elinewidth=3,label=EpsData[1])
plt.legend(bbox_to_anchor=[1.1,1.1])
# for dat in mean_bic:
#     plt.plot(bic_conc,dat,'-',color ='red',alpha =0.2)
#plt.plot(bic_conc,popMean_bic/popMean_bic[0],'-o',linewidth = 4, color ='red',label ='Data 20-80% inh.')



# col = sns.color_palette('Reds',n_colors=7)
# for i in range(2,len(Eps)-1): # take onle 20-80 %
#     plt.plot(g_mod[::-1], mIBI[i,:],'o-',color= col[i],label=Eps[i])
# plt.legend(bbox_to_anchor=[1.1,1.1])

    #plt.plot(g_mod[::-1],EIpopMean_bic/EIpopMean_bic[0],'-o',linewidth = 4,markersize = 10,color ='C0',label ='model with adaptation')
col = sns.color_palette('Reds',n_colors=7)
for i in range(2,len(Eps)-1): # take onle 20-80 %
    plt.plot(g_mod[::-1], mIBI[i,:]/mIBI[i,:][0],'--o',color= col[i],label=Eps[i])
plt.plot(g_mod[::-1], mIBI[2,:]/mIBI[2,:][0],'--o',color= 'k',label=Eps[2])
plt.legend()
plt.plot([1, 0.8,0.7,0.5,0.3,0.2,0.0][::-1][:3], StaticBIC/StaticBIC[0],'-o', 
         color = 'gray',linewidth = 1,label = 'model without adaptation')
plt.legend(bbox_to_anchor=[1.1,1.1])

plt.legend(bbox_to_anchor=[1.1,1.1])


sns.despine()#trim=1)
plt.xticks()#fontsize = 15)
plt.yticks()#fontsize = 15)
plt.xlabel("log [Bic]",fontsize = 10)
plt.ylabel('change in IBI',fontsize = 10)

plt.legend()#loc='center left')#, bbox_to_