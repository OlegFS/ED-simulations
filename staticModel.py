%matplotlib inline
import pandas as pd 
from func.processing_helpers import *
import scipy.io.savemat as savemat

#Burst detection

st,gid = read(params)
bin_size=20
sc,_ = np.histogram(st,np.arange(0,1000000,20))
tau =2000/bin_size
signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')



plt.plot((signal-np.min(signal)/np.min(signal)));
plt.plot(signal);
bins = np.arange(np.min(signal),np.max(signal),np.std(signal))
plt.yscale('log')

Eps = [0.2,0.3,0.5,0.7,0.8,0.95]#the best so far
Js = [7.4,9.1,12.,14.87,17.8,29.]
Js = [7.4,(6,7),10,12,14.,20.]#ex_rate = 0.0007

j = 6.
N=1000
ex_rate =0.0007#*80
#0.0016025641025641027#0.09615384615384616/60
epsilon = 0.5
g = 1.0
params = {'Vthr': 20,
          'Vres': 10,
          'V0':0,
          'tau_ref': 2.0,
          'tau_m':40,
          'd':3.5,
          'N':1000,
          'epsilon':epsilon,
          'p':0.1,
          'g':(((N-(epsilon*N))/(N*epsilon)))*g,
          'J':(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1),#,j,#(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1),#25/np.sqrt(25),#0.94868329805051377,
          'J_ext':10.,#(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1),
          'input':'NE+NI',
          'ex_rate':(ex_rate/5)*((N-(epsilon*N))*0.1),
          'eta':0.0,
          'tau_w':20000,
          'a':0.0,
          #'w':'E',
          'b':0}      
sim_time= 1
# get_properties(params,sim_time,thr='NE')#,plt_t=(0,190000))
st,gid = read(params)
bin_size=20
sc,_ = np.histogram(st,np.arange(0,1000000,20))
tau =2000/bin_size
signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')
NE = params['N']-(params['epsilon']*params['N'])
peakTimes_,peakAmp = find_peaks(sc,height=NE,width=0.5,distance=50)
bInd = giveBurstInd(signal)
peakAmp = [np.max(signal[bI:bI+100]) for bI in bInd]

plt.figure(figsize=(15,5)); 
# plt.plot(sc);
# plt.yscale('log')
plt.plot(signal); 
plt.plot(bInd,peakAmp,'*')#np.ones(len(bInd)),'*')#
# newpeakAmp = [np.max(signal[bI:bI+100]) for bI in new_bInd]
# plt.plot(new_bInd,newpeakAmp,'*r' )
print(np.mean(np.diff(bInd)*20/1000))
# plt.xlim([7200,7500])

# print(len(st)/params['N']/(np.max(st)/1000))
# plt.figure(figsize=( 15,4))

# plt.plot(st,gid,'.',markersize = 1.1)
# plt.xlim([0,5000])
# plot_eirates(params,sim_time)




##### Collect the data ####

Eps = [0.2,0.3,0.5,0.7,0.8,0.95]#the best so far
Js = [7.4,9.1,12.,14.87,17.8,20]


N=1000
ex_rate =0.0004#*80
#0.0016025641025641027#0.09615384615384616/60
g = 1.0
mIBI = []
cvIBI = []
aIBI = []
rIBI = []
S = []
for e_i,epsilon in enumerate(Eps):
    j = Js[e_i]
    if epsilon==0.95:
        ex_rate = 0.0007
    else:
        ex_rate =0.0004#*80

        
    params = {'Vthr': 20,
              'Vres': 10,
              'V0':0,
              'tau_ref': 2.0,
              'tau_m':40,
              'd':3.5,
              'N':1000,
              'epsilon':epsilon,
              'p':0.1,
              'g':(((N-(epsilon*N))/(N*epsilon)))*g,
              'J':(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1),#,j,#(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1),#25/np.sqrt(25),#0.94868329805051377,
              'J_ext':10.,#(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1),
              'input':'NE+NI',
              'ex_rate':(ex_rate/5)*((N-(epsilon*N))*0.1),
              'eta':0.0,
              'tau_w':20000,
              'a':0.0,
              #'w':'E',
              'b':0}      
    sim_time= 1
    # get_properties(params,sim_time,thr='NE')#,plt_t=(0,190000))
    st,gid = read(params)
    bin_size=20
    sc,_ = np.histogram(st,np.arange(0,3000000,20))
    tau =2000/bin_size
    signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')
    NE = params['N']-(params['epsilon']*params['N'])
    peakTimes_,peakAmp = find_peaks(sc,height=NE,width=0.5,distance=50)
    bInd = giveBurstInd(signal)
    peakAmp = [np.max(signal[bI:bI+100]) for bI in bInd]

    plt.figure(figsize=(15,5)); 
    # plt.plot(sc);
    # plt.yscale('log')
    plt.plot(signal); 
    plt.plot(bInd,peakAmp,'*')#np.ones(len(bInd)),'*')#
    # newpeakAmp = [np.max(signal[bI:bI+100]) for bI in new_bInd]
    # plt.plot(new_bInd,newpeakAmp,'*r' )
    print(np.mean(np.diff(bInd)*20/1000))
    ibis = np.diff(bInd)*20/1000
    mIBI.append(np.mean(ibis))
    cvIBI.append(np.std(ibis)/np.mean(ibis))
    aIBI.append(np.mean(peakAmp))
    rIBI.append(np.corrcoef(peakAmp[:-1],ibis)[0,1])
    S.append(signal)
    
    
from func.ED_sim_ad import save_obj, load_obj
save_obj({'perc':Eps,
                         'mIBI':mIBI,
                         'cvIBI':cvIBI,
                         'aIBI':aIBI,
                         'rIBI':rIBI,
                         'Signal':S},
                     'Figures/data/Static_modelN=1000')




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
# plt.yscale('log')

plt.errorbar(na([5,10,20,30,50,70,80,95]),np.nanmean(na(BIC_sat),0),
             np.nanstd(na(BIC_sat),0),fmt= 'o--',color= 'darkorange',label='data: max [BIC]')
plt.errorbar(na(origEps)*100,origIBI,stds,color = 'darkorange',label='data: no BIC')
plt.plot(na(Eps)[np.isfinite(burst_map[:,0])]*100,burst_map[np.isfinite(burst_map[:,0]),0],'-o',color='C0',label='model'); 
plt.plot(na(Eps)[np.isfinite(burst_map[:,1])]*100,burst_map[np.isfinite(burst_map[:,1]),1],'--o',color='C0',label='model: no inh'); 
plt.legend(bbox_to_anchor=(1.1, 0.9))
plt.yscale('log')
plt.xlabel('% inh.')
plt.ylabel('IBI (s)')




# SCAN


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
          'b':0}

Eps =[0.2,0.3,0.5,0.7,0.8,0.95,0.99]
#Eps =[0.05,0.2,0.5,0.8,0.95,]

#
#Gs =[9.0,8/2,1.0,2/8,5/95,]
Gs =[4.0,7/3,1.0,3/7,0.25,5/95,1/99]#

#{'Vthr': 20,
    
Taus = [20000]
Js =[14.,15.,16.,17.,18.]#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,15.0]
#[20.]
#[20.]
# [15.0,16.0,17.,20.0,]#[7.2,9.0,14.4] ,14.4]# 6.7,6.8,7.0, 7.2
ex_rates =[0.09615384615384616/60,0.09615384615384616/70,0.09615384615384616/80,0.09615384615384616/90,0.09615384615384616/100]
    
ex_rate = np.sort(ex_rates)
Jexts =[20.]#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.,15.,20.]
#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,15.0,20.0,25.0]
#[20.0]#,3.0]#,6.7 - is not very nice
N = params['N']

g_modifier=[1.25]#[0,0.8,1.,1.1,1.25,1.5,2.0,3.0,4.0] #[1,4.0]#[0,0.5,1.0,1.5,2.0,4.0]#4.0]#[0,0.5,1,1.5]#,2.0,4.0]#[0.5,1.0,1.5]

burst_map = np.zeros([len(Eps),len(ex_rates),len(Js)])*np.nan
burstCV_map = np.zeros_like(burst_map)*np.nan
burstR_map =  np.zeros_like(burst_map)*np.nan
burstA_map =  np.zeros_like(burst_map)*np.nan
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
                        params['ex_rate'] = (ex_rate/120)*((N-(0.8*N))*0.1)#ex_rate
                        params['g'] =Gs[e_index]*g
                        try:
                            st,gid = read(params)
                            st= na(st)
                            gid= na(gid)
                            sim_time = 100000#np.max(st)
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
                            thr = np.std(sc)*5
                            peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)
                            #plt.plot(peakTimes_,peakAmp['peak_heights'],'*')
                            ibis = np.diff((peakTimes_*20)/1000)
                            burst_map[e_index,er_index,j_index] = np.mean(ibis)
                            
                            burstCV_map[e_index,er_index,j_index] = np.std(ibis)/np.mean(ibis)
                            burstR_map[e_index,er_index,j_index] =np.corrcoef(ibis,peakAmp['peak_heights'][:-1])[0,1]
                            burstA_map[e_index,er_index,j_index] =np.mean(peakAmp['peak_heights'])
                        except:
                            continue
                        if False:

                            plt.figure(figsize=(25,5))
                            #plt.subplot(len(Eps),1,e_index+1)
                            plt.plot(st,gid,'.',markersize = 0.5)
#                           plt.xlim([1000,300000])
                            #plt.plot(np.arange(0,sim_time-20,20)[1000:],sc)#-g_i*5)#-np.mean(sc))/np.std(sc))-(g_i*40))
                            plt.xlabel('time (ms)')
                            plt.ylabel('spikes')
#                                 plt.xticks(np.arange(0,250000/20,))
                           # plt.plot(((sc-np.mean(sc))/np.std(sc))-(g_i*40))
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


#----------------------------------------FIT                   
                            
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
          'b':0}

Eps =[0.2,0.25,0.5,0.75,0.77,0.8,0.85,0.95,0.99]#[0.2,0.3,0.5,0.7,0.8,0.95,0.99]
#Eps =[0.05,0.2,0.5,0.8,0.95,]

#
#Gs =[9.0,8/2,1.0,2/8,5/95,]
#Gs =[4.0,3.,1.,1/3,2/8,5/95,1/99]#[8/2,7/3,1.0,3/7,0.25,5/95,1/99]#

#{'Vthr': 20,
    
Taus = [20000]
Js =[19.]#,15.,16.,17.,18.]#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,15.0]
#[20.]
#[20.]
# [15.0,16.0,17.,20.0,]#[7.2,9.0,14.4] ,14.4]# 6.7,6.8,7.0, 7.2
# ex_rates =[0.0016025641025641027]#[0.00040064102564102574]#[0.09615384615384616/60]#,0.09615384615384616/70,0.09615384615384616/80,0.09615384615384616/90,0.09615384615384616/100] 
ex_rates = [0.000004]#[0.09615384615384616/80]
# ex_rates = [0.0016025641025641027]
    
Jexts =[20.0]#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.,15.,20.]
#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,15.0,20.0,25.0]
#[20.0]#,3.0]#,6.7 - is not very nice
N = params['N']

g_modifier=[1.]#[0,0.8,1.,1.1,1.25,1.5,2.0,3.0,4.0] #[1,4.0]#[0,0.5,1.0,1.5,2.0,4.0]#4.0]#[0,0.5,1,1.5]#,2.0,4.0]#[0.5,1.0,1.5]

burst_map = np.zeros([len(Eps),2])*np.nan
burstCV_map = np.zeros_like(burst_map)*np.nan
burstA_map = np.zeros_like(burst_map)*np.nan
#burstR_map =  np.zeros_like(burst_map)*np.nan
#burstA_map =  np.zeros_like(burst_map)*np.nan
sns.set_style('ticks')
sns.set_context('poster')
#plt.figure(figsize=(25,20))
m_ibis = []
cv_ibis = []

for e_index,epsilon in enumerate(Eps):
    for tau_index, tau in enumerate(Taus):
        for j_index,j in enumerate(Js):
            for er_index,ex_rate in enumerate(ex_rates):
                for jex_index,jex in enumerate(Jexts):
                    for g_i,g in enumerate(g_modifier):
                        params['input'] = 'NE+NI'
                        params['epsilon']  = epsilon
                        params['J'] =(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1)
                        #(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1)
                        params['J_ext'] =jex# (j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1)#jex#params['J']
                        params['ex_rate'] = (ex_rate/5)*((N-(epsilon*N))*0.1)#
                        #(ex_rate/120)*((N-(epsilon*N))*0.1)#(ex_rate/80)*((N-(0.8*N))*0.1)#ex_rate
                        #300
                        print(params['ex_rate'])
                        params['g'] =(((N-(epsilon*N))/(N*epsilon)))*g
                        try:
                            st,gid = read(params)
                            st= na(st)
                            gid= na(gid)
                            sim_time = 700000#np.max(st)
                            bin_size = 20
                            #plt.figure(figsize=(25,3))
                            sc,_ = np.histogram(st,np.arange(0,sim_time,bin_size))
                            #sc = sc[1000:]
                            
                            bin_size = 20
                            NE = N-(epsilon*N)
                            if NE >50:
                                thr = NE/3
                            else:
                                thr =50
                            thr = np.std(sc)*5
                            if epsilon<0.3:
                                
                                peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=100)
                            else:
                                peakTimes_,peakAmp = find_peaks(sc,height=thr,width=0.5,distance=50)
                            #plt.plot(peakTimes_,peakAmp['peak_heights'],'*')
                            ibis = np.diff((peakTimes_*20)/1000)
#                             m_ibis.append(np.mean(ibis))
                            m_ibis.append(len(st)/N/(sim_time/1000))
                            cv_ibis.append(np.std(ibis)/np.mean(ibis))
                            burst_map[e_index,0] = np.mean(ibis)
                            burstCV_map[e_index,0] = np.std(ibis)/np.mean(ibis)
                            
                            #burstCV_map[e_index,er_index,j_index] = np.std(ibis)/np.mean(ibis)
                            #burstR_map[e_index,er_index,j_index] =np.corrcoef(ibis,peakAmp['peak_heights'][:-1])[0,1]
                            burstA_map[e_index,0] =np.mean(peakAmp['peak_heights'])
                        except:
                            continue
                        if True:
                            print('a')
                            plt.figure(figsize=(35,5))
                            #plt.subplot(len(Eps),1,e_index+1)
                            plt.plot(st,gid,'.',markersize = 2.5)
#                           plt.xlim([1000,300000])
#                             plt.plot(sc)
                            #plt.yscale('log')
                            #plt.plot(np.arange(0,sim_time-20,20)[1000:],sc)#-g_i*5)#-np.mean(sc))/np.std(sc))-(g_i*40))
#                             plt.plot(peakTimes_,peakAmp['peak_heights'],'*r')
                            plt.xlabel('time (ms)')
                            plt.ylabel('spikes')
#                                 plt.xticks(np.arange(0,250000/20,))
                           # plt.plot(((sc-np.mean(sc))/np.std(sc))-(g_i*40))
                            #plt.axhline(thr)
                        #plt.yscale('log')
                            tau =1000/bin_size
                            signal  = np.convolve(sc,np.exp(-np.arange(10000/bin_size)/tau),'same')
                            #plt.plot(signal)
                        #F0=np.percentile(signal,[55])
                        #plt.plot(((signal-np.median(signal))/np.median(signal))-(8*e_index))
                        #plt.plot((signal/np.max(signal))-e_index*1)
                        #plt.plot(((signal-np.mean(signal))/np.std(signal))-e_index*20)
                        #((signal-np.mean(signal))/np.std(signal))
                            plt.title(['%s%% inh.  '%na(epsilon*100) +'ex rate='+str(na(params['ex_rate'])*1000)+', J=%s'%params['J']])
                            plt.tight_layout()

                            
                            
# j~1/K scaling
                            

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
          'b':0}

Eps =[0.25,0.5,0.75]#i0.2,0.3,0.5,0.7,0.8,0.95,0.99]
#Eps =[0.05,0.2,0.5,0.8,0.95,]
    #
#Gs =[9.0,8/2,1.0,2/8,5/95,]
Gs =[3.,1.,1/3]#[8/2,7/3,1.0,3/7,0.25,5/95,1/99]#

#{'Vthr': 20,
    
Taus = [20000]
Js =[25.]#,15.,16.,17.,18.]#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,15.0]
#[20.]
#[20.]
# [15.0,16.0,17.,20.0,]#[7.2,9.0,14.4] ,14.4]# 6.7,6.8,7.0, 7.2
# ex_rates =[0.0016025641025641027]#[0.00040064102564102574]#[0.09615384615384616/60]#,0.09615384615384616/70,0.09615384615384616/80,0.09615384615384616/90,0.09615384615384616/100] 
# ex_rates = [0.09615384615384616/80]
ex_rates = [0.000002]#[0.0016025641025641027]
    
Jexts =[20.]#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.,15.,20.]
#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,15.0,20.0,25.0]
#[20.0]#,3.0]#,6.7 - is not very nice
N = params['N']

g_modifier=[1.5]#[0,0.8,1.,1.1,1.25,1.5,2.0,3.0,4.0] #[1,4.0]#[0,0.5,1.0,1.5,2.0,4.0]#4.0]#[0,0.5,1,1.5]#,2.0,4.0]#[0.5,1.0,1.5]

#burst_map = np.zeros([len(Eps),len(ex_rates),len(Js)])*np.nan
#burstCV_map = np.zeros_like(burst_map)*np.nan
#burstR_map =  np.zeros_like(burst_map)*np.nan
#burstA_map =  np.zeros_like(burst_map)*np.nan
sns.set_style('ticks')
sns.set_context('poster')
#plt.figure(figsize=(25,20))
m_ibis = []
cv_ibis = []

for e_index,epsilon in enumerate(Eps):
    for tau_index, tau in enumerate(Taus):
        for j_index,j in enumerate(Js):
            for er_index,ex_rate in enumerate(ex_rates):
                for jex_index,jex in enumerate(Jexts):
                    for g_i,g in enumerate(g_modifier):
                        params['input'] = 'NE+NI'
                        params['epsilon']  = epsilon
                        params['J'] =(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1)
                        #(j*20)/((N-(epsilon*N))*0.1)
                        #(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1)
                        #
                        #
                        params['J_ext'] = jex#params['J']
                        params['ex_rate'] = (ex_rate/5)*((N-(epsilon*N))*0.1)#(ex_rate/80)*((N-(0.8*N))*0.1)#ex_rate
                        #300
                        print(params['ex_rate'])
                        params['g'] =Gs[e_index]*g
                        try:
                            st,gid = read(params)
                            st= na(st)
                            gid= na(gid)
                            sim_time = 800000#np.max(st)
                            bin_size = 20
                            #plt.figure(figsize=(25,3))
                            sc,_ = np.histogram(st,np.arange(0,sim_time,bin_size))
                            sc = sc[2500:]
                            
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
                            m_ibis.append(np.mean(ibis))
                            print('a')
                            cv_ibis.append(np.std(ibis)/np.mean(ibis))
                            #burst_map[e_index,er_index,j_index] = np.mean(ibis)
                            
                            #burstCV_map[e_index,er_index,j_index] = np.std(ibis)/np.mean(ibis)
                            #burstR_map[e_index,er_index,j_index] =np.corrcoef(ibis,peakAmp['peak_heights'][:-1])[0,1]
                            #burstA_map[e_index,er_index,j_index] =np.mean(peakAmp['peak_heights'])
                        except:
                            continue
                        if True:
                            plt.figure(figsize=(25,5))
                            #plt.subplot(len(Eps),1,e_index+1)
                            plt.plot(st,gid,'.',markersize = 0.5)
#                           plt.xlim([1000,300000])
                            #plt.plot(sc)
                            #plt.plot(np.arange(0,sim_time-20,20)[1000:],sc)#-g_i*5)#-np.mean(sc))/np.std(sc))-(g_i*40))
                            #plt.plot(peakTimes_,peakAmp['peak_heights'],'*r')
                            plt.xlabel('time (ms)')
                            plt.ylabel('spikes')
#                                 plt.xticks(np.arange(0,250000/20,))
                           # plt.plot(((sc-np.mean(sc))/np.std(sc))-(g_i*40))
                           # plt.axhline(thr)
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
                            
                            
                            
                            
# Transition 


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
          'b':0}

Eps =[0.25,0.5,0.75]#[0.25,0.5,0.75]#i0.2,0.3,0.5,0.7,0.8,0.95,0.99]
#Eps =[0.05,0.2,0.5,0.8,0.95,]
    #
#Gs =[9.0,8/2,1.0,2/8,5/95,]
Gs =[3.,1.0,1/3]#[3.,1.,1/3]#[8/2,7/3,1.0,3/7,0.25,5/95,1/99]#

#{'Vthr': 20,
    
Taus = [20000]
Js =np.arange(15,25)#[25.]#,15.,16.,17.,18.]#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,15.0]
#[20.]
#[20.]
# [15.0,16.0,17.,20.0,]#[7.2,9.0,14.4] ,14.4]# 6.7,6.8,7.0, 7.2
# ex_rates =[0.0016025641025641027]#[0.00040064102564102574]#[0.09615384615384616/60]#,0.09615384615384616/70,0.09615384615384616/80,0.09615384615384616/90,0.09615384615384616/100] 
# ex_rates = [0.09615384615384616/80]
ex_rates = [0.000002]#[0.0016025641025641027]
    
Jexts =[20.]#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.,15.,20.]
#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,15.0,20.0,25.0]
#[20.0]#,3.0]#,6.7 - is not very nice
N = params['N']

g_modifier=[1,1.25,1.5]#[0,0.8,1.,1.1,1.25,1.5,2.0,3.0,4.0] #[1,4.0]#[0,0.5,1.0,1.5,2.0,4.0]#4.0]#[0,0.5,1,1.5]#,2.0,4.0]#[0.5,1.0,1.5]

burst_map = np.zeros([len(Eps),len(Js),len(g_modifier)])*np.nan
burstCV_map = np.zeros_like(burst_map)*np.nan
burstR_map =  np.zeros_like(burst_map)*np.nan
burstA_map =  np.zeros_like(burst_map)*np.nan
rate_map =  np.zeros_like(burst_map)*np.nan
sns.set_style('ticks')
sns.set_context('poster')
#plt.figure(figsize=(25,20))
m_ibis = []
cv_ibis = []

for e_index,epsilon in enumerate(Eps):
    for tau_index, tau in enumerate(Taus):
        for j_index,j in enumerate(Js):
            for er_index,ex_rate in enumerate(ex_rates):
                for jex_index,jex in enumerate(Jexts):
                    for g_i,g in enumerate(g_modifier):
                        params['input'] = 'NE+NI'
                        params['epsilon']  = epsilon
                        params['J'] =(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1)
                        #(j*20)/((N-(epsilon*N))*0.1)
                        #(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1)
                        #
                        #
                        params['J_ext'] = jex#params['J']
                        params['ex_rate'] = (ex_rate/5)*((N-(epsilon*N))*0.1)#(ex_rate/80)*((N-(0.8*N))*0.1)#ex_rate
                        #300
                        print(params['ex_rate'])
                        params['g'] =Gs[e_index]*g
                        try:
                            st,gid = read(params)
                            st= na(st)
                            gid= na(gid)
                            sim_time = 800000#np.max(st)
                            bin_size = 20
                            #plt.figure(figsize=(25,3))
                            sc,_ = np.histogram(st,np.arange(0,sim_time,bin_size))
                            #sc = sc[1000:]
                            
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

                            print('a')

                            burst_map[e_index,j_index,g_i] = np.mean(ibis)
                            rate_map[e_index,j_index,g_i] = len(st)/N/(np.max(st)/1000)
                            
                            burstCV_map[e_index,j_index,g_i] = np.std(ibis)/np.mean(ibis)
                            burstR_map[e_index,j_index,g_i] =np.corrcoef(ibis,peakAmp['peak_heights'][:-1])[0,1]
                            burstA_map[e_index,j_index,g_i] =np.mean(peakAmp['peak_heights'])
                        except:
                            continue
                        if False:
                            plt.figure(figsize=(25,5))
                            #plt.subplot(len(Eps),1,e_index+1)
                            plt.plot(st,gid,'.',markersize = 0.5)
#                           plt.xlim([1000,300000])
                            #plt.plot(sc)
                            #plt.plot(np.arange(0,sim_time-20,20)[1000:],sc)#-g_i*5)#-np.mean(sc))/np.std(sc))-(g_i*40))
                            #plt.plot(peakTimes_,peakAmp['peak_heights'],'*r')
                            plt.xlabel('time (ms)')
                            plt.ylabel('spikes')
#                                 plt.xticks(np.arange(0,250000/20,))
                           # plt.plot(((sc-np.mean(sc))/np.std(sc))-(g_i*40))
                           # plt.axhline(thr)
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
                            

                            
sns.set_context('talk')
plt.figure(figsize=(25,5))
for i in range(3):
    print(Eps[i])
    plt.subplot(1,3,i+1)
    sns.heatmap(rate_map[i,:,:],annot=True,xticklabels=g_modifier,yticklabels=Js,
               cbar_kws={'label': 'rate (Hz)'})
    plt.title('%s inh. %%'%(Eps[i]*100))
    plt.xlabel('g multiplier')
    plt.ylabel('J for K=5')
                            

# Transition for 25 % inh. 

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
          'b':0}

Eps =[0.25]#,0.5,0.75]#[0.25,0.5,0.75]#i0.2,0.3,0.5,0.7,0.8,0.95,0.99]
#Eps =[0.05,0.2,0.5,0.8,0.95,]
    #
#Gs =[9.0,8/2,1.0,2/8,5/95,]
Gs =[3.]#,1.0,1/3]#[3.,1.,1/3]#[8/2,7/3,1.0,3/7,0.25,5/95,1/99]#

#{'Vthr': 20,
    
Taus = [20000]
Js =np.arange(15,25)#[25.]#,15.,16.,17.,18.]#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,15.0]
#[20.]
#[20.]
# [15.0,16.0,17.,20.0,]#[7.2,9.0,14.4] ,14.4]# 6.7,6.8,7.0, 7.2
# ex_rates =[0.0016025641025641027]#[0.00040064102564102574]#[0.09615384615384616/60]#,0.09615384615384616/70,0.09615384615384616/80,0.09615384615384616/90,0.09615384615384616/100] 
# ex_rates = [0.09615384615384616/80]
ex_rates = [0.000002]#[0.0016025641025641027]
    
Jexts =[19.0]#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.,15.,20.]
#[2.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,15.0,20.0,25.0]
#[20.0]#,3.0]#,6.7 - is not very nice
N = params['N']

g_modifier=[1,1.25,1.5]#[0,0.8,1.,1.1,1.25,1.5,2.0,3.0,4.0] #[1,4.0]#[0,0.5,1.0,1.5,2.0,4.0]#4.0]#[0,0.5,1,1.5]#,2.0,4.0]#[0.5,1.0,1.5]

burst_map = np.zeros([len(Eps),len(Js),len(g_modifier)])*np.nan
burstCV_map = np.zeros_like(burst_map)*np.nan
burstR_map =  np.zeros_like(burst_map)*np.nan
burstA_map =  np.zeros_like(burst_map)*np.nan
rate_map =  np.zeros_like(burst_map)*np.nan
sns.set_style('ticks')
sns.set_context('poster')
#plt.figure(figsize=(25,20))
m_ibis = []
cv_ibis = []

for e_index,epsilon in enumerate(Eps):
    for tau_index, tau in enumerate(Taus):
        for j_index,j in enumerate(Js):
            for er_index,ex_rate in enumerate(ex_rates):
                for jex_index,jex in enumerate(Jexts):
                    for g_i,g in enumerate(g_modifier):
                        params['input'] = 'NE+NI'
                        params['epsilon']  = epsilon
                        params['J'] =(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1)
                        #(j*20)/((N-(epsilon*N))*0.1)
                        #(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1)
                        #
                        #
                        params['J_ext'] = jex#params['J']
                        params['ex_rate'] = (ex_rate/5)*((N-(epsilon*N))*0.1)#(ex_rate/80)*((N-(0.8*N))*0.1)#ex_rate
                        #300
                        print(params['ex_rate'])
                        params['g'] =Gs[e_index]*g
                        try:
                            st,gid = read(params)
                            st= na(st)
                            gid= na(gid)
                            sim_time = 800000#np.max(st)
                            bin_size = 20
                            #plt.figure(figsize=(25,3))
                            sc,_ = np.histogram(st,np.arange(0,sim_time,bin_size))
                            sc = sc[2500:]
                            
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

                            print('a')

                            burst_map[e_index,j_index,g_i] = np.mean(ibis)
                            rate_map[e_index,j_index,g_i] = len(st)/N/(np.max(st)/1000)
                            
                            burstCV_map[e_index,j_index,g_i] = np.std(ibis)/np.mean(ibis)
                            burstR_map[e_index,j_index,g_i] =np.corrcoef(ibis,peakAmp['peak_heights'][:-1])[0,1]
                            burstA_map[e_index,j_index,g_i] =np.mean(peakAmp['peak_heights'])
                        except:
                            continue
                        if False:
                            plt.figure(figsize=(25,5))
                            #plt.subplot(len(Eps),1,e_index+1)
                            plt.plot(st,gid,'.',markersize = 0.5)
#                           plt.xlim([1000,300000])
                            #plt.plot(sc)
                            #plt.plot(np.arange(0,sim_time-20,20)[1000:],sc)#-g_i*5)#-np.mean(sc))/np.std(sc))-(g_i*40))
                            #plt.plot(peakTimes_,peakAmp['peak_heights'],'*r')
                            plt.xlabel('time (ms)')
                            plt.ylabel('spikes')
#                                 plt.xticks(np.arange(0,250000/20,))
                           # plt.plot(((sc-np.mean(sc))/np.std(sc))-(g_i*40))
                           # plt.axhline(thr)
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
                            

                            
sns.set_context('talk')
plt.figure(figsize=(25,5))
for i in range(1):
    print(Eps[i])
    plt.subplot(1,3,i+1)
    sns.heatmap(rate_map[i,:,:],annot=True,xticklabels=g_modifier,yticklabels=Js,
               cbar_kws={'label': 'rate (Hz)'})
    plt.title('%s inh. %%'%(Eps[i]*100))
    plt.xlabel('g multiplier')
    plt.ylabel('J for K=5')