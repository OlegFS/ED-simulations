%matplotlib inline
import pandas as pd 
from func.processing_helpers import *
from func.ED_sim_ad import save_obj, load_obj
import seaborn as sns
sns.set_style('ticks')
sns.set_context('paper')
#preprocess data
EIData = load_obj('Figures/data/EI_modelN=1000_16')
StaticData=load_obj('Figures/data/Static_modelN=1000')



IBIs = pd.read_excel('Figures/data/meanIBI.xlsx','IBI')
sdIBI = pd.read_excel('Figures/data/meanIBI.xlsx','SD')
CVs = pd.read_excel('Figures/data/meanIBI.xlsx','CV')

#Panel A 

# Prepare data
staticIBIS = StaticData['mIBI']
staticEPS =  StaticData['perc']
eiIBIS = EIData['mIBI']
eiEPS =  EIData['perc']

plt.figure(figsize=(5,5))
# Static model lines
plt.plot(na(staticEPS)*100,staticIBIS,'o-',color = 'gray')
plt.plot(na(eiEPS)*100,eiIBIS,'o-')
plt.xticks(np.arange(0,101,20))
sns.despine(trim=0)
plt.xlabel('Inhibitory percentage (%)', fontsize = 9)

#plt.savefig('figs/mean_IBI_fit.eps')

#Panel B


#Panel C
# Prepare data 
staticCV = StaticData['cvIBI']
staticEPS =  StaticData['perc']
eiCV = EIData['cvIBI']
eiEPS =  EIData['perc']

plt.figure(figsize=(5,5))
# Static model lines
plt.plot(na(staticEPS)*100,staticCV,'o-',color = 'gray')
# EI model line
plt.plot(na(eiEPS)*100,eiCV,'o-')
plt.xticks(np.arange(0,101,20))
sns.despine(trim=0)
plt.xlabel('Inhibitory percentage (%)', fontsize = 9)




#Panel D


#Pamel E
S = EIData['Signal']
Eps = EIData['perc']
loc = []
plt.figure(figsize=(14.4,16.4))
with sns.color_palette("Blues_d",n_colors=12):

    for i,signal in enumerate(S):
        plt.plot(signal/np.max(S)-i*0.6,linewidth =1)
        loc.append(-i*0.6)
        plt.xlim([10000,32500])
    plt.yticks(loc,Eps)
    plt.xticks(np.arange(10000,32501,2500),np.arange(0,22501,2500)/1000*20)
    plt.xlabel('time(s)',fontsize = 20)
    plt.ylabel('Inhibitory percentage',fontsize = 20)
    

    

# BIG plot with everything rendered in Python
with sns.color_palette("Reds_d",n_colors=12):
    
    fig = plt.figure(figsize=(15,5))
    grid = plt.GridSpec(2, 4, wspace=0.4, hspace=0.3)


    # --------Panel A----------
    plt.subplot(grid[0, 0])
    
    
    #Data

    
    stds = [np.sqrt(np.nanmean(na(sdIBI)[:,i]**2))/ np.sqrt(len( np.isfinite(na(sdIBI)[:,i]))) for i in range(9)]
    means = [np.nanmean(na(IBIs)[:,i]) for i in range(9) ]
    staticIBIS = StaticData['mIBI']
    staticEPS =  StaticData['perc']
    eiIBIS = EIData['mIBI']
    eiEPS =  EIData['perc']
    eiEPS[-1] =1;
    eiEPS[0] = 0
    EpsData = [0,10,20,25,30,50,70,80,100]
    
    #plot
    # DATA
    plt.errorbar(EpsData,means,stds,fmt='ro-',alpha =0.4, label = 'data')
    #STATIC MODEL
    plt.plot(na(staticEPS)*100,staticIBIS,'o-',color = 'gray',label='Model without adapatation')
    #EI MODEL
    plt.plot(na(eiEPS)*100,eiIBIS,'o-',color='C0',label='Model with adapatation')
    plt.xticks(np.arange(0,101,20))
    sns.despine(trim=0)
    plt.xlabel('Inhibitory percentage (%)', fontsize = 10)
    plt.ylabel('Inter-burst intervals (s)', fontsize = 10)
    plt.legend(fontsize=8)
    # --------Panel B----------
    plt.subplot(grid[0, 1])
    #load data
        #load data
    corrs = na(pd.read_excel('Figures/data/Correltation_manual.xlsx'))
    mean_R = np.nanmean(corrs)
    std_R = np.nanstd(corrs)/np.sqrt(np.sum(np.isfinite(corrs)))
    
    
    #plt.errorbar(0,mean_R,std_R ,fmt='o-',color='red',alpha =0.5);#plt.yscale('log')
    ;#plt.yscale('log')
    # Static model lines
    staticR = StaticData['rIBI']
    staticEPS =  StaticData['perc']
    eiR = EIData['rIBI']
    eiEPS =  EIData['perc']
    #plot
#     plt.errorbar(1,mean_R,std_R ,fmt='o-',color='red',alpha =0.5)
#     plt.plot(na(staticEPS)*100,staticR,'o-',color = 'gray')
    #plt.errorbar(2,np.mean(eiR),np.std(eiR)/np.sqrt(8),fmt ='o-',color='C0')
    #plt.errorbar(3,np.mean(staticR),np.std(staticR)/np.sqrt(6),fmt ='o-',color='C0')
#     plt.xticks(np.arange(0,101,20))
    sns.despine(trim=0)
    plt.xlabel('Inhibitory percentage (%)', fontsize = 10)
    plt.ylabel('Correlation coefficient', fontsize = 10)



    # --------Panel C----------
    plt.subplot(grid[1, 0])
    # Prepare data 
    staticCV = StaticData['cvIBI']
    staticEPS =  StaticData['perc']
    eiCV = EIData['cvIBI']
    eiEPS =  EIData['perc']
    # Static model lines
    [plt.plot([perc]*len(na(CVs)[:,i]), na(CVs)[:,i],'.r',alpha =0.5) for i,perc in enumerate(EpsData)]
    plt.plot(na(staticEPS)*100,staticCV,'o-',color = 'gray')
    # EI model line
    plt.plot(na(eiEPS)*100,eiCV,'o-',color='C0')
    plt.xticks(np.arange(0,101,20))
    sns.despine(trim=0)
    plt.xlabel('Inhibitory percentage (%)', fontsize = 9)

    plt.ylabel('Coefficient of variation', fontsize = 10)


    # --------Panel D----------
    #load data
    amps = na(pd.read_excel('Figures/data/amp_all_export.xlsx'))
    mean_amp = np.nanmean(amps,0)
    std_amp = np.nanstd(amps/mean_amp[2],0)/np.sqrt(np.sum(np.isfinite(amps),0))
    
    plt.subplot(grid[1, 1])
    # Prepare data 
    staticA = StaticData['aIBI']
    staticEPS =  StaticData['perc']
    eiA = EIData['aIBI']
    eiEPS =  EIData['perc']
    # data
    plt.errorbar(EpsData,mean_amp/mean_amp[2],std_amp ,fmt='o-',color='red',alpha =0.5, label ='data');#plt.yscale('log')
    # Static model lines
    plt.plot(na(staticEPS)*100,staticA/staticA[2],'o-',color = 'gray', label ='model without adaptation')
    # EI model line
    plt.plot(na(eiEPS)*100,eiA/eiA[2],'o-',color='C0',label ='model with adaptation')
    plt.xticks(np.arange(0,101,20))
    plt.yscale('log')
    sns.despine(trim=0)
    plt.xlabel('Inhibitory percentage (%)', fontsize = 9)
    plt.ylabel('Amplitude (a.u.)', fontsize = 10)

    #plt.legend()

# --------Panel E----------
    plt.subplot(grid[:, 2:]);
    # Prepare data
    S = EIData['Signal']
    Eps = EIData['perc']
    loc = []

    for i,signal in enumerate(S):
        plt.plot(((signal/np.sqrt(np.mean(signal)))/mean_amp[0])-i*500)

        #plt.plot((signal/EIData['perc'][i])-i*10000,linewidth =1)
        
        loc.append(-i*500)
        plt.xlim([15000,32500])
        
    plt.yticks(loc,list(map(int,na(Eps)*100)))
    plt.xticks(np.arange(15000,32501,2500),np.arange(0,17501,2500)/1000*20)
    
    plt.xlabel('time(s)',fontsize = 10)
    plt.ylabel('Inhibitory percentage',fontsize = 10)
    
 
plt.savefig('Figures/figure4.eps',fmt = 'eps')   
#     plt.tight_layout()