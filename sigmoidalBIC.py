%matplotlib inline
import pandas as pd 
from func.processing_helpers import *
from func.ED_sim_ad import save_obj, load_obj
import seaborn as sns
sns.set_style('ticks')
sns.set_context('paper')
#preprocess data
#EIData = load_obj('Figures/data/EI_modelN=1000')

#Panel A 
# prepare the data 
# Cultures

plt.figure(figsize=(5,5))
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
EIData = load_obj('Figures/data/EI_modelN=1000_bic_t18s')
mIBI = EIData['mIBI']
Eps = EIData['perc']
g_mod = EIData['g_mod']
relIBI = [mIBI[i,]/mIBI[i,0] for i in range(mIBI.shape[0])]
EIpopMean_bic= np.nanmean(relIBI[2:-1],0)
StaticBIC = na([8.9066666666666663, 2.6400000000000001, 1.46875])
c = [0,0.5,1,1.5,3.,10.,40.]
Kd =3
bic_conc = 1-(1/(1+(na(c)/Kd)))

c_fromG = ((1/(1-na(g_mod)))*Kd)-Kd
c_fromG[0] = 40
# bic_conc = (((1+na(g_mod)/Kd))/1)
#with sns.color_palette("Blues_d",n_colors=12):
col = sns.color_palette("Blues_d",n_colors=6)
[plt.errorbar(c,mean_bic[i],sem_bic[i],fmt='-o',color = col[i],capsize=8,markersize = 8, elinewidth=3,label=EpsData[i],alpha =1) for i in range(len(EpsData))]
choice = 1
plt.errorbar(c,mean_bic[choice],sem_bic[choice],fmt='-o',color = 'darkblue',capsize=8, linewidth=3,label=EpsData[choice],markersize = 8)
plt.xscale('symlog')
# for dat in mean_bic:
#     plt.plot(bic_conc,dat,'-',color ='red',alpha =0.2)
#plt.plot(bic_conc,popMean_bic/popMean_bic[0],'-o',linewidth = 4, color ='red',label ='Data 20-80% inh.')
# col = sns.color_palette('Reds',n_colors=7)
# for i in range(2,len(Eps)-1): # take onle 20-80 %
#     plt.plot(g_mod[::-1], mIBI[i,:],'o-',color= col[i],label=Eps[i])
# plt.legend()
#     #plt.plot(g_mod[::-1],EIpopMean_bic/EIpopMean_bic[0],'-o',linewidth = 4,markersize = 10,color ='C0',label ='model with adaptation')
col = sns.color_palette('Reds',n_colors=7)
for i in range(2,len(Eps)-1): # take onle 20-80 %
    plt.plot(c_fromG[::-1], mIBI[i,:]/mIBI[i,:][0],'--o',color= col[i],
             label=Eps[i],alpha =0.7,markersize = 8,markerfacecolor = [1,1,1])
plt.plot(c_fromG[::-1], mIBI[choice+1,:]/mIBI[choice+1,:][0],'--o',color= 'r',label=Eps[choice+1],linewidth=3,markersize = 8)

plt.plot(c_fromG[::-1][0:3], StaticBIC/StaticBIC[0],'-o', 
         color = 'gray',linewidth = 1,label = 'model without adaptation')
sns.despine()#trim=1)
plt.xticks(c,c)#fontsize = 15)
plt.yticks()#fontsize = 15)
plt.xlabel("log [Bic]",fontsize = 10)
plt.ylabel('change in IBI',fontsize = 10)

plt.legend(bbox_to_anchor=[1.0,1.0])#loc='center left')#, bbox_to_anchor=(1, 0.5))



# Fit sigmoidal 
def fsigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a*(x-b)))

EpsData = [0.1]#[0.2,0.25,0.3,0.5,0.7,0.8]
mean_bic=[]
for i,epsilon in enumerate(EpsData):
    bic_ = na(pd.read_excel('Figures/data/bic_IBI_raw_corr.xlsx',str(epsilon)))[:7,1:]
    rel_bic = na(bic_)/bic_[0,:]
    #n = rel_bic.shape[1]
    #mean_bic.append(np.nanmean(rel_bic,1))
    #sem_bic.append([np.std(rel_bic[k,:]/np.sqrt(n)) for k in range(0,len(rel_bic))])

from scipy.optimize import curve_fit
bic_conc = 1-(1/(1+(na(c)/Kd)))
log_c = np.log(c)
log_c[0] = -1
popt, pcov = curve_fit(fsigmoid,np.repeat(log_c,rel_bic.shape[1]),np.concatenate(rel_bic)/np.max(rel_bic))


y = fsigmoid(log_c, *popt)

plt.plot(log_c,y)
plt.plot(np.repeat(log_c,rel_bic.shape[1]),np.concatenate(rel_bic)/np.max(rel_bic),'.');
