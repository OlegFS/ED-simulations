%matplotlib inline
import pandas as pd 
from func.processing_helpers import *
from func.ED_sim_ad import save_obj, load_obj
import seaborn as sns
sns.set_style('ticks')
sns.set_context('paper')
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

import matplotlib
# matplotlib.use('PDF')
import matplotlib.pylab as plt
from matplotlib import rc
plt.rcParams['ps.useafm'] = True
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rcParams['pdf.fonttype'] = 42
#preprocess data
#EIData = load_obj('Figures/data/EI_modelN=1000')

#Panel A 
# prepare the data 
# Cultures
meanBic = []
Eps = [0.2]#[0.1,0.2,0.3,0.5,0.7,0.8,0.95]
mean_bic=[ ]
for i,epsilon in enumerate(Eps):
    bic_ = na(pd.read_excel('Figures/data/bic_IBI_raw_corr.xlsx',str(epsilon)))[:7,1:]
    mean_bic.append(np.nanmean(bic_,1))
popMean_bic = np.nanmean(mean_bic,0)

# Simulations 
EIData = load_obj('Figures/data/EI_modelN=5000_bic')

mIBI = EIData['mIBI']
Eps = EIData['perc']
g_mod = EIData['g_mod']
EIpopMean_bic= np.nanmean(mIBI[2:-1],0)
StaticBIC = na([8.9066666666666663, 2.6400000000000001, 1.46875])





#plt.savefig('figs/BIC_all_later.eps',fmt ='eps')
def tsplot(x,y,std,marker= '-o',**kw):
    x = x
    est = y
    sd = std
    cis = (est - sd, est + sd)
    plt.fill_between(x,cis[0],cis[1],alpha=0.2, **kw)
    plt.plot(x,est,marker,**kw)
    plt.margins(x=0)



fig = plt.figure(figsize=(11,5))
grid = plt.GridSpec(1, 2, wspace=0.2, hspace=0.1)


# --------Panel A----------
# Prepare the data 
# plt.figure(figsize=(5,5))

plt.subplot(grid[0, 0])
meanBic = []
sem_bic = []
EpsData = [0.25,0.2,0.3,0.5,0.7,0.8]
choice = 0
mean_bic=[]
bic_sat = []
for i,epsilon in enumerate(EpsData):
    bic_ = na(pd.read_excel('Figures/data/bic_IBI_raw_corr.xlsx',str(epsilon)))[:7,1:]
    rel_bic = na(bic_)/bic_[0,:]
    n = rel_bic.shape[1]
    mean_bic.append(np.nanmean(rel_bic,1))
    sem_bic.append([np.std(rel_bic[k,:])/np.sqrt(n) for k in range(0,len(rel_bic))])#
    
popMean_bic = np.nanmean(mean_bic,0)
# Simulations 
EIData = load_obj('Figures/data/EI_modelN=1000_bic_t16s')
mIBI = EIData['mIBI']
Eps = EIData['perc']
g_mod = EIData['g_mod'][::-1]
relIBI = [mIBI[i,]/mIBI[i,0] for i in range(mIBI.shape[0])]
EIpopMean_bic= np.nanmean(relIBI[2:-1],0)
StaticBIC = na([8.9066666666666663, 2.6400000000000001, 1.46875])
c = [0,0.5,1,1.5,3.,10.,40.]
Kd =3
bic_conc = 1-(1/(1+(na(c)/Kd)))
c_fromG = ((1/(1-na(g_mod)))*Kd)-Kd
c_fromG[0] = 60
#with sns.color_palette("Blues_d",n_colors=12):
col = sns.color_palette("winter",n_colors=6)
# plt.figure(figsize=(5,5))

# plt.savefig('Figures/figure6A_Data_contr.pdf',fmt = 'pdf')  


# plt.figure(figsize=(5,5))
col = sns.color_palette('winter',n_colors=7)
for i in range(1,len(EpsData)):
#         plt.figure(figsize=(5,5))
        plt.errorbar(na(c),mean_bic[i],sem_bic[i],fmt='o',color = col[i],capsize=8,markersize = 8, elinewidth=3,label=EpsData[i],alpha =1)
        plt.xscale('symlog')
        plt.xticks(c,c)
        plt.legend()
        plt.ylim([0,7])
#         plt.savefig('Figures/figure6A_Data_Eps%s.pdf'%(i),fmt = 'pdf')  
plt.errorbar(c,mean_bic[choice],sem_bic[choice],fmt='o-',color = 'darkblue',capsize=8, linewidth=3,label='control',markersize = 8)
plt.xticks(c,c)
plt.legend()
plt.ylim([0,7])
plt.xscale('symlog')
plt.xticks(c,c)

# sns.despine()
plt.ylim([0,7])
#plt.savefig('Figures/figure6A_Data_Eps_legend_and_axes.pdf',fmt = 'pdf')  
# [tsplot(na(c),mean_bic[i],sem_bic[i],'') for i in range(2,len(EpsData))]
plt.xscale('symlog')
plt.xticks(c,c)#fontsize = 15)
plt.yticks()#fontsize = 15)
plt.ylabel('change in IBI',fontsize = 18)
# plt.legend()
plt.ylim([0,7])
# plt.subplot(1,2,2)
# for dat in mean_bic:
#     plt.plot(bic_conc,dat,'-',color ='red',alpha =0.2)
#plt.plot(bic_conc,popMean_bic/popMean_bic[0],'-o',linewidth = 4, color ='red',label ='Data 20-80% inh.')
col = sns.color_palette('Reds_r',n_colors=8)
# for i in range(2,len(Eps)-1): # take onle 20-80 %
#     plt.plot(g_mod[::-1], mIBI[i,:],'o-',color= col[i],label=Eps[i])
# plt.legend()
#     #plt.plot(g_mod[::-1],EIpopMean_bic/EIpopMean_bic[0],'-o',linewidth = 4,markersize = 10,color ='C0',label ='model with adaptation')

#plt.figure(figsize=(5,5))

col = sns.color_palette('rocket',n_colors=7)
#plt.savefig('Figures/figure6A_Eps%s.pdf'%(0),fmt = 'pdf')  
# mask = [0,2,3,5,7,9,11,13]
# for i in range(3,len(Eps)-1): # take onle 20-80 %
#         #plt.figure(figsize=(5,5))
#         plt.plot(c_fromG[:], mIBI[i,:]/mIBI[i,:][-1],'--s',color= col[i], #/mIBI[i,:][0]
#                label=Eps[i],alpha =0.7,markersize = 8)
#         plt.xscale('symlog')
#         plt.xticks(c,g_mod)
#         plt.ylim([0,7])
# choice =4
plt.plot(c_fromG[:], mIBI[choice+2,:]/mIBI[choice+2,-1],'--s',color= 'r',label=Eps[choice+2],linewidth=3,markersize = 8)#mIBI[choice+1,:][0]
plt.xscale('symlog')
plt.ylim([0,7])
plt.xticks(c,c)
        #plt.savefig('Figures/figure6A_Eps%s.pdf'%(i),fmt = 'pdf')  
#/mIBI[choice+2,:][0]##

plt.plot(c_fromG[::-1][0:3], StaticBIC/StaticBIC[0],'-s', 
          color = 'gray',markersize = 8,linewidth = 3,label ='no adaptation' )
plt.plot(c_fromG[::-1][2],(StaticBIC/StaticBIC[0])[2],'*',color='gray',markersize = 18,markeredgecolor = 'k')
# plt.text(0.9, 0.45, 'no adaptation',fontsize = 10 ,color = 'gray')

plt.ylim([0,7])
# plt.xscale('symlog')
sns.despine()#trim=1)
# plt.xticks(c,c)#fontsize = 15)
# plt.xticks(c,np.round(bic_conc,2))#fontsize = 15)
plt.yticks(fontsize = 12)
# plt.yscale('symlog')
# plt.xlabel("Decrease in IPSP (%)",fontsize = 10)
plt.ylabel('log change in IBI',fontsize = 18)
plt.xlabel("log [Bic]",fontsize = 18)
#loc='center left')#,
plt.ylim([0,7])
plt.xticks(c,c,fontsize = 12)
plt.legend(ncol=2)
# plt.legend(bbox_to_anchor=(1, 0.5))
plt.tight_layout()

# plt.savefig('Figures/figure6A.pdf',fmt = 'pdf')   
#------------Panel B---------------------

#____________Panel B_______________________


# plt.subplot(1,2,2)
plt.subplot(grid[0, 1])
#plt.figure(figsize=(5,5))
#Prepare the data 

meanBic = []
EpsData = [0.05,0.1,0.95]
CD = ['peru','navy','red'][::-1]
CM = ['C0','C0','C0']
meanBic = []
sem_bic = []
mean_bic=[]
for i,epsilon in enumerate(EpsData):
    bic_ = na(pd.read_excel('Figures/data/bic_IBI_raw_corr.xlsx',str(epsilon)))[:7,1:]
    rel_bic = na(bic_)/na(bic_)[0,:]
    n = rel_bic.shape[1]
    mean_bic.append(np.nanmean(rel_bic,1))
    sem_bic.append([np.nanstd(rel_bic[k,:]/np.sqrt(n)) for k in range(0,len(rel_bic))])
popMean_bic = np.nanmean(mean_bic,0)

# Simulations 
mIBI = EIData['mIBI']
Eps = EIData['perc']
g_mod = EIData['g_mod']
relIBI =[mIBI[i,]/mIBI[i,0] for i in range(mIBI.shape[0])] 
EIpopMean_bic= np.nanmedian(relIBI[2:-1],0)
StaticBIC = na([8.9066666666666663, 2.6400000000000001, 1.46875])

col = sns.color_palette("Paired",8)#sns.color_palette("winter",n_colors=3)

col_ind = [1,5,7]
c = na(c)
sem_bic = na(sem_bic)
for i in range(len(EpsData)):
    x = c[np.isfinite(mean_bic[i])]
    y = mean_bic[i][np.isfinite(mean_bic[i])]
    er = sem_bic[i][np.isfinite(mean_bic[i])]
    plt.errorbar(x,y,er,fmt='o',color = col[col_ind[i]], elinewidth=3,label=EpsData[i],alpha =1,
                 capsize=8,markersize = 8) 
plt.xscale('symlog')
plt.legend()
col_ind = [0,4,6]
for ind,i in enumerate([0,1,-1]):
    plt.plot(c_fromG[:], mIBI[i,:]/mIBI[i,-1],'--sr',color= col[col_ind[ind]],label=Eps[i],alpha =1,linewidth=3,markersize = 8)
plt.yscale('symlog')


sns.despine()#trim=1)
plt.xticks(c,c,fontsize = 12)
# plt.yticks([1,10],[1,5,10],fontsize = 12)#fontsize = 15)
plt.xlabel("log [Bic]",fontsize = 18)
plt.ylabel('log Change in IBI',fontsize = 18)

plt.legend(ncol=2)#linewidth=3,markersize = 8



plt.savefig('Figures/figure6B.pdf',fmt = 'pdf')   
plt.savefig('Figures/figure6B.png',fmt = 'png')   

# Panel B












#____________FIT SIGMOIDAL_______

from scipy.optimize import curve_fit
def sigmoidal(x,gamma,m,c):
    return (1/(1+np.exp(-gamma*(x-m))))

def fover(x,a,b,c):
    return c/(a-(b*x))
def line(x,a,b):
    return (a*x)+b

EpsData = [0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.95]
choice = 0
mean_bic=[]
bic_sat = []
for i,epsilon in enumerate(EpsData):
    bic_ = na(pd.read_excel('Figures/data/bic_IBI_raw_corr.xlsx',str(epsilon)))[:7,1:]
    rel_bic = na(bic_)/bic_[0,:]
    n = rel_bic.shape[1]
    mean_bic.append(np.nanmean(rel_bic,1))
    sem_bic.append([np.std(rel_bic[k,:])/np.sqrt(n) for k in range(0,len(rel_bic))])#
  


popt, pcov = curve_fit(sigmoidal, na(g_mod[::-1])*10, mIBI[2]/np.max(mIBI[2]),method='lm')


plt.plot(na(g_mod[::-1])*10,sigmoidal(na(g_mod[::-1])*10,popt[0],popt[1]))
plt.plot(na(g_mod[::-1])*10, mIBI[2]/np.max(mIBI[2]))

est_paramsDat = []
EpsData = [0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.95]
for i,epsilon in enumerate(EpsData):
    bic_ = na(pd.read_excel('Figures/data/bic_IBI_raw_corr.xlsx',str(epsilon)))[:7,1:]
    y = na(bic_)/bic_[0,:]
    y =y.astype('float16')
    x = bic_conc#na(g_mod[::-1])*10
    valid = np.isfinite(np.nanmean(y,1))
    y /=np.nanmean(y[valid,:][-1,:].astype('int16'))
    x = np.repeat(x,y.shape[1])
    y = y.flatten()
#     y = y.max()
    valid =np.isfinite(y)
    popt, pcov = curve_fit(sigmoidal, x[valid],y[valid],method='lm')#(1/(1+np.exp(-gamma*(x-m)))) /np.max(ibis)
    est_paramsDat.append(popt)
    plt.plot(x,sigmoidal(x,popt[0],popt[1],popt[2]))

    plt.plot(x ,y,'.')

est_params = []
for ibis in mIBI:
    x =na(g_mod[::-1])#*10 
    x = x[np.isfinite(ibis)]
    y = ibis[np.isfinite(ibis)]
    y = y/y[0]
    y/=np.max(y)
    popt, pcov = curve_fit(sigmoidal, x, y,method='lm')
    est_params.append(popt)
    plt.plot(x,sigmoidal(x,popt[0],popt[1],popt[2]))
    plt.plot(x, y,'--')

#Comparison 
def lms(y_,y):
    return np.mean((y-y_)**2)

Error = []
BestError = []
Var = []
for i,ibis in enumerate(mean_bic):
    x = na(bic_conc)*10
    x = x[np.isfinite(ibis)]
    y = ibis[np.isfinite(ibis)]
    y = y/np.max(y)
    y_ = sigmoidal(x,est_params[i][0],est_params[i][1],est_params[i][2])
    
    y_best = sigmoidal(x,est_paramsDat[i][0],est_paramsDat[i][1],est_params[i][2])
    Error.append(lms(y_,y ))
    BestError.append(lms(y_best,y))
    
Error = []
BestError = []
Var = []
EpsData = [0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.95]
for i,epsilon in enumerate(EpsData):
    bic_ = na(pd.read_excel('Figures/data/bic_IBI_raw_corr.xlsx',str(epsilon)))[:7,1:]
    y = na(bic_)/bic_[0,:]
    y =y.astype('float16')
    x = bic_conc#na(g_mod[::-1])*10
    y /=np.nanmax(y,0)
    x = np.repeat(x,y.shape[1])
    y = y.flatten()
    valid =np.isfinite(y)
    x = x[valid]
    y = y[valid]
    y_ = sigmoidal(x,est_params[i][0],est_params[i][1],est_params[i][2])
    y_best = sigmoidal(x,est_paramsDat[i][0],est_paramsDat[i][1],est_params[i][2])
    Error.append(lms(y_,y))
    BestError.append(lms(y_best,y))
    
    
    
    

plt.figure(figsize=(5,5));

plt.plot(EpsData,na(Error),'s-',label ='prediction from the model',
        linewidth=3,markersize = 8)
plt.plot(EpsData,BestError,'s-',color = 'gray',label =r'$1/(1+(-\gamma(x-\mu)))$ fit',
        linewidth=3,markersize = 8)
# plt.xticks(EpsData)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

sns.despine(trim=0)
plt.legend(fontsize=14)
plt.xlabel('% of inhibitory neurons',fontsize=14)
plt.ylabel('Mean squared error',fontsize=14)
plt.tight_layout()
plt.savefig('figs/BICsigmoidal_fit.pdf',fmt='pdf')
plt.savefig('figs/BICsigmoidal_fit.png',fmt='png',dpi=100)
# plt.yscale('log')


# plt.plot(na(BestError)/na(Error))
# p
# Error directly from the model
Error = []
BestError = []
Var = []
EpsData = [0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.95]
g_mask = [1,4,6,8,10,12,13]# Value corresponding to the data
for i,epsilon in enumerate(EpsData):
    bic_ = na(pd.read_excel('Figures/data/bic_IBI_raw_corr.xlsx',str(epsilon)))[:7,1:]
    y = na(bic_)/bic_[0,:]
    y =y.astype('float16')
    x = bic_conc#na(g_mod[::-1])*10
    y /=np.nanmax(y,0)
    rep_n = y.shape[1]
    x = np.repeat(x,rep_n)
    y = y.flatten()
    valid =np.isfinite(y)
    x = x[valid]
    y = y[valid]
#     y_ = sigmoidal(x,est_params[i][0],est_params[i][1],est_params[i][2])
    y_ = mIBI[i][g_mask][::-1]
    y_ /=y_[0]
    y_/=np.max(y_)
    y_ =np.repeat(y_, rep_n)
    y_ = y_[valid]
    
    y_best = sigmoidal(x,est_paramsDat[i][0],est_paramsDat[i][1],est_params[i][2])
    Error.append(lms(y_,y))
    BestError.append(lms(y_best,y))
    
    



import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

a = np.cos(2*np.pi*np.linspace(0, 1, 60.))

ax1.plot(range(60), a)
ax2.plot(range(100), np.ones(100)) # Create a dummy plot
ax2.cla()
plt.show()


#
mean_bic=[ ]
for i,epsilon in enumerate(Eps_data):
    bic_ = na(pd.read_excel('Figures/data/bic_IBI_raw_corr.xlsx',str(epsilon)))[:7,1:]
    mean_bic.append(np.nanmean(bic_,1))

# Simulations 
EIData = load_obj('Figures/data/EI_modelN=1000_bic')

mIBI = EIData['mIBI']
Eps = EIData['perc']
g_mod = EIData['g_mod']
StaticBIC = na([8.9066666666666663, 2.6400000000000001, 1.46875])


#plot
plt.subplot(grid[0, 1])
bic_conc = na([1, 0.8,0.7,0.5,0.3,0.2,0.0][::-1])
for i,dat in enumerate(mean_bic):
    dat = na(dat)
    plt.plot(bic_conc[np.isfinite(dat)],dat[np.isfinite(dat)]/dat[0],'-',color=CD[i],label ='%s %% inh. data '%na(Eps_data[i]*100),linewidth = 1.7)


for i in [0,1,-1]: # take onle 0 and 100%
    
     plt.plot(g_mod[::-1], mIBI[i,:]/mIBI[i,:][0],'--',color = CD[i],linewidth = 1.7,label ='model with adpatation')#/mIBI[i,:][0]
    



sns.despine()#trim=1)
plt.xticks()#fontsize = 15)
plt.yticks()#fontsize = 15)
plt.xlabel("log [Bic]",fontsize = 10)
plt.ylabel('change in IBI',fontsize = 10)

plt.legend()#loc='center left')#, bbox_to_anchor=(1, 0.5))
