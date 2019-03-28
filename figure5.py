%matplotlib inline
import pandas as pd 
from func.processing_helpers import *
from func.ED_sim_ad import save_obj, load_obj
import seaborn as sns
sns.set_style('ticks')
sns.set_context('paper')
#preprocess data
#EIData = load_obj('Figures/data/EI_modelN=1000')

g_set = []
for g in [0, 0.1,0.25,0.5,0.75,0.9,1,1.1,1.25,1.5,1.75,1.9,2.,]:
    if g == 0:
        g_set.append(load_obj('Figures/data/fig5/fig5_noInh'))
    elif g==1:
        g_set.append(load_obj('Figures/data/fig5/fig5_withInh'))
    else:
        g_set.append(load_obj('Figures/data/fig5/fig5_withInh%s'%g))

st_noInh = g_set[0][1]
gid_noInh =g_set[0][2]
Collection_noInh = g_set[0][0]

st = g_set[6][1]
gid=g_set[6][2]
Collection = g_set[6][0]


N=1000
epsilon = 0.5
sim_time = 100000
grid = plt.GridSpec(2, 4, wspace=0.7, hspace=0.5)

sns.set_style('ticks')
import seaborn as sns

plt.figure(figsize=(15,5))
plt.subplot(grid[0, 0:2])

# plt.plot(na(Collection)[:,0],na(Collection)[:,2]/1000)
# plt.plot(st,gid,'.')
sc_noInh,_ = np.histogram(st_noInh[gid_noInh<(N-(N*epsilon))],np.arange(0,sim_time,20))
sc,_ = np.histogram(st[gid<(N-(N*epsilon))],np.arange(0,sim_time,20))
plt.plot(np.arange(0,sim_time-20,20)/1000,sc_noInh)
plt.plot(np.arange(0,sim_time-20,20)/1000,sc)
#plt.plot(np.arange(0,sim_time-20,20),sc_i/500,'--')

plt.ylabel('Total spike count',fontsize = 10)

#plt.plot(np.arange(0,sim_time-20,20),sc_noInh_i,'-')
# plt.yscale('log')
# plt.xlim([0,100000])
plt.subplot(grid[1, 0:2])
plt.plot(na(Collection_noInh)[:,0]/1000,na(Collection_noInh)[:,1]/1000)
plt.plot(na(Collection)[:,0]/1000,na(Collection)[:,1]/1000)

# plt.xlim([0,100000])
plt.yscale('log')
plt.ylabel('Average adaptation',fontsize = 10)
plt.xlabel('Time (s)')
sns.despine()
# plt.plot(na(Collection_noInh)[:,0],na(Collection_noInh)[:,2])
# plt.plot(st,gid,'.')


plt.subplot(grid[:, 2:])
Gs = [0,0.1,0.25,0.5,0.75,0.9,1,1.1,1.25,1.5,1.75,1.9,2]

for i,gs in enumerate(g_set):
        plt.errorbar(Gs[i],np.mean(na(gs[0])[:,1]/1000),np.std(na(gs[0])[:,1]/1000),fmt='--o',color ='C0')
        
plt.axvline(1,color='k', alpha =0.4,label='balance')
plt.legend()
plt.yscale('log')
plt.xlabel('Relative inhibitory strength',fontsize = 10)
plt.ylabel('Average adaptation',fontsize = 10)
sns.despine()
plt.savefig('Figures/figure5.eps',fmt = 'eps')   
