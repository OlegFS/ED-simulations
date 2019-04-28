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

Counts = pd.read_excel('data/Results.xlsx')
Counts = Counts.drop(columns = 0.25)
# plt.figure(figsize=(15,4))
# fig, ax = plt.subplots()
plt.figure(figsize=(15,5))
plt.plot(np.linspace(0,10,10),np.linspace(0,100,10),'--k')
# plt.plot(np.arange(0,100,0.1),np.arange(0,1,0.1),'--k')
# sns.swarmplot(data = Counts,orient = 'v',order=np.arange(0,1,0.1))
for i,col in enumerate(Counts):
    y= Counts[col]
    y = y[np.isfinite(y)]
#     plt.axvline(col*100)
    sns.swarmplot(x = [col*100]*len(y),y = y,orient = 'v',order=np.arange(0,110,10),size =10)
plt.yticks(fontsize = 18)
plt.xticks(fontsize = 18)
sns.despine(trim=1)
# plt.title('Fraction of inhibitory cells')
plt.xlabel('Seeded',fontsize = 24)
plt.ylabel('Observed',fontsize = 24)
plt.tight_layout()
plt.savefig('figs/counts2.pdf',fmt = 'pdf')


plt.figure(figsize=(15,5))
for col in Counts:
    y= Counts[col]
    y = y[np.isfinite(y)]
    plt.axvline(col*100)
    plt.hist(y,10)


