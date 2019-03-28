%matplotlib inline
import pandas as pd 
from func.processing_helpers import *
from func.ED_sim_ad import load_obj

N = 1000
epsilon = 0.5
NI = int(epsilon*N)
p=0.1

conn_dict = load_obj('conn/%s_of_%s_fixed_in_p=%s.pkl'%(NI,N,p))
out_deg = []
for i in range(1000):
    out_deg.append(len(conn_dict[i]))

plt.subplot(121) 
plt.hist(out_deg[:N-NI],bins =50)
plt.subplot(122) 
plt.hist(out_deg[N-NI:],bins =50)
