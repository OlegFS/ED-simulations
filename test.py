# Look for the best fit in the network
# Input only to EI
# Varibility is given by E/I 
# The first rought gues for the EI model with N=5001
from func.ED_sim_ad import *
from func.processing_helpers import *

np.random.seed(1234)
random.seed(1234)

j = 18
ex_rate = (0.09615384615384616/5)
N = 1000
epsilon = 0.2

params = {'Vthr': 20,
          'Vres': 10,
          'V0':0,
          'tau_ref': 2.0,
          'tau_m':40,
          'd':3.5,
          'N':1000,
          'epsilon':0.2,
          'p':0.1,
          'g':4.0,
          'J':(j*np.sqrt(5))/np.sqrt((N-(epsilon*N))*0.1),#1.4,#25/np.sqrt(25),#0.94868329805051377,
          'J_ext':8.0,
          'input':'NE+NI',
          'ex_rate':(ex_rate/80)*((N-(epsilon*N))*0.1),
          'eta':0.0,
          'tau_w':18000,
          'a':0.0,
          'b':2}
run(params, sim_time=100000)
st,gid = read(params)
plt.plot(st,gid,'.',markersize = 1)
print(len(st)/100)
