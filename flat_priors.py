import random
import numpy as np
import math

# information on parameters from https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.044006

sampling_frequency = 0
t_0_min = 0
t_0_max = 0
f_0_min = 0
Q = random.uniform(.1,40)
phi_0 = random.uniform(0,2*np.pi)

# Nyquist frequency
f_0_max = sampling_frequency/2
tau = Q/(2*np.pi*f_0)

# power spectral density
# S_n
# ro = ((A**2)*Q)/(2*np.sqrt(2*np.pi)*f_0*S_n(f_0))

# Calculating prior Morlet Wavelet amplitude (Need to check if this is the right formula)
f*=?
A_0 = math.exp(-(np.pi**2*tau**2(f*-f_0)**2))

# number of wavelets
D_min = 0
D = uniform(D_min,100)
