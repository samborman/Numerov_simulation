# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 12:44:12 2022

@author: Sam
"""
#Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

#%%Setup
#Physical constants
hbar = 1.054471817E-34 #Js
m = 1

#Grid settings
N = 10001 #number of grid points, use odd number to include 0
xmin = -5
xmax = 5

#Simulation settings
init_guess = 1E-30
threshold = 1E-14

Nmin=0
Nmax=2

#Initialize grid
x = np.linspace(xmin, xmax, N) #include boundaries
dx = (xmax-xmin)/(N-1) #N-1 because we include the boundary
psi_fwd = np.zeros(N)
psi_bwd = np.zeros(N)
psi = np.zeros(N)

#%%Potential
def potential(x, a):
    return a*x**2

a=0.5
vpot = potential(x, a) #potential evaluated at all points in x
pot_depth = (np.max(vpot)- np.min(vpot))/2 #depth of the wells, only used for plotting the wavefunction at a visible scale

#plot the discretized potential
fig, axpot = plt.subplots()
axpot.plot(x, vpot)
plt.show()

#%%Functions
def k_function(x, E, a):
    return 2*m/(hbar**2) * (E - potential(x, a))



