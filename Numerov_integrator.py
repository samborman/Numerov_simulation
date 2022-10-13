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
hbar = 1 #1.054471817E-34 #Js
m = 1

#Grid settings
N = 10001 #number of grid points, use odd number to include 0
xmin = -5
xmax = 5

#Simulation settings
init_guess = 1E-30
threshold = 1E-10

Nmin=0
Nmax=2

#Initialize grid
x = np.linspace(xmin, xmax, N) #include boundaries
dx = (xmax-xmin)/(N-1) #N-1 because we include the boundary
psi_fwd = np.zeros(N)
psi_bwd = np.zeros(N)
psi = np.zeros(N)

States = {}

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
def Normalization(WaveFunction, WaveFunctionGrid):
    """
    Normalization of a wavefunction using simpsons rule
    In: (list) a wavefunction and its grid
    Out: (list) the wavefunction normalized
    """
    WaveFunction2 = np.abs(WaveFunction)**2
    NormalizationConstant = scipy.integrate.simps(WaveFunction2, WaveFunctionGrid)
    return WaveFunction/np.sqrt(NormalizationConstant)

class State(object):
    """ A state has an Energy and a wavefunction """
    
    def __init__(self, Energy, WaveFunction):
        self.Energy = Energy
        self.WaveFunction = WaveFunction
    
    def getEnergy(self):
        return self.Energy
    
    def getWaveFunction(self):
        return self.WaveFunction
    
    def setTime_dependency(self, WaveFunction_t):
        self.WaveFunction_t = WaveFunction_t
        self.WaveFunction_t_Re = np.real(WaveFunction_t)
        self.WaveFunction_t_Im = np.imag(WaveFunction_t)

#%% Numerov Algorithm
for nodes in range(Nmin, Nmax):   
    
    Emax = np.max(vpot)
    Emin = np.min(vpot)
    
    while True:
        psi_fwd = np.zeros(N)
        psi_bwd = np.zeros(N)
        psi = np.zeros(N)
        
        Eguess = (Emax + Emin)/2
        print("Emin = ", Emin)
        print('Emax = ', Emax)
        print('Eguess = ', Eguess) 
           
        k = 2*m/(hbar**2) * (Eguess - vpot)
          
        #find the rightmost turning point
        icl = N-1
        while vpot[icl] > Eguess:
            icl = icl -1
            if icl < 0:
                print("no classical turning point found")
                
        #find all classical turning points
        turning_points = []
        for i in range(N-1):
            if vpot[i] == Eguess or (vpot[i] < Eguess and vpot[i+1] > Eguess) or (vpot[i] > Eguess and vpot[i+1] < Eguess):
                turning_points.append(i)
        
        #fwd Numerov integration up to icl
        psi_fwd[0] = 0
        psi_fwd[1] = init_guess #initial guess 
        for i in range(2, icl+1):
            psi_fwd[i] = (2*(1-5/12*dx**2*k[i-1])*psi_fwd[i-1] - (1+1/12*dx**2*k[i-2])*psi_fwd[i-2])/(1+1/12*dx**2*k[i]) #Numerov step
            psi[i] = psi_fwd[i]
            
        #bwd numerov integration
        psi_bwd[N-1] = 0
        psi_bwd[N-2] = init_guess
        psi[N-1] = psi_bwd[N-1]
        psi[N-2] = psi_bwd[N-2]
        for i in range(N-3, icl-1, -1):
            psi_bwd[i] = (2*(1-5/12*dx**2*k[i+1])*psi_bwd[i+1] - (1+1/12*dx**2*k[i+2])*psi_bwd[i+2])/(1+1/12*dx**2*k[i]) #Numerov step
            psi[i] = psi_bwd[i]
            
        #scale fwd and bwd to the same size
        scale_factor = psi_bwd[icl]/psi_fwd[icl]
        for i in range(0, icl+1):
            psi[i] = psi_fwd[i]*scale_factor
        
        #Normalize
        psi = Normalization(psi, x)
    
        #Find number of nodes
        n = 0
        for i in range(1, N):
             if psi[i-1]*psi[i] < 0:
                 n = n + 1
        print('number of nodes = ', n)
        
        #update energy according to desired level
        if n > nodes:
            Emax = Eguess
        elif n < nodes:
            Emin = Eguess
        else:
            #Caulculate the discontinuity at the matching point = y'Fwd(xc)-yBwd'(xc)    
            kcl = k[icl]
            fcl = 1+(kcl*dx**2)/12
            discontinuity = (psi[icl-1] + psi[icl+1] - (14-12*fcl)*psi[icl])/dx #Discontinuity according to Taylor expansion
            discontinuity = discontinuity*psi[icl] #Scale discontinuity (matching logarithmic derivative) and update energy. 
            print("Discontinuity = ", discontinuity)
            
            if discontinuity > 0:
                Emax = Eguess
            else:
                Emin = Eguess
                
            #Determine if we have the correct energy and wavefunction by a threshold in the energy bisection method and the matching of the Fwd and Bwd discontinuity
            if Emax - Emin < threshold and np.abs(discontinuity) < threshold:
                print("Converged at eigenvalue E=", Eguess)
                Energy = Eguess            
                WaveFunction = Normalization(psi, x) #Normalize
                print("#--------------------------#")
                break 
    States[nodes] = State(Energy, WaveFunction)
    
#%% Plot
fig, ax = plt.subplots()
ax.plot(x, vpot, label='Potential')
for nodes in range(Nmin, Nmax):
    ax.plot(x, States[nodes].WaveFunction*pot_depth + States[nodes].Energy, label=r'$\psi_{%d}$ @ E=%.2f' %(nodes, States[nodes].Energy))
ax.set_xlabel('x')
ax.set_ylabel(r'$\psi$')
ax.legend()
plt.show()