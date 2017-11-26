# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 16:23:31 2016
keshengxuu@mail.com 
@author: ksxuu
"""
from __future__ import division
import matplotlib
matplotlib.use('Agg')
import numpy as np


def MLE_Euler(HyB,X,N,dt,tempF,parameter):
    '''
    input agrument:
        HyB : the function of HB+Ih model
        X : variables should include v,ar,asd,asr,ah
        N : The number of simulation steps
        tempF: temperature-dependent functions
        parameter : the pairwise parameter space, for instance, gsd/gh parameter space
    output:
        datos: time series of variables for whole nodes 
        mle: Maximal Lyapunov exponent
    References:
        [1]. Sprott, J. C. (2003). Chaos and time-series analysis (Vol. 69). Oxford: Oxford University Press.
        [2]. Jones, D. S., Plank, M., & Sleeman, B. D. (2009). Differential equations and mathematical biology. CRC press.
    '''
    t_trans=5000
    nsim=np.shape(X)[1]
    datos = np.zeros((5,nsim,N))  # the array for saving the neworks
    datos0 = np.zeros((5,nsim)) # the arrat for saving the transient simulation data
    datosd0= np.zeros((5,nsim,N)) # the array for saving the perturbed the neworks
    datos0[:,:] =X
    D=0  #D  control the noise 
    sqrtDdt =np.sqrt(D/dt)
# integrate to get rid of transient behaviour,sothat it settled on the attractor of the dynamics
    j=1
    while j<=t_trans:
        datos0=datos0+dt*HyB(datos0,j*dt,tempF,sqrtDdt,parameter)
        j+=1

# ran two copes of the dynamics,one with initial conditions h and the other is slightly
# pertured initial conditions h_{1}=h+d0,here is datosd0.
    i=1
    sum0 = 0
    d0=0.00001
    datosd0[0,0,0]=d0
    datos[:,:,0] =datos0 # final value of solution is datos[:,:,-1]
    datosd0[:,:,0]=datos[:,:,0]+datosd0[:,:,0] # perturb by d0 to get y2
# integrate both orbits of  networks  by a time dt:
    while i<N: # N is integration steps of times
        datos[:,:,i]=datos[:,:,i-1]+dt*HyB(datos[:,:,i-1],i*dt,tempF,sqrtDdt,parameter);# unpertured systems
        datosd0[:,:,i]=datosd0[:,:,i-1]+dt*HyB(datosd0[:,:,i-1],i*dt,tempF,sqrtDdt,parameter); # slightly pertured  systems
        d1 =np.linalg.norm(datosd0[:,:,i]-datos[:,:,i]); # new separation
        lambda0 =np.log(d1/d0)/dt;   # Lyapunov exponent
        sum0 = sum0+lambda0;        # running sum of Lyapunov exponents
        datosd0[:,:,i] = datos[:,:,i]+(datosd0[:,:,i]-datos[:,:,i])*d0/d1;   # renormalise y2 so separation is d0
        i+=1;
    mle=sum0/N;
    output=[datos,mle]
    return output
