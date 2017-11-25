# -*- coding: utf-8 -*-
"""
Created on Wed May 28 16:31:57 2014
@author: keshengxuu@mail.com 
@author: kesheng Xu
"""
from __future__ import division
import matplotlib
matplotlib.use('Agg')
import numpy as np
import time as TM
import sys
import mlenetwork as mle
import os

if len(sys.argv)>1:
    rseed=int(sys.argv[1])
else:
    rseed=0
np.random.seed(rseed)



#%%
# Networks of HB+Ih model
def HyB(Var,t,tempF,sqdt,parameter):
    [rho,phi]=tempF
    [v,ar,asd,asr,ah]=Var 
    [gsd,gh]=parameter
    ad = 1/(1+np.exp(-zd*(v-V0d)))
    isd = rho*gsd*asd*(v - Ed)
    Imemb=isd + rho*gd*ad*(v - Ed) + rho*(gr*ar + gsr*(asr**2)/(asr**2+0.4**2))*(v-Er) \
                + rho*gh*ah*(v - Eh)+ rho*gl*(v - El)
    arinf = 1/(1+np.exp(-zr*(v-V0r)))
    asdinf = 1/(1+np.exp(-zsd*(v-V0sd)))
    ahinf= 1/(1+np.exp(-zh*(v-V0h)))
# the coupling of gap junction
    Igj = np.sum(CM * Ggj * (v[:,None] - v),-1)
     
    Det=np.array([-Imemb - Igj,
                phi*(arinf - ar)/tr,
                phi*(asdinf - asd)/tsd,
                phi*(-eta*isd - kappa*asr)/tsr,
                phi*(ahinf-ah)/th])
   
    Stoch=np.array([sqdt*np.random.normal(size=np.shape(v)),  #different noise, in this case sqdt  equal zero
#    Stoch=np.array([sqdt*np.random.normal()*np.ones_like(v),    #same noise 
                    np.zeros_like(v),
                    np.zeros_like(v),
                    np.zeros_like(v),
                    np.zeros_like(v)])
    
    return Det+Stoch
#%%
#The default parameter values of the model
gd = 2.5; gr = 2.8; gsd = 0.23; gsr = 0.28;
gl = 0.06; gh = 0.4;
V0d = -25; V0r = -25; zd = 0.25; zr = 0.25;tr = 2;
V0sd = -40; zsd = 0.11; tsd = 10;
eta = 0.014; kappa = 0.18; tsr = 35;
V0h= -85; zh = -0.14; th=125;
Ed = 50; Er = -90; El = -80; Eh = -30;

# the network size
nsim=50
# the temperature
temp=36*np.ones(nsim)
pij=0.4
#%%
# numpy.round is evenly round to the given number of decimals
Ggjvals=np.logspace(-6,0,num=23)
Ggjvals=np.insert(Ggjvals,0,0)
#threads=len(Ggjvals)
CM = np.zeros((nsim,nsim))


#%%
CM[0,-1]=1
for i in range(1,nsim):
    c_p = np.random.uniform(size=i)
    if c_p[-1]<pij:
        CM[i,np.argmax(c_p)]=1
    
    CM[i,i-1]=1
        
CM = CM+CM.T

#Simulation 
tBegin=0
tEnd=20000
dt=0.025
time= np.arange(tBegin, tEnd, dt)
N = time.size
Ggj=1e-2 #Coupling conductance

v=np.random.uniform(low=-70,high=-30,size=nsim)

Gnames="FR30to45chaos"
#Gnames="FR30to45nonchaos"
#Gnames="FR30to45noIh"

#Gnames="FR70to95chaos"
#Gnames="FR70to95nonc
folder=Gnames + '/chaos%gnodes-seed%g/'%(nsim,rseed)

directory = os.path.dirname(folder)
if not os.path.exists(directory):
    os.makedirs(directory)

GVals = np.loadtxt('Datasets/' + Gnames + ".txt")


indices=np.random.choice(range(len(GVals)),nsim,replace=False)

gsd=GVals[indices,0]
gsr=GVals[indices,1]


ar = 1/(1+np.exp(-zr*(v-V0r)));
asd = 1/(1+np.exp(-zsd*(v-V0sd)));
ah= 1/(1+np.exp(-zh*(v-V0h)))
rho = 1.3**((temp-25.)/10)
phi = 3**((temp-25.)/10)
asr = -eta*rho*gsd*asd*(v - Ed)/kappa; 

#initial conditions
X=np.array([v,ar,asd,asr,ah])

print(Ggj,"adaptation ready")

Y0=mle.MLE_Euler(HyB,X=X,N=N,dt=dt,tempF=(rho,phi),parameter=(gsd,gsr)) 

Y=Y0[0]
MLE=Y0[1]

CPUtime=TM.time()
print(Ggj,MLE,CPUtime)
#save the data to the file       
with open(folder+"chmet.txt",'a') as dataf:
    dataf.write("%g,%g\n"%(Ggj,MLE))


