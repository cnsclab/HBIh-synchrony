# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:56:13 2015
The Huber_braun neuronal model function
@author: porio
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate,signal
import time as TM
import sys
import Wavelets
import fcd

if len(sys.argv)>1:
    rseed=int(sys.argv[1])
else:
    rseed=0

def HyB(y,t,tempF):
    [rho,phi]=tempF
    y=np.reshape(y,(5,nnodes),'F')
    v,asd,asr,ah,ar=y

    ad = 1/(1+np.exp(-zd*(v-V0d)))
    isd = rho*gsd*asd*(v - Ed)
    Imemb=isd + rho*gd*ad*(v - Ed) + rho*(gr*ar + gsr*(asr**2)/(asr**2+0.4**2))*(v-Er) \
                + rho*gh*ah*(v - Eh)+ rho*gl*(v - El)
    arinf = 1/(1+np.exp(-zr*(v-V0r)))
    asdinf = 1/(1+np.exp(-zsd*(v-V0sd)))
    ahinf= 1/(1+np.exp(-zh*(v-V0h)))

    Igj = np.sum(CM * Ggj * (v[:,None] - v),-1)

    Det=np.array([-Imemb - Igj,
                phi*(asdinf - asd)/tsd,
                phi*(-eta*isd - kappa*asr)/tsr,
                phi*(ahinf-ah)/th,
                phi*(arinf - ar)/tr])
    Det=Det.flatten('F')
    return Det


#Parameters
gd = 2.5; gr = 2.8; gsd = 0.21; gsr = 0.28;
gl = 0.06; gh = 0.4;
V0d = -25; V0r = -25; zd = 0.25; zr = 0.25;tr = 2;
V0sd = -40; zsd = 0.11; tsd = 10;
eta = 0.014; kappa = 0.18; tsr = 35;
V0h= -85; zh = -0.14; th=125;
Ed = 50; Er = -90; El = -80; Eh = -30;

nnodes=50
Ggj=1e-2 #Coupling conductance

Gnames="FR30to45chaos"
#Gnames="FR30to45nonchaos"
#Gnames="FR30to45noIh"

#Gnames="FR70to95chaos"
#Gnames="FR70to95nonchaos"
#Gnames="FR70to95noIh"

GVals = np.loadtxt('Datasets/' + Gnames + ".txt")
indices=np.random.choice(range(len(GVals)),nnodes,replace=False)

gsd=GVals[indices,0]
gsr=GVals[indices,1]

pij=0.4

np.random.seed(rseed)
CM = np.zeros((nnodes,nnodes))
CM[0,-1]=1
for i in range(1,nnodes):
    c_p = np.random.uniform(size=i)
    if c_p[-1]<pij:
        CM[i,np.argmax(c_p)]=1
    
    CM[i,i-1]=1
     
CM = CM+CM.T

v=np.random.uniform(low=-70,high=-50,size=nnodes) #Random initial voltages
temp=36
#The rest of the variables are set to the steady state
ad = 1/(1+np.exp(-zd*(v-V0d)));
ar = 1/(1+np.exp(-zr*(v-V0r)));
asd = 1/(1+np.exp(-zsd*(v-V0sd)));
ah= 1/(1+np.exp(-zh*(v-V0h)))

rho = 1.3**((temp-25.)/10)
phi = 3**((temp-25.)/10)
asr = -eta*rho*gsd*asd*(v - Ed)/kappa;
tempF=[rho,phi]

# PARAMETERS FOR ADAPTATION SIMULATION
#initial value
y0=np.array([v,asd,asr,ah,ar])
y0=y0.flatten('F')  #The N x 5 array is flattened to work with odeint
adaptTime=15000 #ms
adaptInt=1  #This controls only for the returned values, not the calculation 
Tadapt=np.arange(0,adaptTime,adaptInt)

#PARAMETERS FOR FINAL SIMULATION
runTime=27000  #ms
runInt=0.2
Trun=np.arange(0,runTime,runInt)

Yadapt=integrate.odeint(HyB,y0,Tadapt,args=(tempF,))

print("adaptation ready")

#%% SIMULATION
y0a=Yadapt[-1]  #initial conditions after some adaptation

t0=TM.time()

Y_t=integrate.odeint(HyB,y0a,Trun,args=(tempF,))

CPUtime=TM.time()-t0
print(CPUtime)

#%%
# Signal is filtered and decimated because only the slow oscillation will
# be analyzed
decimate=20;cutfreq=50
b,a=signal.bessel(4,cutfreq*2*runInt/1000,btype='low')

V_t=Y_t[:,::5]
Vfilt=signal.filtfilt(b,a,V_t,axis=0)[::decimate]

freqs=np.arange(2,15,0.2)  #Desired frequencies
Periods=1/(freqs*(decimate*runInt)/1000)    #Desired periods in sample untis
dScales=Periods/Wavelets.Morlet.fourierwl  #desired Scales

#wavel=Wavelets.Morlet(EEG,largestscale=10,notes=20,scaling='log')
wavelT=[Wavelets.Morlet(y1,scales=dScales) for y1 in Vfilt.T]
cwt=np.array([wavel.getdata() for wavel in wavelT])
pwr=np.array([wavel.getnormpower() for wavel in wavelT])

phase=np.array([np.angle(cwt_i) for cwt_i in cwt])

spec=np.sum(pwr,-1)
maxFind=np.argmax(spec,-1)
maxFreq=freqs[maxFind]

bound1=int(1000/(runInt*decimate))
bound2=int((runTime-1000)/(runInt*decimate))
phaseMaxF=phase[range(nnodes),maxFind,bound1:bound2]

phasesynch=np.abs(np.mean(np.exp(1j*phaseMaxF),0))
MPsync=np.mean(phasesynch)
VarPsync=np.var(phasesynch)

print("Mean phase synchrony (R parameter): %g"%MPsync)
print("Metastability (standard deviation of R parameter): %g"%VarPsync)

#%%
PsynchT=np.zeros((nnodes,nnodes,(bound2-bound1)))
Pcoher=np.zeros((nnodes,nnodes))

for ii in range(nnodes):
    for jj in range(ii):
        PsynchT[ii,jj,:]=np.abs(np.mean(np.exp(1j*phaseMaxF[[ii,jj],:]),0))
        Pcoher[ii,jj]=np.abs(np.mean(np.exp(1j*np.diff(phaseMaxF[[ii,jj],:],axis=0))))
Psynch=np.mean(PsynchT,axis=-1)
PsynchMean=np.mean(Psynch[np.tril_indices(nnodes,k=-1)])
PsyncVar=np.mean(np.var(PsynchT[np.tril_indices(nnodes,k=-1)],axis=0))

PcoherMean=np.mean(Pcoher[np.tril_indices(nnodes,k=-1)])
PcoherVar=np.var(Pcoher[np.tril_indices(nnodes,k=-1)])
        
np.savetxt("Smatrix-g%g.txt"%(Ggj),Psynch,fmt='%.6g',delimiter='\t') #Phase synchrony matrix
np.savetxt("Cmatrix-g%g.txt"%(Ggj),Pcoher,fmt='%.6g',delimiter='\t') #Phase locking matrix
#%%

plt.figure(1,figsize=(12,8))
plt.clf()
subpl = [plt.subplot(4,2,i) for i in range(1,9)]

for volt,ax in zip(V_t.T[:8],subpl):
    ax.plot(Trun,volt)

for volt,ax in zip(Vfilt.T[:8],subpl):
    ax.plot(Trun[::decimate],volt)


#%%
plt.figure(3,figsize=(12,8))
plt.clf()
subpl1 = [plt.subplot2grid((4,10),(i%4,5*(i>3)),colspan=4) for i in range(8)]
subpl2 = [plt.subplot2grid((4,10),(i%4,4+5*(i>3))) for i in range(8)]

for pw,ax in zip(pwr[:8],subpl1):
    ax.imshow(pw,cmap='jet',extent=(Trun[0],Trun[-1],freqs[0],freqs[-1]),
              origin='lower',interpolation='none',aspect='auto')

for sp,ax in zip(spec[:8],subpl2):
    ax.plot(sp,freqs)
    ax.set_ylim((freqs[0],freqs[-1]))


PcorrFCD,Pcorr,shift=fcd.extract_FCD(phaseMaxF[:,::],wwidth=500,mode='psync')

np.savetxt("FCDsync-g%g.txt"%(Ggj),PcorrFCD,fmt='%.6g',delimiter='\t') #FCD of synchrony

VarFCDsync=np.var(PcorrFCD[np.tril_indices(len(Pcorr),k=-9)])

Tini=1
Tfin=Trun[-1]/1000 - 1

plt.figure(4,figsize=(10,12))
plt.clf()

plt.subplot2grid((5,5),(0,0),rowspan=2,colspan=5)
plt.plot(Trun[bound1*decimate:bound2*decimate:decimate]/1000,phasesynch)
plt.title('mean P sync')

plt.subplot2grid((5,5),(2,2),rowspan=2,colspan=2)
plt.imshow(PcorrFCD,vmin=0,vmax=1,extent=(Tini,Tfin,Tfin,Tini),interpolation='none',cmap='jet')
plt.title('P sync FCD')
plt.grid()

plt.subplot2grid((5,5),(2,4))
plt.imshow(CM,cmap='gray_r',interpolation='none')
plt.gca().set_xticklabels((),())
plt.gca().set_yticklabels((),())
plt.title('SC')
plt.grid()

plt.subplot2grid((5,5),(3,4))
plt.hist(PcorrFCD[np.tril_indices(len(Pcorr),k=-9)],range=(0,1),color='C1')
ax.text(0.5,0.97,'%.4g'%VarFCDsync,transform=ax.transAxes,ha='center',va='top',fontsize='xx-small')

plt.subplot2grid((5,5),(4,4))
plt.imshow(Pcoher+Pcoher.T+np.eye(nnodes),cmap='jet',vmax=1,vmin=0,interpolation='none')
plt.gca().set_xticklabels((),())
plt.gca().set_yticklabels((),())
plt.title('P coher')
plt.grid()

axes2=[plt.subplot2grid((5,5),pos) for pos in ((4,0),(4,1),(4,2),(4,3))]
for axi,ind in zip(axes2,(20,40,60,80)):
    corrMat=np.zeros((nnodes,nnodes))
    corrMat[np.tril_indices(nnodes,k=-1)]=Pcorr[ind]
    corrMat+=corrMat.T
    corrMat+=np.eye(nnodes)
    
    axi.imshow(corrMat,vmin=0,vmax=1,interpolation='none',cmap='jet')
    
    axi.set_xticklabels((),())
    axi.set_yticklabels((),())
    axi.grid()

plt.tight_layout()

