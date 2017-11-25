# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:57:38 2016

@author: jmaidana
@author: porio
"""
from __future__ import division
import numpy as np
import matplotlib.pylab as plt

def extract_FCD(data,wwidth=1000,maxNwindows=100,olap=0.9,coldata=False,mode='corr'):
    """
    Functional Connectivity Dynamics from a collection of time series

    Parameters:
    -----------
    data : array-like
        2-D array of data, with time series in rows (unless coldata is True)
    wwidth : integer
        Length of data windows in which the series will be divided, in samples
    maxNwindows : integer
        Maximum number of windows to be used. wwidth will be increased if necessary
    olap : float between 0 and 1
        Overlap between neighboring data windows, in fraction of window length
    coldata : Boolean
        if True, the time series are arranged in columns and rows represent time
    mode : 'corr' | 'psync' | 'plock' | 'tdcorr'
        Measure to calculate the Functional Connectivity (FC) between nodes.
        'corr' : Pearson correlation. Uses the corrcoef function of numpy.
        'psync' : Pair-wise phase synchrony.
        'plock' : Pair-wise phase locking.
        'tdcorr' : Time-delayed correlation, looks for the maximum value in a cross-correlation of the data series 
        
    Returns:
    --------
    FCDmatrix : numpy array
        Correlation matrix between all the windowed FCs.
    CorrVectors : numpy array
        Collection of FCs, linearized. Only the lower triangle values (excluding the diagonal) are returned
    shift : integer
        The distance between windows that was actually used (in samples)
            
        

    """
    
    if olap>=1:
        raise ValueError("olap must be lower than 1")
    if coldata:
        data=data.T    
    
    all_corr_matrix = []
    lenseries=len(data[0])
    
    Nwindows=min(((lenseries-wwidth*olap)//(wwidth*(1-olap)),maxNwindows))
    shift=int((lenseries-wwidth)//(Nwindows-1))
    if Nwindows==maxNwindows:
        wwidth=int(shift//(1-olap))
    
    indx_start = range(0,(lenseries-wwidth+1),shift)
    indx_stop = range(wwidth,(1+lenseries),shift)
         
    nnodes=len(data)

    for j1,j2 in zip(indx_start,indx_stop):
        aux_s = data[:,j1:j2]
        if mode=='corr':
            corr_mat = np.corrcoef(aux_s) 
        elif mode=='psync':
            corr_mat=np.zeros((nnodes,nnodes))
            for ii in range(nnodes):
                for jj in range(ii):
                    corr_mat[ii,jj]=np.mean(np.abs(np.mean(np.exp(1j*aux_s[[ii,jj],:]),0)))
        elif mode=='plock':
            corr_mat=np.zeros((nnodes,nnodes))
            for ii in range(nnodes):
                for jj in range(ii):
                    corr_mat[ii,jj]=np.abs(np.mean(np.exp(1j*np.diff(aux_s[[ii,jj],:],axis=0))))
        elif mode=='tdcorr':
            corr_mat=np.zeros((nnodes,nnodes))
            for ii in range(nnodes):
                for jj in range(ii):
                    maxCorr=np.max(np.correlate(aux_s[ii,:],aux_s[jj,:],mode='full')[wwidth//2:wwidth+wwidth//2])
                    corr_mat[ii,jj]=maxCorr/np.sqrt(np.dot(aux_s[ii,:],aux_s[ii,:])*np.dot(aux_s[jj,:],aux_s[jj,:]))
        all_corr_matrix.append(corr_mat)
        
    corr_vectors=np.array([allPm[np.tril_indices(nnodes,k=-1)] for allPm in all_corr_matrix])
    
    CV_centered=corr_vectors - np.mean(corr_vectors,-1)[:,None]
    
    
    return np.corrcoef(CV_centered),corr_vectors,shift

if __name__=='__main__':    

    #Let's create some data with sine waves
    dt=0.002
    runTime=20
    nnodes=20
    Trun=np.arange(0,runTime,dt)
    
    frequencies=np.random.uniform(3,10,size=nnodes)
    phase=np.random.uniform(-np.pi,np.pi,size=nnodes)
    
    timeseries=np.sin(2*np.pi*frequencies[:,None]*(Trun-phase[:,None]))
    
    phasesynch=np.abs(np.mean(np.exp(1j*timeseries),0))
    Pcoher=np.zeros((nnodes,nnodes))
    
    for ii in range(nnodes):
        for jj in range(ii):
            Pcoher[ii,jj]=np.abs(np.mean(np.exp(1j*np.diff(timeseries[[ii,jj],:],axis=0))))

    PcorrFCD,Pcorr,shift=extract_FCD(timeseries,wwidth=500,olap=.9,mode='psync')
    Tini=1
    Tfin=Trun[-1]
    plt.figure(4,figsize=(10,12))
    plt.clf()
    
    plt.subplot2grid((5,4),(0,0),rowspan=2,colspan=4)
    plt.plot(Trun,phasesynch)
    plt.title('mean P sync')
    
    plt.subplot2grid((5,4),(2,0),rowspan=2,colspan=2)
    plt.imshow(PcorrFCD,vmin=0,vmax=1,extent=(Tini,Tfin,Tfin,Tini),interpolation='none',cmap='jet')
    plt.title('Phase synch FCD')
    plt.grid()
    
    plt.subplot2grid((5,4),(2,2),rowspan=2,colspan=2)
    plt.imshow(Pcoher+Pcoher.T+np.eye(nnodes),cmap='jet',vmax=1,vmin=0,interpolation='none')
    plt.gca().set_xticklabels((),())
    plt.gca().set_yticklabels((),())
    plt.title('Phase locking')
    plt.grid()
    
    axes2=[plt.subplot2grid((5,4),pos) for pos in ((4,0),(4,1),(4,2),(4,3))]
    for axi,ind in zip(axes2,(25,50,75,90)):
        corrMat=np.zeros((nnodes,nnodes))
        corrMat[np.tril_indices(nnodes,k=-1)]=Pcorr[ind]
        corrMat+=corrMat.T
        corrMat+=np.eye(nnodes)
        
        axi.imshow(corrMat,vmin=0,vmax=1,interpolation='none',cmap='jet')
        
        axi.set_xticklabels((),())
        axi.set_yticklabels((),())
        
        axi.set_title('t=%.2g'%(ind*Tfin/len(Pcorr)))
        axi.grid()
        
    
