#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 06:34:53 2017

@author: quien
"""

import numpy as np;

from scipy.ndimage import convolve1d as conv1d;
from scipy.optimize import newton as findroot;
import matplotlib.pyplot as plt;
import matplotlib.image as img;

def conv(A,w):
    R = np.copy(A);
    for d in range(len(A.shape)):
        R = conv1d(R,w,axis=d,mode='constant',cval=0.0);
    return R;

def entropy(p):
    return np.mean(-p[p>0.0]*np.log(p[p>0.0]));

class conv_wass:
    def __init__(self,w,sig,T):
        
        self.T = T;
        
        self.H = np.zeros(2*w+1);
        for i in range(-w,w+1):
            self.H[i+w] = np.exp(-(i*i)/sig**2);
        #self.H *= 1.0/(np.sqrt(2.0*np.pi)*sig);
        self.H /= np.sum(self.H);
    def entropy_sharp(self,mu,h0):
        b = 1.0;
        mmu = np.mean(mu);
        if mmu+entropy(mu) > 1+h0:
            b = findroot((lambda a: mmu+entropy(np.power(mu,a))-(1+h0)));
        if b < 0.0:
            b = 1.0;
        return np.power(mu,b);
    def barycenter(self,mus,ais):
        K = len(mus);
        S = mus[0].shape;
        a = np.prod(S);
        v = [np.ones(S) for k in range(K)];
        w = [np.ones(S) for k in range(K)];
        d = [np.ones(S) for k in range(K)];
        
        H0 = np.max([entropy(mus[k]) for k in range(K)]);
             
        mu = np.ones(S);
        for t in range(self.T):
            mu.fill(1.0);
            for i in range(K):
                aux = conv(v[i],self.H);
                ind = aux>0.0;
                aux[ind] = 1.0/(aux[ind]+1e-10);
                w[i] = mus[i]*aux*a;
                
                d[i] = (1.0/a)*v[i]*conv(w[i],self.H);
                aux = np.power(d[i],ais[i]);
                mu = mu*aux;
            mu = self.entropy_sharp(mu,H0);
            for i in range(K):
                aux = np.copy(d[i]);
                ind = aux>0.0;
                aux[ind] = 1.0/(aux[ind]+1e-10);
                v[i] = v[i]*mu*aux;
        return mu / np.sum(mu);


CW = conv_wass(5,8.0,100);
plt.plot(CW.H);
        
A = 1.0-np.mean(img.imread('cat.png'),axis=2);
A /= np.sum(A);
           
B = 1.0-np.mean(img.imread('star.png'),axis=2);
B /= np.sum(B);

C = 1.0-np.mean(img.imread('circle.png'),axis=2);
C /= np.sum(C);

N = 5;

R = np.zeros((N,N,A.shape[0],A.shape[1]));
x = np.linspace(0.0,1.0,N);
for a in range(N):
    for b in range(N):
        print a,b;
        if x[a] + x[b] <= 1.0:
            R[a,b,:,:] = CW.barycenter([A,B,C],[x[a],x[b],1.0-x[a]-x[b]]);

f,axarr = plt.subplots(N,N);
for i in range(N):
    for j in range(N):
        axarr[i,j].imshow(1.0-R[i,j,:,:],cmap='gray');
        axarr[i,j].set_xticklabels([]);
        axarr[i,j].set_yticklabels([]);
        axarr[i,j].grid(False)
plt.show()