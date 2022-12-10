#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 22:52:57 2022

@author: tobias witting witting@mbi-berlin.de

exmaple for ASSL
Short Course 510
Spatio-Temporal Characterization of Ultrafast Laser Pulses
Sunday, 11.12.2022 09:00 - 12:00, Room 122-123
"""

#%% imports
import math
import numpy as np
import unitconversion as uc
import fourier_trafo

import matplotlib.pyplot as plt
import matplotlib        as mpl
%matplotlib inline


#%% helper functions not in above packages
def scaleMax1(y):
    """Scale max to 1."""
    y1 = y / np.max(y)
    return y1


def abs_sqd(x):
    """Absolute squared as real of complx * complx_conj."""
    return np.real(x*np.conj(x))


def gaussian(x, fwhm=1, power=2):
    """Define a gaussian."""
    setSigma = fwhm / (2*(2*np.log(2))**(1/power))
    g = np.exp(-(np.abs(x)**power) / (2*setSigma**power))
    return g


def pulse_phase_ceo(omega, phin=np.array([0, 0, 0, 0])):
    """
    Calculate the spectral phase with CEP.

    For given omega vector and Taylor coefficients.
    Omege vector is already centred at omega0,
    i.e. supply vector as omega-omega0
    """
    phi = 0
    for n in range(len(phin)):
        phi += phin[n] * omega**(n) / math.factorial(n)
    # for n, coeff in enumerate(coefficients):
    #    phi += coeff * omega**(n) / math.factorial(n)
    return phi


def moment(x, y, n, norm_fac=None, axis=-1):
    """Calculate moment of distribution."""
    if norm_fac is None:
        norm_fac = y.sum(axis, keepdims=True)

    return (x**n * y).sum(axis, keepdims=True) / norm_fac


# def gabor(t, Et, windowsize=None, lambda_range=None):

#     if windowsize is None:
#         w = fourier_trafo.conj_axis(t)
#         Ew = fourier_trafo.t2f(Et)
#         w0 = moment(w, abs_sqd(Ew), 1)
#         fwhm1 = 30 * 2*np.pi/w0  # best to define window size based on cycles of field
#     else:
#         fwhm1 = windowsize
#     print("gabor window size: %.1f" % fwhm1)
#     window = gaussian(t - np.atleast_2d(t).T, fwhm1, 2)
#     G = abs_sqd( fourier_trafo.t2f(Et*window, axis=1) )
    
#     if lambda_range is not None:
#         w = fourier_trafo.conj_axis(t)
#         l1 = uc.radfs2nm(w)          
#         ix = sorted( [find_index(l1, lambda_range[0]), find_index(l1, lambda_range[1]) ] )
#         lambda_nm = l1[ix[0]:ix[1]]  
#         Gl = G[:, ix[0]:ix[1]]
#         return Gl, lambda_nm
#     else:
#         return G





#%% setup a grid

Dt = 500
dt = 0.1

t = np.arange(-Dt, Dt, dt)  # create a time grid  units: fs
w = fourier_trafo.conj_axis(t)  # w short for omega units: rad/fs
lnm = uc.radfs2nm(w)  # lambda in nm for plotting

#%% define a pulse and play with the spectral phase
w0 = uc.nm2radfs(800)

Ew = gaussian(w-w0, 1.0)

phiw = pulse_phase_ceo(w-w0, np.array([0, 0, 0, 80]))  # define a Taylor phase
Ew = Ew * np.exp(-1j*phiw)   # add this phase to spectrum

Et_FTL = fourier_trafo.f2t(np.abs(Ew))
Et = fourier_trafo.f2t(Ew)

fig = plt.figure(num=1, figsize=(10, 10))
fig.clear()
plt.rcParams['font.size'] = '20'
ax = fig.subplots(2, 2, sharex='col')
ax[0,0].plot(w, abs_sqd(Ew))
ax[1,0].plot(w, -np.unwrap(np.angle(Ew)))
ax[0,0].set_xlim(1, 4)

ax[0,1].plot(t, abs_sqd(Et_FTL), color='k')
ax[0,1].plot(t, abs_sqd(Et))
ax[1,1].plot(t, np.real(Et_FTL), color='k', linewidth=0.5)
ax[1,1].plot(t, np.real(Et))
ax[0,1].set_xlim(-30, 30)





#%% calculate autocorrelation
ACF_FTL = np.abs(fourier_trafo.f2t( fourier_trafo.t2f(abs_sqd(Et_FTL)) * np.conj(fourier_trafo.t2f(abs_sqd(Et_FTL)) )))
ACF = np.abs(fourier_trafo.f2t( fourier_trafo.t2f(abs_sqd(Et)) * np.conj(fourier_trafo.t2f(abs_sqd(Et)) )))


fig = plt.figure(num=1, figsize=(10, 10))
fig.clear()
plt.rcParams['font.size'] = '20'
ax = fig.subplots(3, 2, sharex='col')
ax[0,0].plot(w, abs_sqd(Ew))
ax[1,0].plot(w, -np.unwrap(np.angle(Ew)))
ax[0,0].set_xlim(1, 4)

ax[0,1].plot(t, abs_sqd(Et_FTL), color='k')
ax[0,1].plot(t, abs_sqd(Et))
ax[1,1].plot(t, np.real(Et_FTL), color='k', linewidth=0.5)
ax[1,1].plot(t, np.real(Et))
ax[0,1].set_xlim(-30, 30)

ax[2,1].plot(t, ACF_FTL, color='k', linewidth=0.5)
ax[2,1].plot(t, ACF)



#%% time domain ptychography

GEw = Ew * gaussian(w-w0, 150e-3, 2)   # define a Gate pulse by a spectral filter derived from unknown pulse Et
GEt = fourier_trafo.f2t(GEw)


tau = np.arange(-100, 100, 5)

def make_trace(PEt=Et, GEt=GEt, tau=tau):
    GEw = fourier_trafo.t2f(GEt)
    GEw_tau = GEw * np.exp(-1j * np.atleast_2d(tau).T * np.atleast_2d(w))
    GEt_tau = fourier_trafo.f2t(GEw_tau, axis=1)
    SEt = GEt_tau * PEt
    SIw = abs_sqd( fourier_trafo.t2f(SEt, axis=1) )
    return SIw

SIw = make_trace(PEt=Et, GEt=GEt, tau=tau)


fig = plt.figure(num=1, figsize=(10, 10))
fig.clear()
plt.rcParams['font.size'] = '20'
ax = fig.subplots(3, 2) # , sharex='col')
ax[0,0].plot(w, abs_sqd(Ew))
ax[0,0].plot(w, abs_sqd(GEw), color='orange')

ax[1,0].plot(w, -np.unwrap(np.angle(Ew)))

ax[0,1].plot(t, scaleMax1(abs_sqd(Et_FTL)), color='k')
ax[0,1].plot(t, scaleMax1(abs_sqd(Et)))
ax[0,1].plot(t, scaleMax1(abs_sqd(GEt)), color='orange')

ax[1,1].plot(t, np.real(Et_FTL), color='k', linewidth=0.5)
ax[1,1].plot(t, np.real(Et))
ax[0,1].set_xlim(-30, 30)

# ax.set_xlim(3, 6)
ax[2,0].pcolorfast(w, tau, SIw)  # , cmap=cms.cividis_white)

ax[0,0].set_xlim(1, 4)
ax[1,0].set_xlim(1, 4)
ax[2,0].set_xlim(3, 6)




#%% spectral interferometry
# allows to measure a phase difference just by a spectral fringe measurement

tau = 100
phidiff = pulse_phase_ceo(w-w0, np.array([0, 0, 30]))
                          
Ew2 = Ew + Ew * np.exp(-1j*phidiff) * np.exp(-1j*pulse_phase_ceo(w-w0, np.array([0, tau])))
SI = abs_sqd(Ew2)   # the spectrometer measures the modulus squared

SI_ft = fourier_trafo.f2t(SI)

acfilter = gaussian(t-tau, 100, 8)

SI_ft_acfilt = fourier_trafo.t2f(SI_ft * acfilter)

phisi = -np.unwrap(np.angle(SI_ft_acfilt))
phifit = np.polyfit(w, phisi, 1, w=abs_sqd(SI))
phisi = phisi - np.polyval(phifit, w)

fig = plt.figure(num=1, figsize=(10, 10))
fig.clear()
plt.rcParams['font.size'] = '20'
ax = fig.subplots(2, 2, sharex='col')
ax[0,0].plot(w, SI)
ax[0,0].set_xlim(1, 4)

ax[0,1].plot(t, scaleMax1(abs_sqd(SI_ft)))
ax[0,1].plot(t, acfilter)
ax[0,1].set_xlim(-200, 200)

ax[0,0].plot(w, 4*np.abs(SI_ft_acfilt), color='r')

ax[1,0].plot(w, phidiff, color='b')
ax[1,0].plot(w, phisi, color='r')
ax[1,0].set_ylim(-10, 50)


