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

phiw = pulse_phase_ceo(w-w0, np.array([0, 0, 20]))  # define a Taylor phase
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




