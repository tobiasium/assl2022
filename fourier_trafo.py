"""
Functions related to Fourier transforms.

author: Tobias Witting witting@mbi-berlin.de
version history:
2020-06-05: created
"""
import numpy as np


# def conj_axis(t):
#     dt = np.mean(np.diff(t))
#     N = len(t)
#     dw = 2*np.pi/N/dt
#     w = (np.arange(N)-N/2+0.5)*dw
#     return w

def conj_axis(t, shift=None):
    """
    Return an omega vector for given time vector.

    When the vector length is odd omega needs to be
    shifted by 1/2 delta omega
    then a DC peak sits exactly at omega=0 in Fourier space.
    """
    dt = np.mean(np.diff(t))
    N = len(t)
    dw = 2*np.pi/N/dt
    if shift is None:
        if len(t) % 2 == 0:
            shift = 0
        else:
            shift = 0.5
    w = (np.arange(N) - N/2 + shift) * dw
    return w


def t2f(x, axis=-1):
    """Fourier transform from time to omega."""
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def f2t(x, axis=-1):
    """Fourier transform from omega to time."""
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def t2f_2d(x, axes=(-2,-1)):
    """2D Fourier transform from time to omega."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)


def f2t_2d(x, axes=(-2, -1)):
    """2D Fourier transform from omega to time."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)

