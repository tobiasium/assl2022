"""
Commonly used unit conversions.
"""
from numpy import pi
from numpy import sqrt

__author__ = "Tobias Witting"
__copyright__ = "Copyright 2018"
__credits__ = ["Tobias Witting"]
__license__ = "none"
__version__ = "1.0"
__maintainer__ = "Tobias Witting"
__email__ = "witting@mbi-berlin.de"
__status__ = "Production"

import numpy as np
#from generaltools import find_index
np.seterr(divide='ignore', invalid='ignore')

C = 299.792458  # speed of light in sinnvoller Einheit nm/fs
CSI = 299792458
PI = pi
JEV = 6.24150974e18  # energy of 1 J in eV
HBAR = 1.05457148e-34  # hbar in m^2 kg s^-1
AUTIME = 2.418884326505e-2
AUEV = 27.2113835095688
# eps = np.finfo(np.float64).eps

def nm2radfs(x):
    #x[find_index(x, 0.)] = eps
    return 2*PI*C/x

def radfs2nm(x):
    #x[find_index(x, 0.)] = eps
    return 2*PI*C/x

def nm2eV(x):
    return 2*PI*CSI/(x*1e-9)*JEV*HBAR

def eV2nm(x):
    return 2*PI*CSI/(x*1e-9)*JEV*HBAR

def radfs2eV(x):
    return x*(JEV*HBAR*1e15)

def eV2radfs(x):
    return x/(JEV*HBAR*1e15)

def eV2au(x):
    return x/AUEV

def au2eV(x):
    return x*AUEV

def fs2au(x):
    return x/AUTIME

def au2fs(x):
    return x*AUTIME

def nm2au(x):
    return 2*PI*CSI/(x*1e-9)*JEV*HBAR/AUEV

def au2nm(x):
    return 2*PI*CSI/(x*1e-9)*JEV*HBAR/AUEV

def Wsqcm2au(x):
    return sqrt(x/3.509e16)

def au2Wsqcm(x):
    return x**2 * 3.509e16

def radfs2cm1(x):
    return x / (2*PI*C*1e-7)