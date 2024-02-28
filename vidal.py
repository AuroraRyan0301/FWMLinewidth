import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import math
import pandas as pd
import scipy
import cmath
import progressbar
from tqdm import tqdm
from utils import *

from scipy.integrate import quad

import constants 

# define a sub-class of FWMpool named VidalPool
class VidalPool(FWMpool):
    def __init__(self, elename, EndEnergyWavNum, FstateFilepath, eleCondition, gamma_a, gamma_b, gamma_c):
        super().__init__(elename, EndEnergyWavNum, FstateFilepath, eleCondition)
        self.gamma_b = gamma_b
        self.gamma_a = gamma_a
        self.gamma_c = gamma_c

def F_shape_func(b,deltak):
    bdeltak = b*deltak
    if bdeltak > 0:
        return 0
    else:
        # absorb L**2 in ahead coeff
        return constants.Pi**2 /4 * (bdeltak)**2 * math.exp(bdeltak) * b**2

def Vidal_phi_s(omega_s, n_s, Pool_density, Kai_third, phi_1, phi_2, phi_3, n_1, n_2, n_3, F_shape_func):
    return abs(24*np.pi**2*omega_s/(constants.c**2*n_s)*Pool_density*Kai_third)**2 * n_s * phi_1* phi_3 * phi_2/n_1/n_3/n_2*F_shape_func

def power_in_SI_to_gauss(si_power):
    return si_power * 1e4 / constants.c

def Kai_T_third(omega1, omega2, omega3,Pool: VidalPool):
    gamma_a = Pool.gamma_a
    gamma_b = Pool.gamma_b
    gamma_c = Pool.gamma_c
    omega_bg = Pool.b - gamma_b * 1j
    sum_all_path = 0
    for a_state in range(len(Pool.channels)):
        for c_state in range(len(Pool.channels)):
            w_ag = Pool.channels[a_state]['f_energy_wavnum']
            w_cg = Pool.channels[c_state]['f_energy_wavnum']
            # ignore the gamma decay of P state
            sum_all_path += Pool.channels[a_state]['gf']*Pool.channels[a_state]['fb'] * Pool.channels[c_state]['gf']*Pool.channels[c_state]['fb'] / \
                (w_ag-gamma_a*1j-omega1) / (w_cg-gamma_c*1j-omega3-omega2-omega1) / (omega_bg-omega2-omega1)
            # can be written as CUDA or multi-thread

    return sum_all_path/constants.hbar**3

            


def Vidal_cal_in_gauss_unit(P1,P2,P3,omega1, omega2, omega3, omega4, FWMPool:VidalPool):
    P1_gauss = power_in_SI_to_gauss(P1)
    P2_gauss = power_in_SI_to_gauss(P2)
    P3_gauss = power_in_SI_to_gauss(P3)
    Pool_density = FWMPool.N
    Pool_density_in_cm = Pool_density * 1e-6

    n_s = RefractiveIndex(omega4,FWMPool)
    n_1 = RefractiveIndex(omega1,FWMPool)
    n_2 = RefractiveIndex(omega2,FWMPool)
    n_3 = RefractiveIndex(omega3,FWMPool)

    y2 = Delta_k(wavenum2roundfreq(omega4),FWMPool) #这里使用Deltak是没有考虑原子密度N的影响
    y1 = Delta_k(wavenum2roundfreq(omega1),FWMPool) #这里使用Deltak是没有考虑原子密度N的影响
    deltak = y2 - y1
    deltak_in_cm = deltak * 1e-2

    confocal_lg_in_cm = FWMPool.confocal * 1e2

    F_shape_f = F_shape_func(confocal_lg_in_cm,deltak_in_cm)
    # import ipdb; ipdb.set_trace()

    Kai_third = Kai_T_third(omega1, omega2, omega3, FWMPool)


    s_output = Vidal_phi_s(omega4, n_s, Pool_density_in_cm, Kai_third, P1_gauss, P2_gauss, P3_gauss, n_1, n_2, n_3, F_shape_f)

    return s_output
