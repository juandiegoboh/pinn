# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 18:22:43 2023

@author: Juan Diego Bohórquez
"""

import numpy as np
import matplotlib.pyplot as plt

def ricker(tstep, dt, fq):
    t = dt * tstep - 1 / fq
    f = (1 - 2 * np.pi**2 *fq**2 * t**2) * np.exp(-np.pi**2 * fq**2 * t**2)
    return f

def gaussian(tstep, dt, fq):
    t = dt * tstep - 1 / fq
    f = np.exp(-2 * np.pi**2 * fq**2 * t**2)
    return f

def gaussian_neg(tstep, dt, fq):
    t = dt * tstep - 1 / fq
    f = -np.exp(-2 * np.pi**2 * fq**2 * t**2)
    return f

nt = 550
# f_rick = [ricker(t, 7.856742e-04, 15) for t in range(nt)]
# f_gauss = [gaussian(t,  7.856742e-04, 15) for t in range(nt)]
f_gauss_neg = [gaussian_neg(t,  7.856742e-04, 15) for t in range(nt)]
f_gauss_neg2 = [gaussian_neg(t,  7.856742e-04, 16) for t in range(nt)]

plt.figure()
plt.plot(f_gauss_neg, label='15 Hz')
plt.plot(f_gauss_neg2, label='16 Hz')
plt.legend()
plt.show()

