# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 12:32:31 2023

@author: Juan Diego Bohórquez
"""
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

cubo_path = r"C:\Users\juan9\OneDrive - UNIVERSIDAD INDUSTRIAL DE SANTANDER\Academico\Maestria Geofisica\Tesis\Codigo\Juan-Diego\propagation-cpml-C-original\propagacion2.bin"

def obtener_datos(nx,nz,nt):
    
    filename = cubo_path

    with open(filename, "rb") as fid:
        data = np.fromfile(fid, dtype=np.float32)
    
        num_rows = int(data.shape[0] / nz)
        data.resize(num_rows, nz)
        data = data.T
    
        cubo = crear_cubo(data, nx, nz, nt)
       
        return cubo

@njit
def crear_cubo(datos: np.array, nx, nz, nt):
    # El cubo queda con dimensiones Nz * Nx * Nt
    cubo = np.zeros((nz, nx, nt), np.float32)

    for k in range(nt):
        for j in range(nx):
            for i in range(nz):
                cubo[i, j, k] = datos[i, j + (k - 1) * nx]
                    
    return cubo

nx = 200
nz = 200
nt = 400

cubo = obtener_datos(nx,nz,nt)

#%%
plt.figure()
plt.imshow(cubo[:,:,399], cmap="jet")
plt.colorbar()
plt.show()




