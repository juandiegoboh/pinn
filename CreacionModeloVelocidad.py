# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:07:02 2023

@author: Juan Diego Bohórquez
"""

import os
import numpy as np
import matplotlib.pyplot as plt

#%%
def velocidad_circular(nx, nz, pos_x, pos_z, radio, velocidad, background_vel):
    x0 = pos_x
    y0 = pos_z
    
    # Crear matriz de velocidad
    velocidades = np.ones((nz, nx))*background_vel
    
    # Crear anomalía de velocidad circular
    for i in range(nz):
        for j in range(nx):
            distancia_al_centro = np.sqrt((j - x0) ** 2 + (i - y0) ** 2)
            if distancia_al_centro <= radio:
                velocidades[i, j] = velocidad
                
    return velocidades


def velocidad_eliptica(nx, nz, pos_x, pos_z, eje_mayor, eje_menor, vel, bg_vel):
    x0 = pos_x
    y0 = pos_z
    
    # Crear matriz de velocidad
    velocidades = np.ones((nz, nx), dtype="float32")*bg_vel  # Inicialmente, todas las velocidades son 1
    
    # Crear anomalía de velocidad circular
    for i in range(nz):
        for j in range(nx):
            distancia_x = j - x0
            distancia_y = i - y0
            distancia_normalizada = (distancia_x / eje_mayor) ** 2 + (distancia_y / eje_menor) ** 2
            if distancia_normalizada <= 1:
                velocidades[i, j] = vel
                
    return velocidades


#%%
# prueba = velocidad_circular(200, 100, 100, 40, 25, 2.5, 3)
# prueba2 = velocidad_eliptica(200, 100, 100, 60, 50, 25, 2.5, 3)
# # Visualizar el modelo de velocidad
# plt.imshow(prueba2, cmap='jet', origin='lower')
# plt.colorbar(label='Velocidad')
# plt.xlabel('x')
# plt.ylabel('z')
# plt.title('Modelo de Velocidad del Subsuelo')
# plt.show()

