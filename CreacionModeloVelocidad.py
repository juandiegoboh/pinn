# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:07:02 2023

@author: Juan Diego Bohórquez
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import zoom
#%%
class ModeloVelocidad:
    def __init__(self, nx, nz):
        self.nx = nx
        self.nz = nz


    def velocidad_circular(self, pos_x, pos_z, radio, velocidad, background_vel):
        x0 = pos_x
        y0 = pos_z
        
        # Crear matriz de velocidad
        velocidades = np.ones((self.nz, self.nx))*background_vel
        
        # Crear anomalía de velocidad circular
        for i in range(self.nz):
            for j in range(self.nx):
                distancia_al_centro = np.sqrt((j - x0) ** 2 + (i - y0) ** 2)
                if distancia_al_centro <= radio:
                    velocidades[i, j] = velocidad
                    
        return velocidades
    
    
    def velocidad_eliptica(self, pos_x, pos_z, eje_mayor, eje_menor, vel, bg_vel):
        x0 = pos_x
        y0 = pos_z
        
        # Crear matriz de velocidad
        velocidades = np.ones((self.nz, self.nx), dtype="float32")*bg_vel  # Inicialmente, todas las velocidades son 1
        
        # Crear anomalía de velocidad circular
        for i in range(self.nz):
            for j in range(self.nx):
                distancia_x = j - x0
                distancia_y = i - y0
                distancia_normalizada = (distancia_x / eje_mayor) ** 2 + (distancia_y / eje_menor) ** 2
                if distancia_normalizada <= 1:
                    velocidades[i, j] = vel
                    
        return velocidades
    
    
    def cargar_modelo_zoom(self, path_orig, size_x, size_z, path_dest, order=1):
        alpha_true = np.load(path_orig).astype("float32") * 1000
        
        # Re escalado de alpha true al tamaño del modelo de entrenamiento
        x_zoom = (size_x) / alpha_true.shape[0]
        z_zoom = (size_z) / alpha_true.shape[1]
        
        zoom_factor = [z_zoom, x_zoom]
        alpha_true0 = zoom(alpha_true, zoom_factor, order=order)
        
        # Se completa el modelo de velocidad para el dominio completo
        alpha_true1 = np.pad(alpha_true0, ((0, int(self.nz-size_z)), 
                                           (0,int(self.nx-size_x))), mode='edge')
        
        
        # Se exporta el archivo binario para leer en C
        self.save_bin(alpha_true1, path_dest)
        
        return alpha_true1
    
    
    def save_bin(self, data, path_dest):
        output_file = open(path_dest, "wb")
        data.T.tofile(output_file)
        output_file.close()
