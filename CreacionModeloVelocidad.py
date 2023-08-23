# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:07:02 2023

@author: Juan Diego Bohórquez
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy.ndimage import zoom
#%%
class ModeloVelocidad:
    def __init__(self, nx, nz, dx, dz, sx, sz):
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.sx = sx
        self.sz = sz
        
        # Coordenadas de Receptores
        self.xsf_arr = []
        self.zsf_arr = []
        
        self.modelo = ''


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
                    
        self.modelo = velocidades        
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
                    
        self.modelo = velocidades
        return velocidades
    
    
    def cargar_modelo_rash(self, path_orig, size_x, size_z, order=1, mode='edge'):
        alpha_true = np.load(path_orig).astype("float32") * 1000
        
        # Re escalado de alpha true al tamaño del modelo de entrenamiento
        x_zoom = (size_x) / alpha_true.shape[0]
        z_zoom = (size_z) / alpha_true.shape[1]
        
        zoom_factor = [z_zoom, x_zoom]
        alpha_true0 = zoom(alpha_true, zoom_factor, order=order)
        
        # Se completa el modelo de velocidad para el dominio completo
        nodos_der = int(0.2/self.dx)
        nodos_izq = int(self.nx-size_x)-nodos_der
        alpha_true1 = np.pad(alpha_true0, ((int(self.nz-size_z), 0), (nodos_izq, nodos_der)), mode=mode)
        
        
        # Se exporta el archivo binario para leer en C
        self.modelo = alpha_true1
        
        return alpha_true1
    
    
    def save_bin(self, data, path_dest):
        output_file = open(path_dest, "wb")
        data.T.tofile(output_file)
        output_file.close()
    
    
    def plot_vel(self, n_absx, n_absz, ax, az, path_dest, name, save):
        xxs, zzs = np.meshgrid(np.linspace(0, self.nx, self.nx),
                               np.linspace(0, self.nz, self.nz))
        
        fig, axs = plt.subplots()
        a = axs.contourf(xxs * self.dx, zzs * self.dz,
                     self.modelo.reshape((xxs.shape)), 100, cmap="jet")
        axs.set(xlabel="x", ylabel="z", title=r"Modelo de velocidad ($\alpha$)" + f" - {self.nz}x{self.nx}")
        axs.scatter(self.sx * self.dx, self.sz * self.dz, c="k", label="Fuente")
        
        if self.xsf_arr:
            axs.plot(self.xsf_arr, self.zsf_arr, "r*", markersize=4, label="Receptores")
        
        axs.add_patch(
            Rectangle(
                (n_absx * self.dx, n_absz * self.dz), ax, az,
                fill=False,
                edgecolor="y",
                ls="--",
                label="Entrenamiento",
            )
        )
        
        plt.colorbar(a)
        plt.axis("scaled")
    
        plt.legend(loc='upper left', fontsize='small')
        plt.show()
        
        if save:
            plt.savefig(f"{path_dest}\\{name}.png", bbox_inches="tight", dpi=320)

