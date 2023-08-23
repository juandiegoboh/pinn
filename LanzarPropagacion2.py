# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:48:25 2023

@author: Juan Diego Bohórquez
"""
import numpy as np
import os
import matplotlib.pyplot as plt

from CreacionModeloVelocidad import ModeloVelocidad
from scipy.ndimage import zoom
from PropagacionAcustica import PropagacionAcustica
from _global import path_experimento, path_seismograms, path_imgs, path_wavefield, path_velocity
from prettytable import PrettyTable

#%%
# Carpeta del experimento
if not os.path.exists(path_experimento):
    subfolders = ["seismograms", "velocity_models", "wavefields", "images"]

    for item in subfolders:
        path = os.path.join(path_experimento, item)
        os.makedirs(path)
        
# %% Carpeta de imagenes
if not os.path.exists(path_imgs):
    # Si no existe, se crea
    os.makedirs(path_imgs)  
    
#%% Parametros de la propagación
fq = 16   # Frecuencia (Hz)
nz = 150  # Puntos del modelo en z
nx = nz * 3  # Puntos del modelo en x (en specfem es 100)
nt = 800  # Puntos del modelo en t
tipo_fuente = "gaussian_neg" # gaussian, ricker, gaussian_neg

n_abs = 20  # nodos para absorber condiciones de frontera en ambas direcciones
# n_abs = 20  # nodos para absorber condiciones de frontera en ambas direcciones
n_absx = n_abs  # nodos del lado izquierdo del dominio
n_absz = n_abs  # el límite superior no absorbe

n_event = 1  # numero de eventos sísmicos

ax_spec = 1.5  # tamaño del dominio antes de eliminar las regiones absorbentes
az_spec = 0.5
xsf = 1.3  # ubicación en x de todos los sismómetros

dh = ax_spec / nx  # Tamaño de la celda
dx = dz = dh
dt = dh * 1000 / (np.amax(3000) * np.sqrt(2))  # Intervalo temporal

# Coordenadas de la fuente (en indices de la matriz)
sx = int(nx / 3)
sz = int(nz / 2)

# Dimensión del dominio para entrenamiento de PINNs.
ax = xsf - n_absx * dx
# solo se elimina el grosor del la frontera absorbente de la izquierda ya que #xsf es (debe ser) más pequeño que donde comienza la frontera absorbente del lado derecho.
az = az_spec - n_absz * dz 

#%% Modelo de velocidad 
velocidad = ModeloVelocidad(nx, nz, dx, dz, sx, sz)

# Modelo de Rash
alpha_true1 = velocidad.cargar_modelo_rash("event1/modelo_vel.npy", ax/dx, az/dz, order=1, mode="edge")

velocidad.plot_vel(n_absx, n_absz, ax, az, path_imgs, "alpha_true0_original", save=True)


#%% Propagacion
propagacion = PropagacionAcustica(fq, nz, nx, nt, ax_spec, az_spec, sx, sz, dh, 
                                  dt, tipo_fuente, n_abs, alpha_true1, path_velocity)

propagacion.lanzar_propagacion()

#%% Cubo
cubo = propagacion.obtener_datos()  # Esta funcion también calcula los gradientes para las componentes

#%% Obtener snapshots
t01 = 0.1
t02 = 0.115
t_la = 0.25

propagacion.plot_snapshots([t01, t02, t_la], save=True, path=path_imgs)

#%% Obtener componentes - Las componentes vienen escaladas con base al número de snaps
componentes = propagacion.componentes_campo([t01, t02, t_la], save=True, export=True,
                              path_export=path_wavefield, path=path_imgs)

#%% Extraer coordenadas
coordenadas = propagacion.coordenadas_campo(path_export=path_wavefield)

#%% Ver modelo 3D
propagacion.modelo3D(cubo)

#%% Sismograma prueba

figure = plt.figure()
plt.plot(cubo[1, 1, :])
plt.show()

#%% Sismogramas
# Posiciones de los sismogramas en coordenadas
n_seis = 20

z0_s = az_spec - 0.003      # z ubicación del 1er sismómetro, 3m debajo de la superficie.
zl_s = 0.01 + n_absz * dz   # z ubicación del último sismómetro, 10m antes de la cpml.

xsf = 1.3 # Posicion x de los sismogramas

xsf_arr = np.array([xsf] * n_seis)
zsf = np.linspace(z0_s, zl_s, n_seis)

sismogramas_x, sismogramas_z = propagacion.sismogramas(xsf_arr, zsf)
propagacion.exportar_sismogramas(n_seis, path_seismograms)

#%% Plot del modelo con los sismogramas
propagacion.plot_xz_sismogramas(xsf_arr, zsf, path_imgs)

#%% Exportar parametros de la propagacion
propagacion.exportar_parametros(path_experimento)
propagacion.exportar_csv(path_experimento)



