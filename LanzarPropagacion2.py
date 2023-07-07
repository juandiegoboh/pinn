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
from _global import carpeta_img, path_folder, modelo_vel_name, path_experimento, path_imgs, path_wavefield, path_velocity

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
nt = 650  # Puntos del modelo en t
tipo_fuente = "gaussian_neg" # gaussian, ricker, gaussian-neg

n_abs = 20  # nodos para absorber condiciones de frontera en ambas direcciones de specfem
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
velocidad = ModeloVelocidad(nx, nz)

# Modelo de Rash
alpha_true1 = velocidad.cargar_modelo_zoom("event1/modelo_vel.npy", ax/dx, az/dz, path_velocity)

#%% Propagacion
propagacion = PropagacionAcustica(fq, nz, nx, nt, sx, sz, dh, dt, tipo_fuente, n_abs, path_velocity)
# propagacion.lanzar_propagacion()

#%% Cubo
cubo = propagacion.obtener_datos()

#%% Obtener snapshots
t01 = 0.1
t02 = 0.115
t_la = 0.25
propagacion.snapshots(cubo, [t01, t02, t_la], save=True, path=path_imgs)

#%% Obtener componentes
componentes = propagacion.componentes_campo(cubo, [t01, t02, t_la], save=True, export=True,
                              path_export=path_wavefield, path=path_imgs)

#%% Extraer coordenadas
coordenadas = propagacion.coordenadas_campo(ax_spec, az_spec, export=True, 
                                            path_export=path_wavefield)

#%% Ver modelo 3D
# propagacion.modelo3D(cubo)

#%% Sismogramas
# Posiciones de los sismogramas en coordenadas
n_seis = 20
xsf_arr = np.array([1.3] * n_seis)
zsf = np.linspace(0, az, 20)

xsf_id, zsd_id = propagacion.sismogramas(cubo, ax_spec, az_spec, xsf_arr, zsf)
coordenadas = list(zip(zsd_id, xsf_id))

matriz = cubo[:,:,150]
mascara = np.zeros_like(matriz, dtype=bool)
mascara[tuple(zip(*coordenadas))] = True

matriz_enmascarada = np.ma.masked_array(matriz, mask=mascara)

plt.figure()
im = plt.imshow(matriz_enmascarada)

# Aplicar la máscara de color al gráfico
# im.set_cmap('Set1')  # Establecer el mapa de colores (puedes elegir cualquier otro mapa)
# im.set_clim(0, 9)  # Establecer los límites de color (ajústalos según tus necesidades)

plt.colorbar(im)
plt.show()
