# -*- coding: utf-8 -*-
"""
Created on Sun May  7 11:47:34 2023

@author: Juan Diego Bohórquez

Archivo que contiene variables globales, principalmente rutas para la propagación
de onda acústica.
"""
import os

#======================== Revisar estas variables ========================
experimento = "6"
carpeta_img = "06 jun 2023"
#=========================================================================

modelo_vel_name = f"modelo_velocidad_{experimento}"

path_folder = os.getcwd()
numero_experimento = f"experimento_{experimento}"
path_experimento = os.path.join(path_folder, f"experimentos\{numero_experimento}")

# Paths a las subcarpetas
path_imgs = os.path.join(path_experimento, "images")
path_wavefield = os.path.join(path_experimento, "wavefields")
path_seismograms = os.path.join(path_experimento, "seismograms")
path_velocity = os.path.join(path_experimento, f"velocity_models\{modelo_vel_name}.bin")