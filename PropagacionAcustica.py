# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:06:00 2023

@author: Juan Diego Bohórquez

Código 
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime
from multireplace import multireplace
from pytictoc import TicToc
from numba import njit
from vedo import Volume, Text2D
from vedo.applications import Slicer3DPlotter

#%%

class PropagacionAcustica:
    path_folder = os.getcwd()
    path_propagador = os.path.join(path_folder, "propagation-cpml-C")
    
    def __init__(self, fq, nz, nx, nt, sx, sz, dh, dt, fuente, n_abs, modelo_vel_path):
        self.fecha, self.hora = self.fecha_hora()
        self.fq = fq
        self.nz = nz
        self.nx = nx
        self.nt = nt
        self.sx = sx
        self.dh = dh
        self.dx = self.dz = self.dh
        self.dt = dt
        self.sz = sz
        self.fuente = fuente # gaussian_neg gaussian, ricker, gaussian-neg
        
        self.n_abs = n_abs
        self.n_absx = self.n_abs
        self.n_absx = self.n_abs
        
        self.modelo_vel_path = self.path_to_c(modelo_vel_path)
        
        # Coordenadas del modelo completo 1 a 1
        self.xxs, self.zzs = np.meshgrid(np.linspace(0, self.nx, self.nx), 
                                         np.linspace(0, self.nz, self.nz))
        
        # Lanzar propagacion
        # self.lanzar_propagacion()
  
    def fecha_hora(self):
        now = datetime.now()
        fecha = now.strftime("%d-%m-%Y")
        hora = now.strftime("%I:%M:%S %p")
        return fecha, hora
        
    def reemplazar_variables(self, template, final, replacements):
        with open(template, "rt") as file:
            data = file.read()
    
            data = multireplace(data, replacements)
    
        with open(final, "wt") as file:
            file.write(data)

    def lanzar_propagacion(self):
        template_path = os.path.join(self.path_propagador, "principal_template.c")
        final_path = os.path.join(self.path_propagador, "principal2.c")
        replacements = {
            "[[Nx]]": f"{self.nx}",
            "[[Nz]]": f"{self.nz}",
            "[[fq]]": f"{self.fq}",
            "[[Sx]]": f"{self.sx}",
            "[[Sz]]": f"{self.sz}",
            "[[dh]]": f"{self.dh*1000}",
            "[[Tout]]": f"{self.nt}",
            "[[n_abs]]": f"{float(self.n_abs)}",
            "[[velocity_path]]": f'"{self.modelo_vel_path}"',
        }
        
        self.reemplazar_variables(template_path, final_path, replacements)
        
        template_path = os.path.join(self.path_propagador, "funciones2_template.h")
        final_path = os.path.join(self.path_propagador, "funciones2.h")
        replacements = {"[[fuente]]": f"{self.fuente}"}
        
        self.reemplazar_variables(template_path, final_path, replacements)
    
        # =============================================================== #
        path_linux = r"/mnt/c/users/juan9/OneDrive\ -\ UNIVERSIDAD\ INDUSTRIAL\ DE\ SANTANDER/Academico/Maestria\ Geofisica/Tesis/Codigo/Juan-Diego/propagation-cpml-C"
    
        lista_comandos = [
            f"cd {path_linux}",
            "cat compilar2.txt",
            "gcc principal2.c CPML.c -lm -Wall -o principal2",
            "./principal2",
            "explorer.exe .",
        ]
    
        lanzar_propagador = "; ".join(lista_comandos)
    
        # Comando en wsl
        os.system(f'wsl ~ -e sh -c "{lanzar_propagador}"')
        
    def path_to_c(self, path):
        '''Path relativo para usar en el codigo de C, 
        solo funciona con la configuracion actual de carpetas'''
        path_to_c = path.split("\\")[-4:]
        path_to_c = "/".join(path_to_c)
        path_to_c = f"../{path_to_c}"
        
        return path_to_c
    
    def obtener_datos(self):
        
        filename = os.path.join(self.path_propagador, "propagacion2.bin")

        with open(filename, "rb") as fid:
            data = np.fromfile(fid, dtype=np.float32)
        
            num_rows = int(data.shape[0] / self.nz)
            data.resize(num_rows, self.nz)
            data = data.T
        
            cubo = crear_cubo(data, self.nx, self.nz, self.nt)
        
        return cubo
    
    def plot_campo(self, P, xx, zz, dx, dz, title, name, snap, save, path):
        fig = plt.figure()
        plt.contourf(xx * dx, zz * dz, P.reshape((xx.shape)), 100, cmap="jet")
        plt.ylabel("z")
        plt.xlabel("x")
        plt.colorbar()
        plt.axis("scaled")
        plt.title(title)
        plt.show()
    
        if save:
            plt.savefig(
                f"{path}\\{name}_{snap}.png", bbox_inches="tight", dpi=320
            )
        # plt.close(fig)
        
    def snapshots(self, data, time_list, save, path=None):
            
        t_index = [int(time / self.dt) for time in time_list]

        if save and path is None:
            raise Exception("Por favor especifique una ruta para guardar las imagenes.")
        else:
            for index, time in enumerate(t_index):
                snap = time
                P = data[:, :, snap]
    
                self.plot_campo(
                    P,
                    self.xxs,
                    self.zzs,
                    self.dx,
                    self.dz,
                    title = f"Snap {snap+1}/{self.nt}. T = {round(snap*self.dt, 3)}",
                    name ="snap",
                    snap = index + 1,
                    save = save,
                    path = path
                )
            
    def componentes_campo(self, data, time_list, save, export, 
                          path_export=None, path=None):
        
        componentes_total = []
        t_index = [int(time / self.dt) for time in time_list]
        
        for i, time in enumerate(t_index):
            snap = time
            P = data[:, :, snap]
    
            # Calculo de la variación del campo en ambas direcciones x e z.
            Uz, Ux = np.gradient(P)
    
            # Escalar al intervalo [-1, 1]
            Ux_scaled = self.escalar_intervalo(Ux, -1, 1)
            Uz_scaled = self.escalar_intervalo(Uz, -1, 1)
    
            # Magnitud del campo de desplazamiento
            # U0_mag = np.sqrt(Ux**2 + Uz**2)
            U0_mag_scaled = np.sqrt(Ux_scaled**2 + Uz_scaled**2)
    
            U_comp = {
                "Componente x": Ux_scaled,
                "Componente z": Uz_scaled,
                "Magnitud $\phi$": U0_mag_scaled
            }
            
            # Extracción de componentes como matrices de (nx*nz,2)
            U0x = Ux_scaled.reshape(-1, 1)
            U0z = Uz_scaled.reshape(-1, 1)
            U0 = np.concatenate((U0x, U0z), axis=1)
            componentes_total.append(U0)
            
            # Guardado de los datos
            if export and path_export is None:
                raise Exception("Por favor especifique una ruta para guardar las componentes de los campos.")
            elif export and path_export is not None:
                np.savetxt(f"{path_export}/componentes_campo_{i+1}.txt", U0)
                
            self.plot_componente(U_comp, save, path, time=i+1)
            
        return componentes_total


    def escalar_intervalo(self, matriz: np.ndarray, min_intervalo: float, 
                          max_intervalo: float):
            
        min_valor = np.min(matriz)
        max_valor = np.max(matriz)
        rango = max_valor - min_valor

        # Escalado manual
        matriz_escalada = ((matriz - min_valor) / rango) * \
            (max_intervalo - min_intervalo) + min_intervalo

        return matriz_escalada
    
    def plot_componente(self, U_comp, save, time, path=None):
        fig, axs = plt.subplots(3, 1, figsize=(5, 7))
        j = 0

        for key, value in U_comp.items():
            a = axs[j].contourf(self.xxs * self.dh, self.zzs * self.dh,
                                value.reshape((self.xxs.shape)), 100, cmap="jet")
            axs[j].axis("scaled")
            axs[j].set(title=f"{key}")
            j += 1

            # Formato colorbar
            ticks = 6
            ticks_space = (np.max(value) - np.min(value)) / ticks
            ticks_list = np.arange(np.min(value), np.max(value), ticks_space)

            plt.colorbar(
                a,
                aspect=18,
                fraction=0.10,
                shrink=0.8,
                ticks=ticks_list,
                format="{x:.2f}",
            )
        plt.show()
            
        if save and path is None:
            raise Exception("Por favor especifique una ruta para guardar las imagenes.")
        else:
            fig.savefig(f"{path}\\componente_{time}_propio_completo.png",
                        bbox_inches="tight", dpi=320)
            
    def coordenadas_campo(self, ax, az, export=False, path_export=None):
        xxss, zzss = np.meshgrid(np.linspace(0, ax, self.nx),
                                 np.linspace(0, az, self.nz))

        coord_campo_x = xxss.reshape(-1, 1)
        coord_campo_z = zzss.reshape(-1, 1)
        coords_campo = np.concatenate((coord_campo_x, coord_campo_z), axis=1)

        if export and path_export is None:
            raise Exception("Por favor especifique una ruta para guardar las coordenadas.")
        elif export and path_export is not None:
            np.savetxt(f"{path_export}/xz_componentes_campos.txt", coords_campo)
        
        return coords_campo
    
    def modelo3D(self, data):
        vol = Volume(data)

        plot3D = Slicer3DPlotter(
            vol,
            bg="white",
            bg2="lightblue",
            cmaps=("gist_ncar_r", "jet", "Spectral_r", "hot_r", "bone_r"),
            use_slider3d=False,
        )

        plot3D.close()
        
    def sismogramas(self, data, ax, az, pos_x, pos_z):
        # Posiciones de los sismogramas con respecto al cubo (indices)
        xsf_id = [int(x * (self.nx - 1) / ax) for x in pos_x]
        zsf_id = [int(z * (self.nz - 1) / az) for z in pos_z]
        
        grad_z, grad_x = np.gradient(data, axis=(0, 1))
        
        sismogramas_x = np.zeros((0,))
        sismogramas_z = np.zeros((0,))
        
        tt = np.linspace(0, self.nt * self.dt, self.nt)
        
        # Concatenación de sismogramas
        for i in range(len(pos_x)):
            u_cubo_x = grad_x[zsf_id[i], xsf_id[i], :]
            u_cubo_z = grad_z[zsf_id[i], xsf_id[i], :]
            
        return xsf_id, zsf_id
    
#%%
@njit
def crear_cubo(datos: np.array, nx, nz, nt):
    # El cubo queda con dimensiones Nz * Nx * Nt
    cubo = np.zeros((nz, nx, nt), np.float32)

    for k in range(nt):
        for j in range(nx):
            for i in range(nz):
                cubo[i, j, k] = datos[i, j + (k - 1) * nx]
                    
    return cubo
