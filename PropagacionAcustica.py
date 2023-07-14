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
from prettytable import PrettyTable
#%%
t = TicToc()

class PropagacionAcustica:
    path_folder = os.getcwd()
    path_propagador = os.path.join(path_folder, "propagation-cpml-C")
    
    def __init__(self, fq, nz, nx, nt, size_x, size_z, sx, sz, dh, dt, fuente, 
                 n_abs, modelo_vel, modelo_vel_path):
        
        self.fecha, self.hora = self.fecha_hora()
        self.fq = fq
        self.nz = nz
        self.nx = nx
        self.nt = nt
        self.size_x = size_x
        self.size_z = size_z
        self.sx = sx
        self.dh = dh
        self.dx = self.dz = self.dh
        self.dt = dt
        self.sz = sz
        self.fuente = fuente # gaussian_neg gaussian, ricker, gaussian-neg
        
        self.n_abs = n_abs
        self.n_absx = self.n_abs
        self.n_absx = self.n_abs
        
        # Modelo de velocidad
        self.modelo_vel = modelo_vel
        self.modelo_vel_path = modelo_vel_path
        self.vel_path_c = self.__export_velocity()
        
        # Tiempo de propagacion
        self.t_total = self.nt * self.dt
        
        # Coordenadas del modelo completo 1 a 1
        self.xxs, self.zzs = np.meshgrid(np.linspace(0, self.nx, self.nx), 
                                         np.linspace(0, self.nz, self.nz))
        
        
        
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
            "[[velocity_path]]": f'"{self.vel_path_c}"',
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
        t.tic()
        os.system(f'wsl ~ -e sh -c "{lanzar_propagador}"')
        tiempo = t.tocvalue()
        
        self.tiempo_computo = tiempo
        
    def __export_velocity(self):
        modelo = self.modelo_vel
        path_dest = self.modelo_vel_path
        
        path_to_c = path_dest.split("\\")[-4:]
        path_to_c = "/".join(path_to_c)
        path_to_c = f"../{path_to_c}"
        
        self.__save_bin(modelo, path_dest)
        return path_to_c
    
    def __save_bin(self, data, path_dest):
        output_file = open(path_dest, "wb")
        data.T.tofile(output_file)
        output_file.close()
    
    def obtener_datos(self):
        
        filename = os.path.join(self.path_propagador, "propagacion2.bin")

        with open(filename, "rb") as fid:
            data = np.fromfile(fid, dtype=np.float32)
        
            num_rows = int(data.shape[0] / self.nz)
            data.resize(num_rows, self.nz)
            data = data.T
        
            cubo = crear_cubo(data, self.nx, self.nz, self.nt)
        
        self.calculo_gradiente(cubo)      
        self.cubo_propagacion = cubo
        
        return cubo
    
    # Definición del decorador para agregar información adicional
    def plot_campo(self, P, title, name, snap=0, save=False, path='', seismogram=False, **seismograms):
        fig = plt.figure()
        plt.contourf(self.xxs * self.dx, self.zzs * self.dz, P.reshape((self.xxs.shape)), 100, cmap="jet")
        plt.ylabel("z")
        plt.xlabel("x")
        plt.colorbar()
        plt.axis("scaled")
        plt.title(title)
        
        if seismogram:
            plt.plot(seismograms['x_pos'], seismograms['z_pos'], "r*", markersize=4, label="Receptores")
            if save:
                plt.savefig(
                    f"{path}\\{name}.png", bbox_inches="tight", dpi=320
                )
    
        elif save:
            plt.savefig(
                f"{path}\\{name}_{snap}.png", bbox_inches="tight", dpi=320
            )
        plt.show()
        
        # plt.close(fig)
    

    def plot_snapshots(self, time_list, save, path=None):
            
        t_index = [int(time / self.dt) for time in time_list]

        if save and path is None:
            raise Exception("Por favor especifique una ruta para guardar las imagenes.")
        else:
            for index, time in enumerate(t_index):
                snap = time
                P = self.cubo_propagacion[:, :, snap]
    
                self.plot_campo(
                    P,
                    title = f"Snap {snap+1}/{self.nt}. T = {round(snap*self.dt, 3)}",
                    name ="snap",
                    snap = index + 1,
                    save = save,
                    path = path
                )
        
        self.snap_time = time_list
    
    def calculo_gradiente(self, data):
        grad_z, grad_x = np.gradient(data, axis=(0, 1))
        
        self.grad_z = grad_z
        self.grad_x = grad_x
        
          
    def componentes_campo(self, time_list, save, export, 
                          path_export=None, path=None):
        
        componentes_total = []
        t_index = [int(time / self.dt) for time in time_list]
        
        # Array para concatenar
        campos_x = np.empty((self.nz, self.nx*len(t_index)))
        campos_z = np.empty((self.nz, self.nx*len(t_index)))
        
        ini = 0
        cols = self.nx
        for i, time_i in enumerate(t_index):
            snap = time_i
            
            Ux_orig = self.grad_x[:, :, snap]
            Uz_orig = self.grad_z[:, :, snap]
            
            campos_x[:, ini:ini+cols] = Ux_orig
            campos_z[:, ini:ini+cols] = Uz_orig
            
            ini += cols
            
        # Escalado de los campos y división del array
        campos_x_es = np.split(self.escalar_intervalo(campos_x, -1, 1), len(t_index), axis=1)
        campos_z_es = np.split(self.escalar_intervalo(campos_z, -1, 1), len(t_index), axis=1)
        
        for i, time_i in enumerate(t_index):
            snap = time_i
    
            # Magnitud del campo de desplazamiento escalado
            Ux_snap = campos_x_es[i]
            Uz_snap = campos_z_es[i]
            U0_mag = np.sqrt(Ux_snap**2 + Uz_snap**2)
    
            U_comp = {
                "Componente x": Ux_snap,
                "Componente z": Uz_snap,
                "Magnitud $\phi$": U0_mag,
            }
            
            # Extracción de componentes como matrices de (nx*nz,2)
            U0x = Ux_snap.reshape(-1, 1)
            U0z = Uz_snap.reshape(-1, 1)
            U0 = np.concatenate((U0x, U0z), axis=1)
            componentes_total.append(U0)
            
            # Guardado de los datos
            if export and path_export is None:
                raise Exception("Por favor especifique una ruta para guardar las componentes de los campos.")
            elif export and path_export is not None:
                np.savetxt(f"{path_export}/componentes_campo_{i+1}.txt", U0)
                
            self.plot_componente(U_comp, save, i+1, path)
            
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
            
    def coordenadas_campo(self, export=False, path_export=None):
        xxss, zzss = np.meshgrid(np.linspace(0, self.size_x, self.nx),
                                 np.linspace(0, self.size_z, self.nz))

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
        
    def sismogramas(self, pos_x, pos_z):
        # Posiciones de los sismogramas con respecto al cubo (indices)
        xsf_id = [int(x * (self.nx - 1) / self.size_x) for x in pos_x]
        zsf_id = [int(z * (self.nz - 1) / self.size_z) for z in pos_z]
        
        grad_z, grad_x = self.grad_x, self.grad_z
        
        sismogramas_x = np.zeros((0,))
        sismogramas_z = np.zeros((0,))
        
        # Concatenación de sismogramas
        for i in range(len(pos_x)):
            u_cubo_x = grad_x[zsf_id[i], xsf_id[i], :]
            u_cubo_z = grad_z[zsf_id[i], xsf_id[i], :]
            
            sismogramas_x = np.concatenate((sismogramas_x, u_cubo_x), axis=0)
            sismogramas_z = np.concatenate((sismogramas_z, u_cubo_z), axis=0)
            
        # Escalado
        sismogramas_x = self.escalar_intervalo(sismogramas_x, -1, 1)
        sismogramas_z = self.escalar_intervalo(sismogramas_z, -1, 1)
        
        self.sismogramas_x = sismogramas_x
        self.sismogramas_z = sismogramas_z
            
        return sismogramas_x, sismogramas_z
    
    def exportar_sismogramas(self, n_seis, path_export):
        sub_sismogramas_x = np.split(self.sismogramas_x, n_seis)
        sub_sismogramas_z = np.split(self.sismogramas_z, n_seis)
                
        tt = np.linspace(0, self.nt * self.dt, self.nt)
        
        for i in range(n_seis):
            sismograma_x = np.concatenate((tt.reshape(-1, 1),
                                           sub_sismogramas_x[i].reshape(-1, 1)), axis=1)
        
            sismograma_z = np.concatenate((tt.reshape(-1, 1),
                                           sub_sismogramas_z[i].reshape(-1, 1)), axis=1)
        
            # Guardar el sismograma individualmente
            nombre_x_temp = f"{path_export}/JDX.semd"
            nombre_z_temp = f"{path_export}/JDZ.semd"
        
            nombre_x_fin = f"{path_export}/JD.S{i+1:04d}.BXX.semd"
            nombre_z_fin = f"{path_export}/JD.S{i+1:04d}.BXZ.semd"
        
            self.__guardar_sismograma(sismograma_x, nombre_x_temp, nombre_x_fin)
            self.__guardar_sismograma(sismograma_z, nombre_z_temp, nombre_z_fin)
        
        self.n_seis = n_seis

    
    def __guardar_sismograma(self, sismograma, ruta_temporal, ruta_final):
        if os.path.exists(ruta_final):
            os.remove(ruta_final)
        np.savetxt(ruta_temporal, sismograma)
        os.rename(ruta_temporal, ruta_final)
    

    def plot_xz_sismogramas(self, x_pos, z_pos, path):
        self.plot_campo(P = self.modelo_vel, 
                        title = fr"Modelo de velocidad $\alpha$ - {self.nz}$\times${self.nx}",
                        name = "modelo_receptores", 
                        save = True,
                        path = path,
                        seismogram = True,
                        x_pos = x_pos, 
                        z_pos = z_pos)

    def exportar_parametros(self, path_export):
        tabla = PrettyTable()
        
        parametros_log = {
            "Frecuencia de la fuente" : f'{self.fq} Hz',
            "Tipo de fuente" : f'{self.fuente}',
            "Puntos en Z" : f'{self.nz}',
            "Puntos en X" : f'{self.nx}',
            "Puntos en t" : f'{self.nt}',
            "Tamaño en X" : f'{self.size_x*1000} m',
            "Tamaño en Z" : f'{self.size_z*1000} m',
            "Duración total" : f'{self.t_total:.4f} seg',
            "Nodos absorbentes" : f'{self.n_abs}',
            "Número de fuentes" : '1',
            "Número de sismómetros": f'{self.n_seis}',
            "Tamaño de la celda": f'{self.dh:.4f} m',
            "Coordenada x fuente": f'{self.sx*self.dx*1000} m',
            "Coordenada z fuente": f'{self.sz*self.dz*1000} m',
        }
        
        tabla.add_column("Parámetros", list(parametros_log.keys()))
        tabla.add_column("Valor", list(parametros_log.values()))
        tabla.align["Parámetros"] = "l"
        
        # Ruta y nombre del archivo
        archivo = f"{path_export}/datos_propagacion.txt"
        contenido = f"Parámetros de la propagación, {self.fecha} {self.hora}\n" 
        
        contenido += f"\n{tabla.get_string()}\n"
        contenido += f"\nTiempo de propagación {self.tiempo_computo:.4f} segundos"
            
        # Guardar el contenido en el archivo
        with open(archivo, "w") as file:
            file.write(contenido)
            
    
    def exportar_csv(self, path_export):
        parametros_basicos = {
            "Frecuencia" : self.fq,
            "Tipo fuente" : self.fuente,
            "Puntos Z" : self.nz,
            "Puntos X" : self.nx,
            "Puntos t" : self.nt,
            "Tamaño X" : self.size_x*1000,
            "Tamaño Z" : self.size_z*1000,
            "Duración" : self.t_total,
            "Nodos abs" : self.n_abs,
            "Numero fuentes" : 1,
            "Numero sismometros": self.n_seis,
            "Tamaño celda x": self.dx,
            "Tamaño celda z": self.dz,
            "Paso temporal": self.dt,
            "Coord x fuente": self.sx*self.dx*1000,
            "Coord z fuente": self.sz*self.dz*1000,
            "t01": self.snap_time[0],
            "t02": self.snap_time[1],
            "t03": self.snap_time[2],
            }
        
        df = pd.DataFrame(parametros_basicos, index=[0])

        # Ruta del archivo CSV
        archivo_csv = f"{path_export}/datos_propagacion.csv"
        
        df.to_csv(archivo_csv, index=False)
    
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
