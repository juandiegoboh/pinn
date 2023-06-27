# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:22:24 2023

Comparación de Campos y Sismogramas generados

@author: Juan Diego Bohórquez
"""

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import pandas as pd

from matplotlib import cm
from matplotlib.patches import Rectangle
from pytictoc import TicToc
from textwrap import fill

from _global import carpeta_img, numero_experimento, path_folder

# %% Rutas

path_experimento = os.path.join(path_folder, f"experimentos\{numero_experimento}")
path_parametros = os.path.join(path_experimento, "datos_propagacion.csv")

path_imgs = os.path.join(path_experimento, "images")

eventos_spec = os.path.join(path_folder, "event1")

sismos_spec = os.path.join(eventos_spec, "seismograms")
campos_spec = os.path.join(eventos_spec, "wavefields")

sismos_jd = os.path.join(path_experimento, "seismograms/")
campos_jd = os.path.join(path_experimento, "wavefields")

# %% Funciones


class GeneracionDatos:
    def __init__(
        self,
        Lx,
        Lz,
        ax_spec,
        az_spec,
        dx,
        dz,
        xsf,
        n_absx,
        n_absz,
        t01,
        t02,
        n_ini,
        nx,
        nz,
        u_scl,
        t_m,
        t_st,
        t_s,
    ):
        self.Lx = Lx
        self.Lz = Lz

        # Tamaño del dominio antes de eliminar las regiones absorbentes (completo)
        self.ax_spec = ax_spec
        self.az_spec = az_spec

        # Pasos espaciales
        self.dx = dx
        self.dz = dz

        # Posicion sismometros
        self.xsf = xsf

        # Nodos absorbentes
        self.n_absx = n_absx
        self.n_absz = n_absz

        # Tamaño del modelo sin fronteras absorbentes
        self.ax = self.xsf - self.n_absx * self.dx
        self.az = self.az_spec - self.n_absz * self.dz

        # Tiempos de los snaps
        self.t01 = t01
        self.t02 = t02

        # Numero de puntos en cada dimensión para interpolar
        self.n_ini = n_ini

        # Tamaño del dominio del propagador
        self.nx = nx
        self.nz = nz
        self.zl_s = 0.06 - self.n_absz * self.dz

        self.u_scl = u_scl

        # inicio de la parte no absorbente del dominio en specfem
        self.xf = self.n_absx * self.dx
        self.zf = self.n_absz * self.dz

        # Grilla no absorbente
        self.xxs, self.zzs, self.xxzzs = self.grilla_no_absorbente()
        self.xx_ni, self.zz_ni, self.xxzz_ni = self.grilla_sin_interpolar()
        self.xx, self.zz = self.grilla_entrenamiento_PINN()

        self.t_m = t_m
        self.t_st = t_st
        self.t_s = t_s

    def plot_componentes(
        self,
        campos: list,
        magnitudes: list,
        save: bool = False,
        nombre: str = "",
        close: bool = False,
    ):
        """
        Función que genera el plot de las componentes del campo de presión.

        Parameters
        ----------
        elemetos_completos : list
            Lista de diccionarios que contienen la información de los campos. Ej:

            campo1_elementos = {"Componente x" : U_ini1x,
                                "Componente z" : U_ini1z,
                                "Magnitud $\phi$": campo_spec1}
        Returns
        -------
        Plot de las componentes de los campos

        """

        for i in range(len(campos)):
            fig, axs = plt.subplots(len(campos), 1, figsize=(5, 7))
            j = 0

            keys = ["Componente x", "Componente z", "Magnitud $\phi$"]
            values = [campos[i][0], campos[i][1], magnitudes[i]]
            elementos = dict(zip(keys, values))

            for k, v in elementos.items():
                a = axs[j].contourf(
                    self.xx * self.Lx,
                    self.zz * self.Lz,
                    v.reshape(self.xx.shape),
                    100,
                    cmap="jet",
                )

                axs[j].axis("scaled")
                axs[j].set(title=f"{k}")
                j += 1

                # Colorbar formato
                ticks = 6
                ticks_space = (np.max(v) - np.min(v)) / ticks
                ticks_list = np.arange(np.min(v), np.max(v), ticks_space)

                plt.colorbar(
                    a,
                    aspect=18,
                    fraction=0.10,
                    shrink=0.8,
                    ticks=ticks_list,
                    format="{x:.2f}",
                )

            plt.tight_layout()

            if save:
                plt.savefig(
                    f"{path_imgs}\\componente_{i+1}_{nombre}.png",
                    bbox_inches="tight",
                    dpi=320,
                )
            plt.show()

            if close:
                plt.close(fig)

    def plot_campos_conjunto(
        self, campos: list, campos_ni: list, title, save, nombre, close=False
    ):
        """
        Función que genera el plot de los campos sin interpolar, y luego de interpolado.

        Parameters
        ----------
        campos : list
            Lista de campos de presión a graficar después de interpolar.
        campos_ni : list
            Lista de campos de presión a graficar antes de interpolar.
        xx_ni : TYPE
            Coordenadas x sin interpolar.
        zz_ni : TYPE
            Coordenadas z sin interpolar.
        xx : TYPE
            Coordenadas x de la grilla de interpolación.
        zz : TYPE
            Coordenadas z de la grilla de interpolación.
        title : str
            Titulo del plot

        Returns
        -------
        Plot comparativo de los campos antes y después de interpolar.

        """
        fig, axs = plt.subplots(
            len(campos), 2, figsize=(10, 4), sharex=True, sharey=True
        )
        row, col = 0, 0

        for n in range(len(campos) * 2):
            ax0 = axs[row][col]
            if col == 0:
                im = ax0.contourf(
                    self.xx_ni * self.Lx,
                    self.zz_ni * self.Lz,
                    campos_ni[row].reshape(self.xx_ni.shape),
                    100,
                    cmap="jet",
                )
                if row == 0:
                    ax0.set_title(
                        rf"Propagación completa {self.xx_ni.shape[0]} $\times$ {self.xx_ni.shape[1]}"
                    )
            elif col == 1:
                im = ax0.contourf(
                    self.xx * self.Lx,
                    self.zz * self.Lz,
                    campos[row].reshape(self.xx.shape),
                    100,
                    cmap="jet",
                )
                if row == 0:
                    ax0.set_title(
                        rf"Propagación interpolada {self.xx.shape[0]} $\times$ {self.xx.shape[1]}"
                    )

            row += 1
            if row > len(campos) - 1:
                col += 1
                row = 0

        plt.suptitle(f"{title}", weight="semibold", y=0.97)
        plt.tight_layout()
        for ax0 in axs.reshape(-1):
            ax0.axis("scaled")

        fig.subplots_adjust(hspace=0.1, wspace=0.07)
        fig.colorbar(im, ax=axs.ravel(), shrink=0.8, format="{x:.2f}")

        if save:
            plt.savefig(
                f"{path_imgs}\\Comparacion_propagacion_{nombre}.png",
                bbox_inches="tight",
                dpi=330,
            )

        plt.show()

        if close:
            plt.close(fig)

    def cargar_coordenadas(self, path, scale=False):
        # coordenadas en las que se registra la salida del campo de onda en specfem.
        X0 = np.loadtxt(path)

        if scale:
            # specfem funciona con unidades de metros, por lo que debemos convertirlos a Km.
            X0 /= 1000

        X0[:, 0:1] = X0[:, 0:1] / self.Lx  # escalar el dominio espacial
        X0[:, 1:2] = X0[:, 1:2] / self.Lz  # escalar el dominio espacial
        xz = np.concatenate((X0[:, 0:1], X0[:, 1:2]), axis=1)

        return xz

    def cargar_campos(self, path_folder):
        wfs = sorted(os.listdir(path_folder))
        U0 = [np.loadtxt(f"{path_folder}/" + f) for f in wfs]
        return U0, wfs

    def grilla_no_absorbente(self):
        xxs, zzs = np.meshgrid(
            np.linspace(self.xf / self.Lx, self.xsf / self.Lx, self.n_ini),
            np.linspace(self.zf / self.Lz, self.az_spec / self.Lz, self.n_ini),
        )
        xxzzs = np.concatenate((xxs.reshape((-1, 1)), zzs.reshape((-1, 1))), axis=1)

        return xxs, zzs, xxzzs

    def grilla_sin_interpolar(self):
        xx_ni, zz_ni = np.meshgrid(
            np.linspace(0, self.ax_spec / self.Lx, self.nx),
            np.linspace(0, self.az_spec / Lz, self.nz),
        )
        xxzz_ni = np.concatenate(
            (xx_ni.reshape((-1, 1)), zz_ni.reshape((-1, 1))), axis=1
        )

        return xx_ni, zz_ni, xxzz_ni

    def grilla_entrenamiento_PINN(self):
        xx, zz = np.meshgrid(
            np.linspace(0, self.ax / self.Lx, self.n_ini),
            np.linspace(0, self.az / self.Lz, self.n_ini),
        )

        return xx, zz

    def plot_dominio(
        self,
        titulo="Dominios computacionales",
        save=False,
        nombre="Specfem",
        close=False,
    ):
        fig, axs = plt.subplots()
        axs.scatter(
            self.xxs * self.Lx,
            self.zzs * self.Lz,
            s=0.3,
            label="Área sin fronteras cpml",
        )
        axs.add_patch(
            Rectangle(
                (self.xf, self.zf),
                self.ax,
                self.az,
                fill=False,
                edgecolor="r",
                label="Entrenamiento PINN",
            )
        )
        axs.add_patch(
            Rectangle(
                (0, 0),
                self.ax_spec,
                self.az_spec,
                fill=False,
                edgecolor="b",
                label="Dominio total",
            )
        )

        axs.set_aspect("equal", adjustable="box")
        axs.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

        axs.set(
            xlim=[-0.001, self.ax_spec + 0.002], ylim=[-0.002, self.az_spec + 0.001]
        )

        plt.box(on=False)

        plt.legend(
            fontsize="small", ncols=3, bbox_to_anchor=(0.5, 0.0), loc="upper center"
        )
        plt.tight_layout()
        plt.title(f"{titulo}", weight="medium", y=1.15)

        if save:
            plt.savefig(
                f"{path_imgs}\\Esquema_{nombre}.png", bbox_inches="tight", dpi=330
            )
            plt.savefig(
                f"{path_imgs}\\Esquema_{nombre}.eps",
                bbox_inches="tight",
                dpi=330,
                format="eps",
            )
        plt.show()

        if close:
            plt.close(fig)

    def interpolar_campos(self, xz, U0, xxzz):
        U_ini1 = interpolate.griddata(xz, U0[0], xxzz, fill_value=0.0)
        U_ini1x = U_ini1[:, 0:1] / self.u_scl
        U_ini1z = U_ini1[:, 1:2] / self.u_scl
        campo1 = [U_ini1x, U_ini1z]

        U_ini2 = interpolate.griddata(xz, U0[1], xxzz, fill_value=0.0)
        U_ini2x = U_ini2[:, 0:1] / self.u_scl
        U_ini2z = U_ini2[:, 1:2] / self.u_scl
        campo2 = [U_ini2x, U_ini2z]

        U_spec = interpolate.griddata(xz, U0[2], xxzz, fill_value=0.0)
        U_specx = U_spec[:, 0:1] / self.u_scl
        U_specz = U_spec[:, 1:2] / self.u_scl
        campo3 = [U_specx, U_specz]

        return campo1, campo2, campo3

    def calculo_magnitud(self, campo_x, campo_z):
        magnitud = np.sqrt(campo_x**2 + campo_z**2)

        return magnitud

    def calculo_magnitudes(self, *campos: list):
        magnitudes = []
        for campo in campos:
            magnitud = self.calculo_magnitud(campo[0], campo[1])
            magnitudes.append(magnitud)

        return magnitudes

    def cargar_sismogramas(self, folder, componente):
        componente = componente.upper()
        sms = sorted(os.listdir(folder))
        smsz = [f for f in sms if f[-6] == componente]  # Z cmp seismos
        seismo_list = [np.loadtxt(folder + f) for f in smsz]  # Z cmp seismos

        return seismo_list

    def clip_tiempo(self, t_spec):
        cut_u = t_spec > self.t_s
        cut_l = t_spec < self.t_st
        return cut_u, cut_l

    def submuestreo_temporal(self, l_f, l_sl, l_su, t_spec):
        index = np.arange(l_sl, l_su, l_f)

        t_spec_sub = t_spec[index].reshape((-1, 1))

        # cambiar el eje del tiempo de nuevo a cero. la longitud de t_spec_sub debe ser igual a t_m-t_st
        t_spec_sub = t_spec_sub - t_spec_sub[0]

        return t_spec_sub, index

    def submuestreo_sismogramas(self, seismo_list, index):
        sismo_list_new = seismo_list.copy()

        for ii in range(len(seismo_list)):
            sismo_list_new[ii] = seismo_list[ii][index]

        S = sismo_list_new[0][:, 1].reshape(
            -1, 1
        )  # Componente en Z del primer sismometro
        for ii in range(len(sismo_list_new) - 1):
            S = np.concatenate((S, sismo_list_new[ii + 1][:, 1].reshape(-1, 1)), axis=0)

        return S

    def escalar_sismograma(self, S):
        return S / self.u_scl

    def plot_sismogramas(self, Sx, Sz, nombre, save=False, close=False):
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        ax[0].plot(Sx, lw=0.7)
        ax[0].set_title("Componente en X")
        ax[1].plot(Sz, lw=0.7)
        ax[1].set_title("Componente en Z")

        plt.tight_layout()
        if save:
            plt.savefig(
                f"{path_imgs}\\Sismogramas_PINN_{nombre}.eps",
                bbox_inches="tight",
                dpi=330,
                format="eps",
            )
            plt.savefig(
                f"{path_imgs}\\Sismogramas_PINN_{nombre}.png",
                bbox_inches="tight",
                dpi=330,
            )
        plt.show()

        if close:
            plt.close(fig)


# %% Caso propagación Specfem
# Parametros modelo
Lx = 3
Lz = 3

# Tamaño del dominio antes de eliminar las regiones absorbentes (completo)
ax_spec = 1.5
az_spec = 0.5

# Pasos espaciales
dx = 0.015
dz = 0.005

# Posicion sismometros
xsf = 1.3

# Nodos absorbentes
n_absx = 10
n_absz = 10

# Tiempos de los snaps
t01 = 0.1
t02 = 0.115

# Numero de puntos en cada dimensión para interpolar
n_ini = 40

# Tamaño del dominio del propagador
nx = 401
nz = 401
zl_s = 0.06 - n_absz * dz

u_scl = 1 / 3640

t_m = 0.5  # tiempo total para el entrenamiento de la PDE.
t_st = 0.1  # aquí es cuando se toma el primer I.C de specfem
t_s = 0.5  # serie de tiempo total utilizada de los sismogramas

# %%
specfem = GeneracionDatos(
    Lx,
    Lz,
    ax_spec,
    az_spec,
    dx,
    dz,
    xsf,
    n_absx,
    n_absz,
    t01,
    t02,
    n_ini,
    nx,
    nz,
    u_scl,
    t_m,
    t_st,
    t_s,
)

# %% Carga de datos
xz = specfem.cargar_coordenadas(
    "event1/wavefields/wavefield_grid_for_dumps_000.txt", scale=True
)
U0, wfs = specfem.cargar_campos(campos_spec)

# %%
specfem.plot_dominio(
    titulo="Dominios de Specfem", save=True, nombre="Specfem", close=True
)

# %% Interpolación de los campos

# Campos con la misma dimension
campo1_ni, campo2_ni, campo3_ni = specfem.interpolar_campos(xz, U0, specfem.xxzz_ni)
# Campos reducidos
campo1, campo2, campo3 = specfem.interpolar_campos(xz, U0, specfem.xxzzs)

magnitudes = specfem.calculo_magnitudes(campo1, campo2, campo3)
magnitudes_ni = specfem.calculo_magnitudes(campo1_ni, campo2_ni, campo3_ni)

# %%
specfem.plot_campos_conjunto(
    magnitudes[:-1],
    magnitudes_ni[:-1],
    "Propagación original Specfem",
    save=True,
    nombre="Specfem",
    close=True,
)

# %%
campos = [campo1, campo2, campo3]
specfem.plot_componentes(
    campos, magnitudes, save=True, nombre="original_PINN", close=True
)

# %% Sismogramas specfem
seismo_listz = specfem.cargar_sismogramas("event1/seismograms/", "Z")
seismo_listx = specfem.cargar_sismogramas("event1/seismograms/", "X")

# el tiempo de specfem no comienza desde cero para los sismos, por lo que se desplazan hacia adelante a cero
t_spec = -seismo_listz[0][0, 0] + seismo_listz[0][:, 0]
cut_u, cut_l = specfem.clip_tiempo(t_spec)

# este es el índice del eje de tiempo en specfem después del cual t>t_m
l_su = len(cut_u) - sum(cut_u)
l_sl = sum(cut_l)

# %% Submuestreos
t_spec_sub, index = specfem.submuestreo_temporal(100, l_sl, l_su, t_spec)

Sz = specfem.submuestreo_sismogramas(seismo_listz, index)
Sx = specfem.submuestreo_sismogramas(seismo_listx, index)

Sz = specfem.escalar_sismograma(Sz)
Sx = specfem.escalar_sismograma(Sx)

# %%
specfem.plot_sismogramas(Sx, Sz, "Specfem", save=True, close=True)

#%% Plots sustentacion - Experimento 4
# import matplotlib.colors as mcolors
# colors = list(mcolors.TABLEAU_COLORS.values())

# fig,axs = plt.subplots(1,10, figsize=(7,5))
# for i in range(10):
#     x = seismo_listx[i][:,1]
#     y = range(len(seismo_listx[i][:,1]))
              
#     axs[i].plot(x, y, c=colors[i])
#     axs[i].axis('off')
    
# plt.tight_layout()
# plt.savefig(r'C:\Users\juan9\OneDrive - UNIVERSIDAD INDUSTRIAL DE SANTANDER\Academico\Maestria Geofisica\Tesis\Propuesta de Investigacion\Sustentacion\Graficos\plots\sismogramas.eps', format='eps')
# plt.show()

# %% Caso propagación Propia
df = pd.read_csv(path_parametros)
# Parametros modelo
Lx = 3
Lz = 3

# Tamaño del dominio antes de eliminar las regiones absorbentes (completo)
ax_spec = df["Tamaño X"][0]/1000
az_spec = df["Tamaño Z"][0]/1000

# Tamaño del dominio del propagador
nz = df["Puntos Z"][0]
nx = df["Puntos X"][0]

# Pasos espaciales
dx = df["Tamaño celda x"][0]
dz = df["Tamaño celda z"][0]
dh = dx  # Tamaño de la celda

# Posicion sismometros
xsf = 1.3

# Nodos absorbentes
n_absx = df["Nodos abs"][0]
n_absz = df["Nodos abs"][0]

# Tiempos de los snaps
t01 = 0.1
t02 = 0.115

# Numero de puntos en cada dimensión para interpolar
n_ini = 50

u_scl = 1 / 1

t_m = 0.5  # tiempo total para el entrenamiento de la PDE.
t_st = 0.1  # aquí es cuando se toma el primer I.C de specfem
t_s = 0.5  # serie de tiempo total utilizada de los sismogramas

# %%
propagacion = GeneracionDatos(
    Lx,
    Lz,
    ax_spec,
    az_spec,
    dx,
    dz,
    xsf,
    n_absx,
    n_absz,
    t01,
    t02,
    n_ini,
    nx,
    nz,
    u_scl,
    t_m,
    t_st,
    t_s,
)

# %% Carga de datos
xz = propagacion.cargar_coordenadas(
    f"{campos_jd}/xz_componentes_campos.txt", scale=False
)
U0, wfs = propagacion.cargar_campos(campos_jd)

# %%
propagacion.plot_dominio(
    titulo="Dominios Propios", save=True, nombre="Propio", close=True
)

# %% Interpolación de los campos

# Campos con la misma dimension
campo1_ni, campo2_ni, campo3_ni = propagacion.interpolar_campos(
    xz, U0, propagacion.xxzz_ni
)
# Campos reducidos
campo1, campo2, campo3 = propagacion.interpolar_campos(xz, U0, propagacion.xxzzs)

magnitudes = propagacion.calculo_magnitudes(campo1, campo2, campo3)
magnitudes_ni = propagacion.calculo_magnitudes(campo1_ni, campo2_ni, campo3_ni)

# %%
propagacion.plot_campos_conjunto(
    magnitudes[:-1],
    magnitudes_ni[:-1],
    "Propagación Propia",
    save=True,
    nombre="propia",
    close=True
)

# %%
campos_list_jd = [campo1, campo2, campo3]
propagacion.plot_componentes(
    campos_list_jd, magnitudes, save=True, nombre="propia_interpolada", close=True
)

# %% Sismogramas propios
seismo_listz = propagacion.cargar_sismogramas(f"{sismos_jd}", "Z")
seismo_listx = propagacion.cargar_sismogramas(f"{sismos_jd}", "X")

t_spec = -seismo_listz[0][0, 0] + seismo_listz[0][:, 0]
cut_u, cut_l = propagacion.clip_tiempo(t_spec)

# este es el índice del eje de tiempo en specfem después del cual t>t_m
l_su = len(cut_u) - sum(cut_u)
l_sl = sum(cut_l)

# %% Submuestreos
t_spec_sub, index = propagacion.submuestreo_temporal(2, l_sl, l_su, t_spec)

Sz = propagacion.submuestreo_sismogramas(seismo_listz, index)
Sx = propagacion.submuestreo_sismogramas(seismo_listx, index)

Sz = propagacion.escalar_sismograma(Sz)
Sx = propagacion.escalar_sismograma(Sx)

# %%
propagacion.plot_sismogramas(Sx, Sz, "Propios", save=True, close=True)
