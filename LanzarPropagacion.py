# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:10:29 2023

@author: Juan Diego Bohórquez
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from vedo import Volume, Text2D
from vedo.applications import Slicer3DPlotter
from numba import njit
from pytictoc import TicToc

from multireplace import multireplace
from datetime import datetime
from scipy.ndimage import zoom
from scipy.ndimage import laplace
from prettytable import PrettyTable

from CreacionModeloVelocidad import velocidad_eliptica
from _global import modelo_vel_name, numero_experimento, path_folder

# %%
now = datetime.now()
fecha = now.strftime("%d-%m-%Y")
hora = now.strftime("%I:%M:%S %p")

# %% Declaración de ruta principal y creacion de carpeta de experimento

path_experimento = os.path.join(
    path_folder, f"experimentos\{numero_experimento}")

# Carpeta del experimento
if not os.path.exists(path_experimento):
    subfolders = ["seismograms", "velocity_models", "wavefields", "images"]

    for item in subfolders:
        path = os.path.join(path_experimento, item)
        os.makedirs(path)

# %% Definición de paths
path_propagador = os.path.join(path_folder, "propagation-cpml-C")

velocity_path_bin = os.path.join(
    path_experimento, f"velocity_models\{modelo_vel_name}.bin"
)

# Path relativo para usar en el codigo de C
velocity_path_c = velocity_path_bin.split("\\")[-4:]
velocity_path_c = "/".join(velocity_path_c)
velocity_path_c = f"../{velocity_path_c}"

path_modelo_original = "event1/modelo_vel.npy"

path_wavefield_jd = os.path.join(path_experimento, "wavefields")
path_seismograms_jd = os.path.join(path_experimento, "seismograms")
path_imgs = os.path.join(path_experimento, "images")

# %% Carpeta de imagenes
if not os.path.exists(path_imgs):
    # Si no existe, se crea
    os.makedirs(path_imgs)

# %% Definición de fuciones de plot


def plot_campo(P, xx, zz, dx, dz, title, snap, save, field_name):
    fig = plt.figure()
    plt.contourf(xx * dx, zz * dz, P.reshape((xxs.shape)), 100, cmap="jet")
    plt.ylabel("z")
    plt.xlabel("x")
    plt.colorbar()
    plt.axis("scaled")
    plt.title(title)
    plt.show()

    if save:
        plt.savefig(
            f"{path_imgs}\\{field_name}_{snap}.png", bbox_inches="tight", dpi=320
        )
    plt.close(fig)


def escalar_intervalo(matriz: np.ndarray, min_intervalo: float, max_intervalo: float):
    min_valor = np.min(matriz)
    max_valor = np.max(matriz)
    rango = max_valor - min_valor

    # Escalado manual
    matriz_escalada = ((matriz - min_valor) / rango) * \
        (max_intervalo - min_intervalo) + min_intervalo

    return matriz_escalada


# %% Cambio de parámetros de la propagación

# Se definen los parametros de la propagación
fq = 16   # Frecuencia (Hz)
Nz = 150  # Puntos del modelo en z
Nx = Nz * 3  # Puntos del modelo en x (en specfem es 100)
Nt = 650  # Puntos del modelo en t
tipo_fuente = "gaussian_neg" # gaussian, ricker, gaussian-neg

n_abs = 20  # nodos para absorber condiciones de frontera en ambas direcciones de specfem
n_absx = n_abs  # nodos del lado izquierdo del dominio
n_absz = n_abs  # el límite superior no absorbe

n_event = 1  # numero de eventos sísmicos
n_seis = 20  # número de sismómetros de entrada de SPECFEM

# %% Otros parametros de propagación

ax_spec = 1.5  # tamaño del dominio en specfem antes de eliminar las regiones absorbentes
az_spec = 0.5
xsf = 1.3  # x ubicación de todos los sismómetros en specfem

# Se define la velocidad verdadera del subsuelo (Modelo PINN)
alpha_true = np.load(path_modelo_original).astype("float32") * 1000

dh = ax_spec / Nx  # Tamaño de la celda
dx = dz = dh
dt = dh * 1000 / (np.amax(alpha_true) * np.sqrt(2))  # Intervalo temporal
s_spec = 5e-5  # Paso de tiempo de specfem (0.05 ms)

# Coordenadas de la fuente (en indices de la matriz)
Sx = int(Nx / 3)
Sz = int(Nz / 2)

# Coordenadas en metros de la fuente
sx_m, sz_m = Sx*dx*1000, Sz*dz*1000

# Dimensión del dominio para entrenamiento de PINNs.
ax = xsf - n_absx * dx
# solo se elimina el grosor del la frontera absorbente de la izquierda ya que #xsf es (debe ser) más pequeño que donde comienza la frontera absorbente del lado derecho.
az = az_spec - n_absz * dz  # dimensión del dominio en la dirección z (se elimina solo la frontera de abajo)

# ------------------- Sismometros Marco PINN -------------------#
# z ubicación del primer sismómetro de SPECFEM en el marco de referencia de PINN. Aquí debe estar en km mientras que en SPECFEM está en metros.
z0_s = 0.45
zl_s = 0.06 - 10 * dz # z ubicación del último sismómetro en profundidad. Esto no tiene que ser cero y puede ser mayor, especialmente si tiene una frontera absorbente en la parte inferior

# Coordenadas receptores
xsf_arr = np.array([xsf] * n_seis)
zsf = np.linspace(z0_s, zl_s, n_seis)
# --------------------------------------------------------------#

t01 = 2000 * s_spec  # disposición inicial en este momento de specfem. Primer snap
t02 = 2300 * s_spec  # segunda disposición "inicial". Segundo snap
t_la = 5000 * s_spec  # datos de prueba para comparar specfem y PINN entrenados

# Dimensiones totales (incluidas fronteras absorbentes)
x_total = Nx * dx
z_total = Nz * dz
t_total = Nt * dt

# %% Este bloque solo funciona para cargar el modelo de velocidad de Rash

# Re escalado de alpha true al tamaño del modelo de entrenamiento
x_zoom = (ax/dx) / alpha_true.shape[0]
z_zoom = (az/dz) / alpha_true.shape[1]

zoom_factor = [z_zoom, x_zoom]
alpha_true0 = zoom(alpha_true, zoom_factor, order=1)

# Se completa el modelo de velocidad para el dominio completo
alpha_true1 = np.pad(alpha_true0, ((0, int(Nz-az/dz)), (0,int(Nx-ax/dx))), mode='edge')

# Se exporta el archivo binario para leer en C
output_file = open(velocity_path_bin, "wb")
alpha_true1.T.tofile(output_file)
output_file.close()

# %% Plot modelo para entrenamiento - Original SPECFEM
n_ini = 50

Lx = Lz = 3
xx, zz = np.meshgrid(np.linspace(0, 1.25, n_ini), 
                       np.linspace(0, 0.45, n_ini))

fig = plt.figure()
plt.contourf(xx, zz, alpha_true.reshape((xx.shape)), 100, cmap="jet")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("z")
plt.title(
    fr"Modelo de velocidad ($\alpha$) - Original {alpha_true.shape[0]}x{alpha_true.shape[1]}")
plt.scatter(Sx * dx, Sz * dz, c="k", label="Fuente")
plt.plot(xsf_arr*.95, zsf, "r*", markersize=4, label="Receptores")
plt.axis("scaled")

plt.legend(loc='upper left', fontsize='small')
plt.show()

plt.savefig(f"{path_imgs}\\alpha_true_PINN.png", bbox_inches="tight", dpi=320)
plt.close(fig)

# %% Plot modelo interpolado (usado en la propagación)

xxs, zzs = np.meshgrid(np.linspace(0, Nx, Nx), np.linspace(0, Nz, Nz))

# ------------------- Sismometros Marco propio -------------------#
# Se debe hacer una ubicación absoluta de los sismogramas

z0_s = az_spec - 0.003      # z ubicación del 1er sismómetro, 3m debajo de la superficie.
zl_s = 0.01 + n_absz * dz   # z ubicación del último sismómetro, 10m antes de la cpml.

# Coordenadas receptores
xsf_arr = np.array([xsf] * n_seis)
zsf = np.linspace(z0_s, zl_s, n_seis)
# ---------------------------------------------------------------#

fig = plt.figure()
plt.contourf(xxs * dx, zzs * dz,
             alpha_true1.reshape((xxs.shape)), 100, cmap="jet")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("z")
plt.title(r"Modelo de velocidad ($\alpha$) - Interpolado " + f"{Nz}x{Nx}")
plt.scatter(Sx * dx, Sz * dz, c="k", label="Fuente")
plt.plot(xsf_arr, zsf, "r*", markersize=4, label="Receptores")
plt.axis("scaled")

plt.legend(loc='best', fontsize='small')
plt.show()

plt.savefig(f"{path_imgs}\\alpha_true0_original.png", bbox_inches="tight", dpi=320)
plt.close(fig)

# %% Modificación del script en C


def reemplazar_variables(template, final, replacements):
    with open(template, "rt") as file:
        data = file.read()

        data = multireplace(data, replacements)

    with open(final, "wt") as file:
        file.write(data)


t = TicToc()


# %% Comando de linux
def lanzar_propagacion():
    template_path = os.path.join(path_propagador, "principal_template.c")
    final_path = os.path.join(path_propagador, "principal2.c")
    replacements = {
        "[[Nx]]": f"{Nx}",
        "[[Nz]]": f"{Nz}",
        "[[fq]]": f"{fq}",
        "[[Sx]]": f"{Sx}",
        "[[Sz]]": f"{Sz}",
        "[[dh]]": f"{dh*1000}",
        "[[Tout]]": f"{Nt}",
        "[[n_abs]]": f"{float(n_abs)}",
        "[[velocity_path]]": f'"{velocity_path_c}"',
    }
    
    reemplazar_variables(template_path, final_path, replacements)
    
    template_path = os.path.join(path_propagador, "funciones2_template.h")
    final_path = os.path.join(path_propagador, "funciones2.h")
    replacements = {"[[fuente]]": f"{tipo_fuente}"}
    
    reemplazar_variables(template_path, final_path, replacements)

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
    print(f"Tiempo de propagación {tiempo} segundos")
    
    return  tiempo


tiempo_propagacion = lanzar_propagacion()

# %% Leer el binario
path_cubos = os.path.join(path_folder, "cubos")

filename = os.path.join(path_propagador, "propagacion2.bin")


@njit
def crear_cubo(prueba: np.array):
    # El cubo queda con dimensiones Nz * Nx * Nt
    cubo = np.zeros((Nz, Nx, Nt))

    for k in range(Nt):
        for j in range(Nx):
            for i in range(Nz):
                cubo[i, j, k] = prueba[i, j + (k - 1) * Nx]
    return cubo


t.tic()

with open(filename, "rb") as fid:
    data = np.fromfile(fid, dtype=np.float32)

    num_rows = int(data.shape[0] / Nz)
    data.resize(num_rows, Nz)
    data = data.T

    cubo = crear_cubo(data)
t.toc("Preparación del cubo")

# %% Snapshots originales
sn1 = int(t01 / dt)
sn2 = int(t02 / dt)
sn3 = int(t_la / dt)

num_snaps = [sn1, sn2, sn3]

for i in range(len(num_snaps)):
    snap = num_snaps[i]
    P = cubo[:, :, snap]

    plot_campo(
        P,
        xxs,
        zzs,
        dx,
        dz,
        title=f"Snap {snap+1}/{Nt}. T = {round(snap*dt, 3)}",
        snap=i + 1,
        save=True,
        field_name="snap",
    )


# %% Componentes del campo

for i in range(len(num_snaps)):
    snap = num_snaps[i]
    P = cubo[:, :, snap]

    # Calculo de la variación del campo en ambas direcciones x e z. Campo de desplazamiento
    Uz, Ux = np.gradient(P)

    # Normalización al intervalo [-1, 1]
    Ux_scaled = escalar_intervalo(Ux, -1, 1)
    Uz_scaled = escalar_intervalo(Uz, -1, 1)

    # Magnitud del campo de desplazamiento
    U0_mag = np.sqrt(Ux**2 + Uz**2)
    U0_mag_scaled = np.sqrt(Ux_scaled**2 + Uz_scaled**2)

    U_comp = {
        "Componente x": Ux_scaled,
        "Componente z": Uz_scaled,
        "Magnitud $\phi$": U0_mag_scaled
    }

    fig, axs = plt.subplots(3, 1, figsize=(5, 7))
    j = 0

    for key, value in U_comp.items():
        a = axs[j].contourf(
            xxs * dh, zzs * dh, value.reshape((xxs.shape)), 100, cmap="jet"
        )
        axs[j].axis("scaled")
        axs[j].set(title=f"{key}")
        j += 1

        # Colorbar formato
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

    fig.savefig(f"{path_imgs}\\componente_{i+1}_propio_completo.png",
                bbox_inches="tight", dpi=320)
    plt.close(fig)

    U0x = Ux_scaled.reshape(-1, 1)
    U0z = Uz_scaled.reshape(-1, 1)
    U0 = np.concatenate((U0x, U0z), axis=1)

    np.savetxt(f"{path_wavefield_jd}/componentes_campo_{i+1}.txt", U0)

# %% Coordenadas del campo

xxss, zzss = np.meshgrid(np.linspace(0, ax_spec, Nx),
                         np.linspace(0, az_spec, Nz))

coord_campo_x = xxss.reshape(-1, 1)
coord_campo_z = zzss.reshape(-1, 1)
coords_campo = np.concatenate((coord_campo_x, coord_campo_z), axis=1)

np.savetxt(f"{path_wavefield_jd}/xz_componentes_campos.txt", coords_campo)

# %% Modelo 3D
vol = Volume(cubo)

plot3D = Slicer3DPlotter(
    vol,
    bg="white",
    bg2="lightblue",
    cmaps=("gist_ncar_r", "jet", "Spectral_r", "hot_r", "bone_r"),
    use_slider3d=False,
)

plot3D.close()

# %% Sismogramas de valor del campo

# Posiciones de los sismogramas con respecto al cubo (indices)
xsf_arr_id = [int(x * (Nx - 1) / ax_spec) for x in xsf_arr]
zsf_arr_id = [int(z * (Nz - 1) / az_spec) for z in zsf]

# Sismogramas para las componentes del desplazamiento Ux, Uz
t.tic()
grad_z, grad_x = np.gradient(cubo, axis=(0, 1))
t.toc("Calculo gradiente del cubo completo")

sismogramas_x = np.zeros((0,))
sismogramas_z = np.zeros((0,))

tt = np.linspace(0, Nt * dt, Nt)

# Concatenación de sismogramas
for i in range(len(zsf)):

    u_cubo_x = grad_x[zsf_arr_id[i], xsf_arr_id[i], :]
    u_cubo_z = grad_z[zsf_arr_id[i], xsf_arr_id[i], :]

    # Para el plot
    sismogramas_x = np.concatenate((sismogramas_x, u_cubo_x), axis=0)
    sismogramas_z = np.concatenate((sismogramas_z, u_cubo_z), axis=0)

# Escalado
sismogramas_x = escalar_intervalo(sismogramas_x, -1, 1)
sismogramas_z = escalar_intervalo(sismogramas_z, -1, 1)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(sismogramas_x, lw=0.8, label='sismogramas x')
ax[0].set_title("Sismogramas X")
ax[1].plot(sismogramas_z, lw=0.8, label='sismogramas z', c='orange')
ax[1].set_title("Sismogramas Z")

plt.show()
plt.close(fig)

# %%

def guardar_sismograma(sismograma, ruta_temporal, ruta_final):
    if os.path.exists(ruta_final):
        os.remove(ruta_final)
    np.savetxt(ruta_temporal, sismograma)
    os.rename(ruta_temporal, ruta_final)

# %%
# Dividir los sismogramas para guardarlos
sub_sismogramas_x = np.split(sismogramas_x, n_seis)
sub_sismogramas_z = np.split(sismogramas_z, n_seis)

for i in range(n_seis):

    sismograma_x = np.concatenate((tt.reshape(-1, 1),
                                   sub_sismogramas_x[i].reshape(-1, 1)), axis=1)

    sismograma_z = np.concatenate((tt.reshape(-1, 1),
                                   sub_sismogramas_z[i].reshape(-1, 1)), axis=1)

    # Guardar el sismograma individualmente
    nombre_x_temp = f"{path_seismograms_jd}/JDX.semd"
    nombre_z_temp = f"{path_seismograms_jd}/JDZ.semd"

    nombre_x_fin = f"{path_seismograms_jd}/JD.S{i+1:04d}.BXX.semd"
    nombre_z_fin = f"{path_seismograms_jd}/JD.S{i+1:04d}.BXZ.semd"

    guardar_sismograma(sismograma_x, nombre_x_temp, nombre_x_fin)
    guardar_sismograma(sismograma_z, nombre_z_temp, nombre_z_fin)


# %%

tabla = PrettyTable()

parametros_basicos = {
    "Frecuencia" : fq,
    "Tipo fuente" : tipo_fuente,
    "Puntos Z" : Nz,
    "Puntos X" : Nx,
    "Puntos t" : Nt,
    "Tamaño X" : x_total*1000,
    "Tamaño Z" : z_total*1000,
    "Duración" : t_total,
    "Nodos abs" : n_abs,
    "Numero fuentes" : n_event,
    "Numero sismometros": n_seis,
    "Tamaño celda x": dx,
    "Tamaño celda z": dz,
    "Paso temporal": dt,
    "Coord x fuente": sx_m,
    "Coord z fuente": sz_m,
    "t01": t01,
    "t02": t02,
    "t03": t_la,
    }

parametros_log = {
    "Frecuencia de la fuente" : f'{fq} Hz',
    "Tipo de fuente" : f'{tipo_fuente}',
    "Puntos en Z" : f'{Nz}',
    "Puntos en X" : f'{Nx}',
    "Puntos en t" : f'{Nt}',
    "Tamaño en X" : f'{x_total*1000} m',
    "Tamaño en Z" : f'{z_total*1000} m',
    "Duración total" : f'{t_total:.4f} seg',
    "Nodos absorbentes" : f'{n_abs}',
    "Número de fuentes" : f'{n_event}',
    "Número de sismómetros": f'{n_seis}',
    "Tamaño de la celda": f'{dh:.4f} m',
    "Coordenada x fuente": f'{sx_m} m',
    "Coordenada z fuente": f'{sz_m} m',
    }

# Definición de la tabla de parámetros

tabla.add_column("Parámetro", list(parametros_log.keys()))
tabla.add_column("Valor", list(parametros_log.values()))
tabla.align["Parámetros"] = "l"

# Ruta y nombre del archivo
archivo = f"{path_experimento}/datos_propagacion.txt"
contenido = f"Parámetros de la propagación, experimento {numero_experimento.split('_')[-1]} {fecha} {hora}\n" 

contenido += f"\n{tabla.get_string()}\n"
contenido += f"\nTiempo de propagación {tiempo_propagacion:.4f} segundos"
    
# Guardar el contenido en el archivo
with open(archivo, "w") as file:
    file.write(contenido)
    
#%%% Guardado de csv
df = pd.DataFrame(parametros_basicos, index=[0])

# Ruta del archivo CSV
archivo_csv = f"{path_experimento}/datos_propagacion.csv"

df.to_csv(archivo_csv, index=False)