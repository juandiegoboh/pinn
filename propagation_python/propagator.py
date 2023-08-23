# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 20:46:16 2022

@author: Juan Diego Bohórquez

Implementación de la función de propagación de una onda acústica teniendo en
cuenta la adición de una capa de C-PML para eliminar los fenómenos de reflexión
en los bordes
"""

import numpy as np
from get_CPML import get_CPML
from numba import njit

#%% Fuentes sismicas
def gaussian_neg(tstep, dt, fq):
  t = dt * tstep - 1 / fq
  f = -np.exp(-2 * (np.pi ** 2) * (fq ** 2 ) * (t ** 2))
  return f

def gaussian(tstep, dt, fq):
  t = dt * tstep - 1 / fq
  f = np.exp(-2 * (np.pi ** 2) * (fq ** 2 ) * (t ** 2))
  return f

def ricker(tstep, dt, fq):
  t = dt * tstep - 1 / fq
  f = (1 - 2 * (np.pi ** 2) * (fq ** 2 ) * (t ** 2)) * np.exp(-(np.pi ** 2) * (fq ** 2) * (t ** 2))
  return f

fuentes = {
    "gaussian_neg": gaussian_neg,
    "gaussian": gaussian,
    "ricker": ricker,
    }

#%%
@njit
def calculate_propagator(m, src, Ix0, Iz0, dx, dz, dt, max_offset, fq, fronteras):
    '''
    Función que genera la propagación de una onda acústica utilizando la solución de diferencias finitas.

    Parametros: 
    ----------
    src : np.array
        Vector que representa la fuente de la propagación
    
    Para los demás parámetros ver definición en la función propagator().

    Returns
    -------
    P : np.array
        Campo de presión de la onda acústica en el tiempo. Tamaño = (Nx x Nz x Nt)
    '''
    max_ix = max_offset/dx
    # Se calculan Nx, Nz, Nt según los datos de entrada
    Nx = int(m.shape[0])
    Nz = int(m.shape[1])
    Nt = int(len(src))
    
    # Inicialización de arrays
    P = np.zeros((Nx, Nz, Nt), dtype=np.float64)    # Campo de Presión
    P_tmp = np.zeros((Nx, Nz), dtype=np.float64)
    P_Iz0 = np.zeros((Nt, Nx), dtype=np.float64)
    
    # Inicialización de las segundas derivadas
    d2P_dx2 = np.float64(0)
    d2P_dz2 = np.float64(0)
    
    d2P_dt2 = np.zeros((Nx, Nz, Nt), dtype=np.float64)
    
    dP_dx = np.zeros((Nx, Nz), dtype=np.float64)
    dP_dz = np.zeros((Nx, Nz), dtype=np.float64)
    
    F_dPdx = np.zeros((Nx, Nz), dtype=np.float64)
    F_dPdz = np.zeros((Nx, Nz), dtype=np.float64)
    F_d2Pdx2 = np.zeros((Nx, Nz), dtype=np.float64)
    F_d2Pdz2 = np.zeros((Nx, Nz), dtype=np.float64)
    
    # Velocidad de la onda
    v2 = m*m

    CPMLimit = 20
    
    # Precomputo de operaciones
    one_over_dx2 = 1/(dx**2)
    one_over_dz2 = 1/(dz**2)
    one_over_dx = 1/dx
    one_over_dz = 1/dz
    
    v_max = np.amax(m)
    v_min = np.amin(m)
    
    # Parametros de la onda
    wave_l = v_min/fq
    points_per_wavelength = wave_l/dx
    
    courant_number = v_max * dt/dx * np.sqrt(2)
    
    # Funciones para las condiciones C-PML
    R = 1e-4
    v_cpml = v_max
    
    # Cálculo de los elementos de la C-PML
    a_x,a_x_half,b_x,b_x_half,a_z,a_z_half,b_z,b_z_half = get_CPML(CPMLimit,R,v_cpml,Nx,Nz,dx,dz,dt,fq)
    
    left = fronteras[0]
    right = fronteras[1]
    top = fronteras[2]
    bottom = fronteras[3]
    
    # Definición de rango de la propagación sin fronteras
    rango_x_min = 0
    rango_x_max = Nx - 1
    rango_z_min = 0
    rango_z_max = Nz - 1
    
    # Definición del rango de la propagación con fronteras
    if left:
        rango_x_min = CPMLimit + 1
    if right:
        rango_x_max = Nx-CPMLimit
    if top:
        rango_z_min = CPMLimit + 1
    if bottom:
        rango_z_max = Nz-CPMLimit
    
    for it in range(1, Nt):
        # Snapshot
        P_tmp = np.copy(P[:,:,it])
        
        # Condiciones de frontera libre
        d2P_dt2[:,1,it] = 0       
        
        # Solución de diferencias finitas (Propagacion de la onda) Depende de las fronteras
        for ix in range(rango_x_min, rango_x_max):
            for iz in range(rango_z_min, rango_z_max):
                d2P_dx2 = (P_tmp[ix+1,iz]+P_tmp[ix-1,iz]-2*P_tmp[ix,iz]) * one_over_dx2
                d2P_dz2 = (P_tmp[ix,iz+1]+P_tmp[ix,iz-1]-2*P_tmp[ix,iz]) * one_over_dz2
                d2P_dt2[ix,iz,it] = d2P_dx2 + d2P_dz2
                
        # P_tmp[Nx-1,:] = 0
        # P_tmp[0,:] = 0
        # P_tmp[:,Nz-1] = 0

        if left:
            # Inicio de condiciones C-PML (lado izquierdo)
            for ix in range(1, CPMLimit+1):
                for iz in range(1, Nz-1):
                    dP_dx[ix,iz] = P_tmp[ix+1,iz] - P_tmp[ix,iz] 
                    dP_dx[ix,iz] = one_over_dx * dP_dx[ix,iz]
        
                    dP_dz[ix,iz] = P_tmp[ix,iz+1] - P_tmp[ix,iz] 
                    dP_dz[ix,iz] = one_over_dz * dP_dz[ix,iz]
                    
                    # Creación de los elementos de CPML 
                    F_dPdx[ix,iz] = F_dPdx[ix,iz] * b_x_half[ix] + a_x_half[ix] * dP_dx[ix,iz]
                    dP_dx[ix,iz] = dP_dx[ix,iz] + F_dPdx[ix,iz]
         
                    F_dPdz[ix,iz] = F_dPdz[ix,iz] * b_z[iz] + a_z[iz] * dP_dz[ix,iz]
                    dP_dz[ix,iz] = dP_dz[ix,iz] + F_dPdz[ix,iz]
                    
                    d2P_dx2 = one_over_dx * (dP_dx[ix,iz] - dP_dx[ix-1,iz])
                    d2P_dz2 = one_over_dz * (dP_dz[ix,iz] - dP_dz[ix,iz-1])
                
                    F_d2Pdx2[ix,iz] = F_d2Pdx2[ix,iz] * b_x[ix] + a_x[ix] * d2P_dx2
                    d2P_dx2 += F_d2Pdx2[ix,iz]
                    
                    F_d2Pdz2[ix,iz] = F_d2Pdz2[ix,iz] * b_z[iz] + a_z[iz] * d2P_dz2
                    d2P_dz2 += F_d2Pdz2[ix,iz]
                    
                    # Igualdad de la ecuación de onda
                    d2P_dt2[ix,iz,it] = d2P_dx2 + d2P_dz2
    
        if right:
            # Inicio de condiciones C-PML (lado derecho)
            for ix in range(Nx-2, Nx-CPMLimit-2, -1):
                for iz in range(1, Nz-1):
                    dP_dx[ix,iz] = P_tmp[ix,iz] - P_tmp[ix-1,iz]
                    dP_dx[ix,iz] = one_over_dx * dP_dx[ix,iz]
                    
                    dP_dz[ix,iz] = P_tmp[ix,iz+1] - P_tmp[ix,iz] 
                    dP_dz[ix,iz] = one_over_dz * dP_dz[ix,iz] 
                    
                    # Creación de los elementos de CPML 
                    F_dPdx[ix,iz] = F_dPdx[ix,iz] * b_x_half[ix-1] + a_x_half[ix-1] * dP_dx[ix,iz]
                    dP_dx[ix,iz] += F_dPdx[ix,iz]
                    
                    F_dPdz[ix,iz] = F_dPdz[ix,iz] * b_z[iz] + a_z[iz] * dP_dz[ix,iz]
                    dP_dz[ix,iz] += F_dPdz[ix,iz]
                    
                    d2P_dx2 = one_over_dx * (dP_dx[ix+1,iz] - dP_dx[ix, iz])
                    d2P_dz2 = one_over_dz * (dP_dz[ix,iz] - dP_dz[ix, iz-1])
                    
                    F_d2Pdx2[ix,iz] = F_d2Pdx2[ix,iz] * b_x[ix] + a_x[ix] * d2P_dx2
                    d2P_dx2 += F_d2Pdx2[ix,iz]
                    
                    F_d2Pdz2[ix,iz] = F_d2Pdz2[ix,iz] * b_z[iz] + a_z[iz] * d2P_dz2
                    d2P_dz2 += F_d2Pdz2[ix,iz]
                    
                    # Igualdad de la ecuación de onda
                    d2P_dt2[ix,iz,it] = d2P_dx2 + d2P_dz2      

        if bottom:
            # Inicio de condiciones C-PML (parte inferior)
            for ix in range(CPMLimit+1, Nx-CPMLimit-1):
                for iz in range(Nz-2, Nz-CPMLimit-2, -1):
                    dP_dx[ix,iz] = P_tmp[ix+1,iz] - P_tmp[ix,iz]
                    dP_dx[ix,iz] = one_over_dx * dP_dx[ix,iz]
                    
                    dP_dz[ix,iz] = P_tmp[ix,iz] - P_tmp[ix,iz-1] 
                    dP_dz[ix,iz] = one_over_dz * dP_dz[ix,iz] 
                    
                    # Creación de los elementos de CPML 
                    F_dPdx[ix,iz] = F_dPdx[ix,iz] * b_x[ix] + a_x[ix] * dP_dx[ix,iz]
                    dP_dx[ix,iz] += F_dPdx[ix,iz]
                    
                    F_dPdz[ix,iz] = F_dPdz[ix,iz] * b_z_half[iz-1] + a_z_half[iz-1] * dP_dz[ix,iz]
                    dP_dz[ix,iz] += F_dPdz[ix,iz]
                    
                    d2P_dx2 = one_over_dx * (dP_dx[ix,iz] - dP_dx[ix-1, iz])
                    d2P_dz2 = one_over_dz * (dP_dz[ix,iz+1] - dP_dz[ix, iz])
                    
                    F_d2Pdx2[ix,iz] = F_d2Pdx2[ix,iz] * b_x[ix] + a_x[ix] * d2P_dx2
                    d2P_dx2 += F_d2Pdx2[ix,iz]
                    
                    F_d2Pdz2[ix,iz] = F_d2Pdz2[ix,iz] * b_z[iz] + a_z[iz] * d2P_dz2
                    d2P_dz2 += F_d2Pdz2[ix,iz]
                    
                    # Igualdad de la ecuación de onda
                    d2P_dt2[ix,iz,it] = d2P_dx2 + d2P_dz2    
                    
        if top:      
            # Inicio de condiciones C-PML (parte superior)
            for ix in range(CPMLimit+1, Nx-CPMLimit-1):
                for iz in range(1, CPMLimit+1):
                    dP_dx[ix,iz] = P_tmp[ix+1,iz] - P_tmp[ix,iz]
                    dP_dx[ix,iz] = one_over_dx * dP_dx[ix,iz]
                    
                    dP_dz[ix,iz] = P_tmp[ix,iz+1] - P_tmp[ix,iz] 
                    dP_dz[ix,iz] = one_over_dz * dP_dz[ix,iz] 
                    
                    # Creación de los elementos de CPML 
                    F_dPdx[ix,iz] = F_dPdx[ix,iz] * b_x[ix] + a_x[ix] * dP_dx[ix,iz]
                    dP_dx[ix,iz] += F_dPdx[ix,iz]
                    
                    F_dPdz[ix,iz] = F_dPdz[ix,iz] * b_z_half[iz] + a_z_half[iz] * dP_dz[ix,iz]
                    dP_dz[ix,iz] += F_dPdz[ix,iz]
                    
                    d2P_dx2 = one_over_dx * (dP_dx[ix,iz] - dP_dx[ix-1, iz])
                    d2P_dz2 = one_over_dz * (dP_dz[ix,iz] - dP_dz[ix, iz-1])
                    
                    F_d2Pdx2[ix,iz] = F_d2Pdx2[ix,iz] * b_x[ix] + a_x[ix] * d2P_dx2
                    d2P_dx2 += F_d2Pdx2[ix,iz]
                    
                    F_d2Pdz2[ix,iz] = F_d2Pdz2[ix,iz] * b_z[iz] + a_z[iz] * d2P_dz2
                    d2P_dz2 += F_d2Pdz2[ix,iz]
                    
                    # Igualdad de la ecuación de onda
                    d2P_dt2[ix,iz,it] = d2P_dx2 + d2P_dz2
        
        # ==== Fin de la cpml ====            
        
        # Integración de tiempo
        P[:,:,it+1] = (dt**2) * v2 * d2P_dt2[:,:,it] + 2*P[:,:,it] - P[:,:,it-1]
        
        # Inyección de la fuente
        P[Ix0,Iz0,it+1] = P[Ix0,Iz0,it+1] + src[it+1]
        
        # Selección de los componentes de la superficie
        P_Iz0[it,:] = P[:,Iz0-1,it]
    
    # Se retornan las variables importantes para el metodo principal
    return P, Ix0, Iz0, max_ix, Nx, Nz

#%%
def propagator(m:np.array, Ix0:int, Iz0:int, dx:float, dz:float, dt:float, nt:int, max_offset:float, fq:int, tipo_fuente:str, fronteras:list=[]):
    '''
    Método que llama a la función calculate_propagator y entrega el resultado del campo P al usuario
    Además define las entradas de las fronteras CPML y las convierte en un np.array
    
    Parameters
    ----------
    m : np.array
        Matriz de velocidades del modelo, dimensiones nx,nz.
    Ix0 : int
        Posición en X de la perturbación, en puntos del modelo (Nx).
    Iz0 : int
        Posición en Z de la perturbación, en puntos del modelo (Nx).
    dx : float
        Tamaño de la celda en x.
    dz : float
        Tamaño de la celda en z.
    dt : float
        Tamaño de la rebanada temporal.
    nt : int
        Cantidad de pasos en el eje temporal.
    max_offset : float
    
    fq : int
        Frecuencia de la onda fuente.
    tipo_fuente : str
        Tipo de onda a utilizar. Las fuentes disponibles son "ricker", "gaussian" y "gaussian_neg".
    fronteras : list
        Lista que indica que fronteras absorbentes activar. Ej: ["top", "bottom"]

    Returns
    -------
    P : np.array
        Campo de presión calculado por la función calculate_propagator().
    '''
    
    # Fronteras  
    left = True if "left" in fronteras else False
    right = True if "right" in fronteras else False
    top = True if "top" in fronteras else False
    bottom = True if "bottom" in fronteras else False
    
    fronteras = np.array([left, right, top, bottom], dtype="bool")
    
    # Fuente sismica
    src = np.ones(nt)

    for i in range(nt):
        src[i] = fuentes[tipo_fuente](i, dt, fq)
        
    # Calculo de la propagación
    P, Ix0, Iz0, max_ix, Nx, Nz = calculate_propagator(m, src, Ix0, Iz0, dx, dz, dt, max_offset, fq, fronteras)
    
    return P

