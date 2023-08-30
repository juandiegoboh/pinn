# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 00:10:46 2022

@author: Juan Diego Bohórquez
"""

import numpy as np
from numba import njit

@njit
def get_CPML(CPMLimit, R, Vcpml, Nx, Nz, dx, dz, dt, frec):
    '''
    Función que retorna las matrices que corresponden al término de convolución
    (variable auxiliar psi) de una CPML (Convolutional Perfect Matching Layer),
    dados unos límites establecidos, para la solución de condiciones de 
    fronteras en un modelamiento acústico. 
    
    Basado en (Komatitsch & Tromp, 2003), (Komatitsch & Martin, 2007) y 
    (Pasalic & McGarry, 2010).

    Parametros
    ----------
    CPMLimit : int
        Tamaño en puntos de la zona CPML.
    R : float
        Coeficiente de reflexión teórico después de la discretización. Recomendado
        1e-3 (Komatitsch & Tromp, 2003)
    Vcpml : float
        Velocidad en el medio.
    Nx : int
        Numero de puntos en x.
    Nz : int
        Numero de puntos en x.
    dx : float
        Distancia entre los puntos en la dirección x.
    dz : float
        Distancia entre los puntos en la dirección z.
    dt : float
        Intervalo temporal entre cada franja del modelo.
    frec : float
        Frecuencia de la fuente.

    Retorna
    -------
    a_x : array
        Primer término de la convolución en el eje x.
    a_x_half : array
        Primer término de la convolución en los puntos medios del eje x.
    b_x : array
        Segundo término de la convolución en el eje x.
    b_x_half : array
        Segundo término de la convolución en los puntos medios del eje x.
    a_z : array
        Primer término de la convolución en el eje z.
    a_z_half : TYPE
        Primer término de la convolución en los puntos medios del eje z.
    b_z : TYPE
        Segundo término de la convolución en el eje z.
    b_z_half : TYPE
        Segundo término de la convolución en los puntos medios del eje z.

    '''
    # Distancia de la zona CPML
    D_pml_x = CPMLimit*dx
    D_pml_z = CPMLimit*dz

    # Damping profile empírico inicial (Komatitsch & Tromp, 2003)
    d0_x = -(3/(2*D_pml_x))*np.log(R)
    d0_z = -(3/(2*D_pml_z))*np.log(R)
    
    x = np.zeros((CPMLimit+1))
    z = np.zeros((CPMLimit+1))

    # Velocidades
    alpha_x = np.zeros((CPMLimit+1))
    alpha_z = np.zeros((CPMLimit+1))
    
    x_half = np.zeros((CPMLimit+1))
    z_half = np.zeros((CPMLimit+1))
    
    alpha_x_half = np.zeros((CPMLimit+1))
    alpha_z_half = np.zeros((CPMLimit+1))
    
    for i in range(CPMLimit+1):
        # Coordenadas x y z
        x[i] = (CPMLimit - i)*dx  
        z[i] = (CPMLimit - i)*dz
        # Velocidades de onda P en el CPML, disminuyen al aumentar x y z
        alpha_x[i] = np.pi*frec*(D_pml_x - x[i])/D_pml_x    
        alpha_z[i] = np.pi*frec*(D_pml_z - z[i])/D_pml_z
        # Coordenadas x y z en el punto medio de dx y dz
        x_half[i] = (CPMLimit - i)*dx - dx/2
        z_half[i] = (CPMLimit - i)*dz - dz/2
        # Velocidades de onda P en el CPML en los puntos medios
        alpha_x_half[i] = np.pi*frec*(D_pml_x - x_half[i])/D_pml_x
        alpha_z_half[i] = np.pi*frec*(D_pml_z - z_half[i])/D_pml_z
    
    # Arrays para la dirección Z
    d_z = np.zeros((Nz))
    b_z = np.zeros((Nz))
    a_z = np.zeros((Nz))
    d_z_half = np.zeros((Nz-1))
    b_z_half = np.zeros((Nz-1))
    a_z_half = np.zeros((Nz-1))
    
    # Parte inferior, solo toma la zona CPML          
    for i in range(Nz-CPMLimit-1, Nz):
        d_z[i] = d0_z*Vcpml*((z[Nz-i-1]/D_pml_z)**2)    # Damping en función de x
        b_z[i] = np.exp(-(d_z[i] + alpha_z[Nz-i-1]) * dt)
        a_z[i] = d_z[i]/(d_z[i] + alpha_z[Nz-i-1]) * (b_z[i]-1)
        
        # Condición de la frontera de la zona CPML
        if i + 1 == Nz-CPMLimit:
            d_z_half[i-1] = 0
            b_z_half[i-1] = 0
            a_z_half[i-1] = 0
        else:
            d_z_half[i-1] = d0_z*Vcpml*((z_half[Nz-i-1]/D_pml_z)**2)
            b_z_half[i-1] = np.exp(-(d_z_half[i-1] + alpha_z_half[Nz-i-1]) * dt)
            a_z_half[i-1] = d_z_half[i-1]/(d_z_half[i-1] + alpha_z_half[Nz-i-1]) * (b_z_half[i-1]-1)
            
    # Parte superior, solo toma la zona CPML
    for i in range(CPMLimit+1):
        d_z[i] = d0_z*Vcpml*((z[i]/D_pml_z)**2)    # Damping en función de x
        b_z[i] = np.exp(-(d_z[i] + alpha_z[i]) * dt)
        a_z[i] = d_z[i]/(d_z[i] + alpha_z[i]) * (b_z[i]-1)
        
        # Condición de la frontera de la zona CPML
        if i == CPMLimit:
            d_z_half[i] = 0
            b_z_half[i] = 0
            a_z_half[i] = 0
        else:
            d_z_half[i] = d0_z*Vcpml*((z_half[i]/D_pml_z)**2)
            b_z_half[i] = np.exp(-(d_z_half[i] + alpha_z_half[i]) * dt)
            a_z_half[i] = d_z_half[i]/(d_z_half[i] + alpha_z_half[i]) * (b_z_half[i]-1)

    # Arrays para la dirección X
    d_x = np.zeros((Nx))
    b_x = np.zeros((Nx))
    a_x = np.zeros((Nx))
    d_x_half = np.zeros((Nx-1))
    b_x_half = np.zeros((Nx-1))
    a_x_half = np.zeros((Nx-1))
    
    # Parte izquierda, solo toma la zona CPML
    for i in range(CPMLimit+1):
        d_x[i] = d0_x*Vcpml*((x[i]/D_pml_x)**2)     # Damping en función de x
        b_x[i] = np.exp(-(d_x[i] + alpha_x[i]) * dt)
        a_x[i] = d_x[i]/(d_x[i] + alpha_x[i]) * (b_x[i]-1)
        
        # Condición de la frontera de la zona CPML
        if i == CPMLimit:
            d_x_half[i] = 0
            b_x_half[i] = 0
            a_x_half[i] = 0
        else:
            d_x_half[i] = d0_x*Vcpml*((x_half[i]/D_pml_x)**2)
            b_x_half[i] = np.exp(-(d_x_half[i] + alpha_x_half[i]) * dt)
            a_x_half[i] = d_x_half[i]/(d_x_half[i] + alpha_x_half[i]) * (b_x_half[i]-1)

    # Parte derecha, solo toma la zona CPML
    for i in range(Nx-CPMLimit-1, Nx):
        d_x[i] = d0_x*Vcpml*((x[Nx-i-1]/D_pml_x)**2)    # Damping en función de x
        b_x[i] = np.exp(-(d_x[i] + alpha_x[Nx-i-1]) * dt)
        a_x[i] = d_x[i]/(d_x[i] + alpha_x[Nx-i-1]) * (b_x[i]-1)
        
        # Condición de la frontera de la zona CPML
        if i + 1 == Nx-CPMLimit:
            d_x_half[i-1] = 0
            b_x_half[i-1] = 0
            a_x_half[i-1] = 0
        else:
            d_x_half[i-1] = d0_x*Vcpml*((x_half[Nx-i-1]/D_pml_x)**2)
            b_x_half[i-1] = np.exp(-(d_x_half[i-1] + alpha_x_half[Nx-i-1]) * dt)
            a_x_half[i-1] = d_x_half[i-1]/(d_x_half[i-1] + alpha_x_half[Nx-i-1]) * (b_x_half[i-1]-1)
            
    return a_x, a_x_half, b_x, b_x_half, a_z, a_z_half, b_z, b_z_half
