

/* 
 * File:   CPML.h
 * Defines the structure for the CPML layer
 * Author: jaap (Based on the acoustic2D fortran code of komatits and martin, but adapted
 *               to be generally usable as a C structure for any type of 2D equations)
 *              https://github.com/komatits/seismic_cpml
 *
 * Created on April 17, 2018, 1:34 PM
 * 
 *
 * 
 * EXAMPLE USE:
 * 
 *
 *  
        //CPML constants
        CPML* cpml = malloc(sizeof(CPML));
        cpml->n_cpml_x = 10;                    // amount of CPML grid points
        cpml->n_cpml_z = 10;
        cpml->CPML_layer_power = 2;             // attenuation power in CPML formula
        cpml->R = 0.001;                        // see Komatisch and Martin 2007
        cpml->kappa_max = 1;                    // for the definition of the constants
        cpml->CPML_X_MAX = true;                //turn cpml layers on or off
        cpml->CPML_X_MIN = true;
        cpml->CPML_Z_MAX = true;
        cpml->CPML_Z_MIN = false;

        //nx_grid is the amount of points in your model of the x direction
        //nz grid is the amount of points in your model of the z direction
        //define the extra grid spacing needed for CPML(model with added cpml points)
        int nx = nx_grid + (cpml->CPML_X_MIN + cpml->CPML_X_MAX) * cpml->n_cpml_x;
        int nz = nz_grid + (cpml->CPML_Z_MIN + cpml->CPML_Z_MAX) * cpml->n_cpml_z;

        //Note that you can also keep your original model dimensions 
        // eg. nx = nx_grid, nz = nz_grid. 
        //then the cpml will act inside your model.
 
  
 * 
 * After setting the values above, call the initialization function to calculate
 * a(x), a(z), b(x), b(z) as described in Komatisch and Martin
 * see doc below for parameter explanation
 
        initialize_CPML(CPML* cpml, int nx, int nz, real dx, real dz, real dt,
               real vp, real fq)

 * Now you can use cpml->ax and cpml->ax_half etc if you have a staggered grid,
 * if not, you can just use ax, az, bx, bz, kappa_x, kappa_z
 * 
 
 * Example implementation in a propagation loop:
 *  (this is the adjoint elastic stress propagation using a staggered grid)
 * The memory variables are used for CPML and are of size nx * nz;
 * 
        real aVx_dx, aVx_dz, aVz_dx, aVz_dz;
         //propagate adjoint stress fields
        for(int i = 1;i < nx - 1;i++){
             for(int j = 1; j < nz - 1; j++){
                 aVx_dx = (Id2(aVx,i+1,j) - Id2(aVx,i,j)) / dx;
                 aVx_dz = (Id2(aVx,i,j+1) - Id2(aVx,i,j)) / dz;
                 aVz_dz = (Id2(aVz,i,j) - Id2(aVz,i,j-1)) / dz;
                 aVz_dx = (Id2(aVz,i,j) - Id2(aVz,i-1,j)) / dx;

                 //Update CPML components using correct values at the half-spaced 
                 // staggered grid points
                 Id2(mem_aVx_dx,i,j) = cpml->bx_half[i] * Id2(mem_aVx_dx,i,j) + cpml->ax_half[i] * aVx_dx;
                 Id2(mem_aVx_dz,i,j) = cpml->bz_half[j] * Id2(mem_aVx_dz,i,j) + cpml->az_half[j] * aVx_dz;
                 Id2(mem_aVz_dx,i,j) = cpml->bx[i] * Id2(mem_aVz_dx,i,j) + cpml->ax[i] * aVz_dx;
                 Id2(mem_aVz_dz,i,j) = cpml->bz[j] * Id2(mem_aVz_dz,i,j) + cpml->az[j] * aVz_dz;

                 //alter the derivatives with the cpml vars
                 aVx_dx = aVx_dx / cpml->kappa_x_half[i] + Id2(mem_aVx_dx,i,j);
                 aVx_dz = aVx_dz / cpml->kappa_z_half[j] + Id2(mem_aVx_dz,i,j);
                 aVz_dx = aVz_dx / cpml->kappa_x[i] + Id2(mem_aVz_dx,i,j);
                 aVz_dz = aVz_dz / cpml->kappa_z[j] + Id2(mem_aVz_dz,i,j);


                 //propagation formula, note the minus sign we backpropagate in time
                 Id2(aTxx,i,j) = Id2(aTxx,i,j) - dt * aVx_dx;
                 Id2(aTzz,i,j) = Id2(aTzz,i,j) - dt * aVz_dz;
                 Id2(aTxz,i,j) = Id2(aTxz,i,j) - dt * (aVx_dz + aVz_dx);
             }   
         }
 
 * 
 * 
 * 
 */

#ifndef CPML_H
#define CPML_H

#include "macros.h"


typedef struct {
   //CPML constants
    int n_cpml_x;
    int n_cpml_z;                       // amount of CPML grid points
    real CPML_layer_power;         // attenuation power in CPML formula
    real kappa_max;                   // see Komatisch and Martin 2007
    real R;                    // for the definition of the constants
    boolean CPML_X_MIN; 
    boolean CPML_X_MAX;
    boolean CPML_Z_MIN;
    boolean CPML_Z_MAX;
    
// Define the attenuation functions see Komatisch and Martin 2007
    real* alpha_x;
    real* alpha_x_half;
    real* alpha_z;
    real* alpha_z_half;
    real* dcpm_x;
    real* dcpm_x_half;
    real* dcpm_z;
    real* dcpm_z_half;

    real* bx;
    real* bx_half;
    real* bz;
    real* bz_half;
    real* ax;
    real* ax_half;
    real* az;
    real* az_half;
    
    real* kappa_x;
    real* kappa_x_half;
    real* kappa_z;
    real* kappa_z_half;
    
    real d0x, d0z;
    
} CPML;

/**
 * Initializes CPML variables 
 * @param cpml pointer to the CPML struct which we want to initialize
 * @param nx amount of points in the x-direction(including the CPML points already)
 * @param nz amount of points in the z-direction(including the CPML points already)
 * @param dx delta x
 * @param dz delta z
 * @param vp max wavespeed inside the layer(used for calculating the d0 value)
 * @param fq frequency of the source(used for cacluating the alpha value)
 */
void initialize_CPML(CPML* cpml, int nx, int nz, real dx, real dz, real dt,
        real vp, real fq);


#endif /* CPML_H */

