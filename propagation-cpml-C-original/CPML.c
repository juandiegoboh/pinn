/*
 * Contains functions to calculate and initialize the CPML layer for a model
 * This is performed for a staggered grid
 * Based on the fortran code of Dimitri and Komatisch(2007)
 * https://github.com/komatits/seismic_cpml
 * See CPML.h for more documentation and an example
 */

#include "CPML.h"
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

void calc_cpml_layer_values(CPML* cpml, int i, real abcissa_in_cpml, real* alpha, real* dcpml,
        real* kappa, real cpml_thickness, real d0, real alpha_max_PML);
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
        real vp, real fq){
    cpml->alpha_x = (real*) calloc(nx, sizeof(real));
    cpml->alpha_x_half = (real*) calloc(nx, sizeof(real));
    cpml->alpha_z = (real*) calloc(nz, sizeof(real));
    cpml->alpha_z_half = (real*) calloc(nz, sizeof(real));
    cpml->dcpm_x = (real*) calloc(nx, sizeof(real));
    cpml->dcpm_x_half = (real*) calloc(nx, sizeof(real));
    cpml->dcpm_z = (real*) calloc(nz, sizeof(real));
    cpml->dcpm_z_half = (real*) calloc(nz, sizeof(real));
    cpml->bx = (real*) calloc(nx, sizeof(real));
    cpml->bx_half = (real*) calloc(nx, sizeof(real));
    cpml->bz = (real*) calloc(nz, sizeof(real));
    cpml->bz_half = (real*) calloc(nz, sizeof(real));
    cpml->ax = (real*) calloc(nx, sizeof(real));
    cpml->ax_half = (real*) calloc(nx, sizeof(real));
    cpml->az = (real*) calloc(nz, sizeof(real));
    cpml->az_half = (real*) calloc(nz, sizeof(real));

    cpml->kappa_x = (real*) calloc(nx, sizeof(real));
    cpml->kappa_x_half = (real*) calloc(nx, sizeof(real));
    cpml->kappa_z = (real*) calloc(nz, sizeof(real));
    cpml->kappa_z_half = (real*) calloc(nz, sizeof(real));


    cpml->d0x = -3 * vp * log(cpml->R) / (2 * cpml->n_cpml_x * dx);
    cpml->d0z = -3 * vp * log(cpml->R) / (2 * cpml->n_cpml_z * dz);

    real alpha_max_PML = fq * M_PI;
    real x_origin_min = cpml->n_cpml_x * dx;
    real x_origin_max = (nx - cpml->n_cpml_x) * dx;
    real cpml_thickness_x = cpml->n_cpml_x * dx;
    real z_origin_min = cpml->n_cpml_z * dz;
    real z_origin_max = (nz - cpml->n_cpml_z) * dz;
    real cpml_thickness_z = cpml->n_cpml_z * dz;
    int i;
 

    /*******************************************************
     * X Side
     */
    real xval, abcissa_in_cpml;
    for(i=0; i < nx; i ++){

        xval = i * dx;

        //initialize all kappas to unity to avoid division by 0
        cpml->kappa_x[i] = 1;
        cpml->kappa_x_half[i] = 1;



        if(cpml->CPML_X_MIN){
            // top border
            abcissa_in_cpml = x_origin_min - xval;

            calc_cpml_layer_values(cpml, i, abcissa_in_cpml,
                    cpml->alpha_x,cpml->dcpm_x, cpml->kappa_x, cpml_thickness_x,
                    cpml->d0x, alpha_max_PML);

            //top border for i+1/2 grid cells
            abcissa_in_cpml = x_origin_min - (xval + 0.5 * dx);

            calc_cpml_layer_values(cpml, i, abcissa_in_cpml,
                    cpml->alpha_x_half,
                    cpml->dcpm_x_half,
                    cpml->kappa_x_half,
                    cpml_thickness_x, cpml->d0x, alpha_max_PML);
        }

        if(cpml->CPML_X_MAX){
            // bottom border
            abcissa_in_cpml = xval - x_origin_max;

            calc_cpml_layer_values(cpml, i, abcissa_in_cpml,
                    cpml->alpha_x,cpml->dcpm_x, cpml->kappa_x, cpml_thickness_x,
                    cpml->d0x, alpha_max_PML);

            //bottom border for i+1/2 grid cells
            abcissa_in_cpml = (xval + 0.5 * dx) - x_origin_max;

            calc_cpml_layer_values(cpml, i, abcissa_in_cpml,
                    cpml->alpha_x_half,
                    cpml->dcpm_x_half,
                    cpml->kappa_x_half,
                    cpml_thickness_x, cpml->d0x, alpha_max_PML);
        }

        cpml->bx[i] = Exp(- (cpml->dcpm_x[i] / cpml->kappa_x[i]
                + cpml->alpha_x[i]) * dt);

        cpml->bx_half[i] = Exp(- (cpml->dcpm_x_half[i] / cpml->kappa_x_half[i]
                + cpml->alpha_x_half[i]) * dt);

        if(abs(cpml->dcpm_x[i]) > 1e-6){
           cpml->ax[i] = cpml->dcpm_x[i] / (cpml->kappa_x[i] *
                   (cpml->dcpm_x[i] + cpml->kappa_x[i] * cpml->alpha_x[i]))
                   * (cpml->bx[i] - 1);
        }

        if(abs(cpml->dcpm_x_half[i]) > 1e-6){
           cpml->ax_half[i] = cpml->dcpm_x_half[i] / (cpml->kappa_x_half[i] *
                   (cpml->dcpm_x_half[i] + cpml->kappa_x_half[i] * cpml->alpha_x_half[i]))
                   * (cpml->bx_half[i] - 1);
        }
    }

    /*******************************************************
    * Z Side
    */
    real zval;
    for(i=0; i < nz; i ++){

        zval = i * dz;
        cpml->kappa_z[i] = 1;
        cpml->kappa_z_half[i] = 1;
        if(cpml->CPML_Z_MIN){
            // left border
            abcissa_in_cpml = z_origin_min - zval;

            calc_cpml_layer_values(cpml, i, abcissa_in_cpml,
                    cpml->alpha_z,cpml->dcpm_z, cpml->kappa_z, cpml_thickness_z,
                    cpml->d0z, alpha_max_PML);

            //left border for i+1/2 grid cells
            abcissa_in_cpml = z_origin_min - (zval + 0.5 * dz);

            calc_cpml_layer_values(cpml, i, abcissa_in_cpml,
                    cpml->alpha_z_half,
                    cpml->dcpm_z_half,
                    cpml->kappa_z_half,
                    cpml_thickness_z, cpml->d0z, alpha_max_PML);
        }

        if(cpml->CPML_Z_MAX){
            // right border
            abcissa_in_cpml = zval - z_origin_max;

            calc_cpml_layer_values(cpml, i, abcissa_in_cpml,
                    cpml->alpha_z,cpml->dcpm_z, cpml->kappa_z, cpml_thickness_z,
                    cpml->d0z, alpha_max_PML);

            //right border for i+1/2 grid cells
            abcissa_in_cpml = (zval + 0.5 * dz) - z_origin_max;

            calc_cpml_layer_values(cpml, i, abcissa_in_cpml,
                    cpml->alpha_z_half,
                    cpml->dcpm_z_half,
                    cpml->kappa_z_half,
                    cpml_thickness_z, cpml->d0z, alpha_max_PML);
        }

        cpml->bz[i] = Exp(- (cpml->dcpm_z[i] / cpml->kappa_z[i]
                + cpml->alpha_z[i]) * dt);

        cpml->bz_half[i] = Exp(- (cpml->dcpm_z_half[i] / cpml->kappa_z_half[i]
                + cpml->alpha_z_half[i]) * dt);

        if(abs(cpml->dcpm_z[i]) > 1e-6){
           cpml->az[i] = cpml->dcpm_z[i] / (cpml->kappa_z[i] *
                   (cpml->dcpm_z[i] + cpml->kappa_z[i] * cpml->alpha_z[i]))
                   * (cpml->bz[i] - 1);
        }

        if(abs(cpml->dcpm_z_half[i]) > 1e-6){
           cpml->az_half[i] = cpml->dcpm_z_half[i] / (cpml->kappa_z_half[i] *
                   (cpml->dcpm_z_half[i] + cpml->kappa_z_half[i] * cpml->alpha_z_half[i]))
                   * (cpml->bz_half[i] - 1);
        }


    }
}


void calc_cpml_layer_values(CPML* cpml, int i, real abcissa_in_cpml, real* alpha, real* dcpml,
        real * kappa,
        real cpml_thickness, real d0, real alpha_max_PML){
    real abcissa_normalized;


    //only build non-zero cpml values if inside the current layer
    if(abcissa_in_cpml > 0){

        abcissa_normalized = (abcissa_in_cpml  / cpml_thickness);

        // See the komatisch and martin on the choice of kappa. from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
        kappa[i] = 1 + (cpml->kappa_max - 1) * pow(abcissa_normalized, cpml->CPML_layer_power);

        dcpml[i] = d0 * pow(abcissa_normalized, cpml->CPML_layer_power);

        alpha[i] = (1 - abcissa_normalized) * alpha_max_PML
                + 0.1 * alpha_max_PML;
    }
}
