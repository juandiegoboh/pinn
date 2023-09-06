#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "CPML.h"       //CPML
#include "funciones2.h" //CPML
#define IdS(A, i, j) (A)[(i)*Nz + (j)]
int main()
{
  int Nx = 450;
  int Nz = 150;
  int Sx = 150;
  int Sz = 75;
  int Tout = 800;
  int i, j;
  float c_max = 0.0;
  float *c = (float *)calloc(Nx * Nz, sizeof(float));

  float dh = 3.333333333;
  float fq = 16; // Play for dispersion from 10 to 100
  float dt = 0.0;

  int borde = 30; // amount of CPML grid points
  CPML *cpml = malloc(sizeof(CPML));
  cpml->n_cpml_x = borde;
  cpml->n_cpml_z = borde;
  cpml->CPML_layer_power = 2; // attenuation power in CPML formula
  cpml->R = 0.0001;           // see Komatisch and Martin 2007
  cpml->kappa_max = 1;        // for the definition of the constants
  cpml->CPML_X_MAX = true;    // true                //turn cpml layers on (true) or off (false)
  cpml->CPML_X_MIN = true;    // true
  cpml->CPML_Z_MAX = true;    // false
  cpml->CPML_Z_MIN = true;    // true

  for (i = 0; i < Nx; i++)
  {
    for (j = 0; j < Nz; j++)
    {
      IdS(c, i, j) = 3000.0;
    }
  }

  // // Leer la matriz desde el archivo binario
  // FILE *archivo = fopen("../experimentos/experimento_9/velocity_models/modelo_velocidad_9.bin", "rb");

  // if (archivo == NULL)
  // {
  //   printf("Error al abrir el archivo.\n");
  //   return 1; // Terminar el programa con un código de error
  // }

  // fread(c, sizeof(float), Nx * Nz, archivo);
  // fclose(archivo);

  // Velocidad maxima
  for (i = 0; i < Nx; i++)
  {
    for (j = 0; j < Nz; j++)
    {
      if (c_max < IdS(c, i, j))
      {
        c_max = IdS(c, i, j);
      }
    }
  }

  dt = dh / (c_max * sqrt(2));                          // Keeping stability
  initialize_CPML(cpml, Nx, Nz, dh, dh, dt, c_max, fq); // Initialize CPML

  Propagador(Nx, Nz, Sx, Sz, Tout, dh, dt, c, fq, cpml);

  system("nimage < propagacion2.bin n1=200 n2=200 legend=1 title='Propagacion2' &");

  return 0;
}
