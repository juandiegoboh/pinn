float ricker(int tstep, float dt, float fq)
{
#define Pi 3.14159265358979323846
  float f;
  float t;
  t = dt * tstep - 1 / fq;
  f = (1 - 2 * Pi * Pi * fq * fq * t * t) * exp(-Pi * Pi * fq * fq * t * t);
  return (f);
}

float gaussian(int tstep, float dt, float fq)
{
#define Pi 3.14159265358979323846
  float f;
  float t;
  t = dt * tstep - 1 / fq;
  f = exp(-2 * Pi * Pi * fq * fq * t * t);
  return (f);
}

float gaussian_neg(int tstep, float dt, float fq)
{
#define Pi 3.14159265358979323846
  float f;
  float t;
  t = dt * tstep - 1 / fq;
  f = -exp(-2 * Pi * Pi * fq * fq * t * t);
  return (f);
}

void Propagador(int Nx, int Nz, int Sx, int Sz, int Tout, float dh, float dt, float *c, float fq, CPML *cpml /*CPML pointer*/)
{
#define IdS(A, i, j) (A)[(i)*Nz + (j)]
  int i, j, t;
  float *P1 = (float *)calloc(Nx * Nz, sizeof(float));
  float *P2 = (float *)calloc(Nx * Nz, sizeof(float));
  float *P3 = (float *)calloc(Nx * Nz, sizeof(float));
  float *d2pdx = (float *)calloc(Nx * Nz, sizeof(float));
  float *mem_d2pdx = (float *)calloc(Nx * Nz, sizeof(float)); // CPML
  float *d2pdz = (float *)calloc(Nx * Nz, sizeof(float));
  float *mem_d2pdz = (float *)calloc(Nx * Nz, sizeof(float)); // CPML
  float *d2pdt = (float *)calloc(Nx * Nz, sizeof(float));
  float *videop = (float *)calloc(Nx * Nz * Tout, sizeof(float));
  float *fuente = (float *)calloc(Tout, sizeof(float));

  int tstep;
  FILE *volume;

  for (tstep = 0; tstep < Tout; tstep++)
  {
    fuente[tstep] = gaussian_neg (tstep, dt, fq);
  }

  for (t = 0; t < Tout; t++)
  {
    for (i = 1; i < Nx - 1; i++)
    {
      for (j = 1; j < Nz - 1; j++)
      {
        IdS(d2pdx, i, j) = (IdS(P2, i + 1, j) - 2.0 * IdS(P2, i, j) + IdS(P2, i - 1, j)) / (dh * dh);
        IdS(d2pdz, i, j) = (IdS(P2, i, j + 1) - 2.0 * IdS(P2, i, j) + IdS(P2, i, j - 1)) / (dh * dh);

        IdS(mem_d2pdx, i, j) = cpml->bx[i] * IdS(mem_d2pdx, i, j) + cpml->ax[i] * IdS(d2pdx, i, j); // CPML
        IdS(mem_d2pdz, i, j) = cpml->bz[j] * IdS(mem_d2pdz, i, j) + cpml->az[j] * IdS(d2pdz, i, j); // CPML

        IdS(d2pdx, i, j) = IdS(d2pdx, i, j) / cpml->kappa_x[i] + IdS(mem_d2pdx, i, j); // CPML
        IdS(d2pdz, i, j) = IdS(d2pdz, i, j) / cpml->kappa_z[j] + IdS(mem_d2pdz, i, j); // CPML

        IdS(d2pdt, i, j) = IdS(c, i, j) * IdS(c, i, j) * (IdS(d2pdx, i, j) + IdS(d2pdz, i, j));

        IdS(P3, i, j) = IdS(c, i, j) * IdS(c, i, j) * dt * dt * (IdS(d2pdx, i, j) + IdS(d2pdz, i, j)) + 2.0 * IdS(P2, i, j) - IdS(P1, i, j);
      }
    }
    for (i = 1; i < Nx - 1; i++)
    {
      for (j = 1; j < Nz - 1; j++)
      {
        IdS(P1, i, j) = IdS(P2, i, j);
        IdS(P2, i, j) = IdS(P3, i, j);
      }
    }

    IdS(P2, Sx, Sz) = IdS(P2, Sx, Sz) + fuente[t]; // inyecta la fuente

    memcpy(videop + Nx * Nz * t, P3, Nx * Nz * sizeof(float));
  }

  volume = fopen("propagacion2.bin", "wb");
  fwrite(videop, sizeof(float), Nx * Nz * Tout, volume);
  fclose(volume);

  free(P1);
  free(P2);
  free(P3);
  free(d2pdx);
  free(mem_d2pdx); // CPML
  free(d2pdz);
  free(mem_d2pdz); // CPML
  free(d2pdt);
  free(videop);
}
