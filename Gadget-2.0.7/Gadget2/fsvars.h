#ifndef FSVARS_H
#define FSVARS_H

#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include "tags.h"

#define hbar                1.05457173E-34    //m^2*kg*s^-1
#define kb                  1.3806488E-23     //m^2*kg*s^-2*k^-1
#define Gr                  6.67384E-11       //m^3*kg^-1*s^-2
#define clight              2.9979e8          //m*s^-1
#define cc                  8.9874e16
#define ktoev               8.617e-5
#define mpc_to_m            3.0857e22
#define ev_to_kg            1.78266191E-36
#define mass_of_sun         1.989E30

extern struct free_streaming
{
  double *h;
  double *fnu, *Omega_nu_temp;
  double fnu_total, num_kbins, Omega_nu_temp_total;
  double **Phiarray, *Phiarrayk, **Phiarray_local;
  double *loga, *s;
  double **rho_tot;     /* Delta_tot contains delta_tot(k, a)*/
  double *loga_for_rho;    /* corresponding a for Delta_tot, in log scale. */
  int count;              /* the number of delta_tot */
  gsl_interp_accel *acc_s; /* variable used for interpolation. */
  gsl_spline *spline_s; 
  gsl_interp_accel *acc_h;
  gsl_spline *spline_h;
  gsl_interp_accel *acc_phi;
  gsl_spline *spline_phi;
} 
  Fs;

#endif
