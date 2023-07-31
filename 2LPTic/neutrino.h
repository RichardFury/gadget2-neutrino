#include <stdio.h>
#include <gsl/gsl_rng.h>

#define hbar                1.05457173E-34   //m^2*kg*s^-1
#define kb                  1.3806488E-23    //m^2*kg*s^-2*k^-1
#define Gr                  6.67384E-11      //m^3*kg^-1*s^-2
#define  c                  2.9979e8        //m*s^-1
#define   cc                8.9874e16
#define ktoev               8.617e-5
#define mpc_to_m            3.0857e22
#define ev_to_kg            1.78266191E-36
#define mass_of_sun         1.989E30
#define DELTA_MASS_LARGE    2.32e-3
#define DELTA_MASS_SMALL    7.59e-5

double neutrino_partition(double p, void *param);
double neutrino_integration(double a, double m, double xi, int neutrinoindex);