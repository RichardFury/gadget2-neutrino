#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

#include "allvars.h"
#include "proto.h"

void Neutrino_init();
double hubble(double a);
double neutrino_integration(double a, double m, double xi, int neutrinoindex);
double numdens(double xi, int neutrinoindex);
double frstr(double k, double delta_nu_0, int neutrinoindex, int kindex);
void Neutrino_free();
void Table_rho_init(int num_kbins);
