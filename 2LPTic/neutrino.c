/* These amendment neutrino.c and neutrino.h add the influence
 coefficient relation shows the influence may not be neglectable.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>

#include "allvars.h"
#include "proto.h"
#include "neutrino.h"

struct neutrino_params
{
    double T;
    double m;
    double xi; 
    /* data */
};

/* here I replace the original approximated formula to the 
accuracy one, in which the mass is not negelectable in 
exponential. */
double neutrino_partition(double PT, void *Params)
{
    struct neutrino_params* Param = (struct neutrino_params *) Params;
    double partition;
    double Tnu = Param->T;
    double Mass_nu = Param->m;
    double Xi_nu = Param->xi;
    double Mass_n;

    Mass_n = Mass_nu / ktoev;
    partition = pow(PT, 3) * sqrt(1 + pow(Mass_n / (PT * Tnu), 2)) / (exp(PT - Xi_nu) + 1);
    partition += pow(PT, 3) * sqrt(1 + pow(Mass_n / (PT * Tnu), 2)) / (exp(PT + Xi_nu) + 1);

    return partition;
}

double neutrino_integration(double a, double m, double xi, int neutrinoindex)
{
    #define WORKSIZE2 100000
    gsl_function F2;
    gsl_integration_workspace *workspace2;

    double integrate_result, integrate_abserr;
    double Rho_nu;
    double T_nu;
    T_nu = Tneu0 / a;
    struct neutrino_params alpha = {T_nu, m, xi};

    workspace2 = gsl_integration_workspace_alloc(WORKSIZE2);
    F2.function = &neutrino_partition;
    F2.params = &alpha;

    gsl_integration_qagiu(&F2, 0.0, 0, 1.0e-8, WORKSIZE2, workspace2, &integrate_result, &integrate_abserr);
    Rho_nu = integrate_result * unittrans * pow(T_nu, 4);
    Rho_nu = Rho_nu / rocr;
    gsl_integration_workspace_free(workspace2);

#ifdef STERILE
    if (neutrinoindex == STERILE)
    {
        Rho_nu *= (Neff - 3.046);
    }
#endif
    return Rho_nu;
}


