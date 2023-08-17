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
#include "power.h"

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
    struct neutrino_params *Param = (struct neutrino_params *)Params;
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

/* initialize the neutrino cosmology. */
void init_cosmology(void)
{
    int i;
    /* Unit conversion. */
    H0 = 100. * HubbleParam * 1000. / ((float)(1E6 * 3.0857E16));
    unittrans = pow(kb, 4) / ((pow(hbar, 3)) * (pow(c, 3)) * 2 * M_PI * M_PI);
    rocr = (cc * 3.0 * H0 * H0) / (8.0 * M_PI * Gr);

    /* Determine lepton asymmetry. */
    switch (lepton_asymmetry)
    {
        case 1:
            Xi[1] = cal_xi2(Xi[2]);
            Xi[0] = cal_xi1(Xi[1], Xi[2]);
            break;

        case 0:
            Xi[1] = Xi[2];
            Xi[0] = Xi[2];
            break;

        default:
            break;
    }

    /* Determine the neutrino mass hierachy. */
    switch (mass_hierarchy)
	{
	    case 0:
	    	/* Normal Hierarchy */
	    	Mass[1] = sqrt(Mass[0] * Mass[0] + DELTA_MASS_SMALL);
	    	Mass[2] = sqrt(Mass[1] * Mass[1] + DELTA_MASS_LARGE);
	    	break;
    
	    case 1:
	    	/* Inverted Hierarchy */
	    	Mass[1] = sqrt(Mass[0] * Mass[0] + DELTA_MASS_LARGE);
	    	Mass[2] = sqrt(Mass[1] * Mass[1] + DELTA_MASS_SMALL);
	    	break;

	    case 2:
	    	/* Identical */
	    	Mass[1] = Mass[0];
	    	Mass[2] = Mass[0];
	    	break;

	    default:
	    	break;
	}
    
    /* Determine the expansion history, calculate the energy 
    density for each species. */
    if (expan_on == 1)
	{
		Omega_nu0_expan = 0.0;
		for (i = 0; i < NNeutrino; i++)
		{
			Omega_nu0_expan += neutrino_integration(1.0, Mass[i], Xi[i], i);
		}
		
	}
#ifdef STERILE
	Omega_nu0_frstr = neutrino_integration(1.0, 0.0, 0.0, -1) * (NNeutrino + Neff - 4.046);
#else
	Omega_nu0_frstr = neutrino_integration(1.0, 0.0, 0.0, -1) * NNeutrino;
#endif

    if(deductfromDE == 1)
    {
        OmegaLambda = OmegaLambda - Omega_nu0_expan;
        Omega2 = Omega;
        Omega = Omega + Omega_nu0_expan;
    }
    if(deductfromDE == 0)
    {
        OmegaLambda = OmegaLambda;
        Omega2 = Omega - Omega_nu0_expan;
        Omega = Omega;
    }

    /* Calculate the growth factor. */
    double a_ini = 1 / (1 + Redshift);
    D11 = growth(1.0);
    D10 = growth(a_ini);
    double theta = 2.7250 / 2.7;
    D11 *= (1 + 2.5 * 1e4 * (Omega) * HubbleParam * HubbleParam * pow(theta, -4));
    D11 *= 2.5 * (Omega);
    D10 *= (1 + 2.5 * 1e4 * (Omega) * HubbleParam * HubbleParam * pow(theta, -4));
    D10 *= 2.5 * (Omega);

    /* Output neutrino cosmology information */
    if(ThisTask == 0)
    {
        printf("Neutrino information:\n");
        printf("Type \t");
        for(i = 0; i < NNeutrino; i++) printf("%d\t ", i + 1);
        printf("\nMass \t");
        for(i = 0; i < NNeutrino; i++) printf("%.03f \t", Mass[i]);
        printf("\nxi   \t");
        for(i = 0; i < NNeutrino; i++) printf("%.03f \t", Xi[i]);
        printf("\ng_i  \t");
        for(i = 0; i < NNeutrino; i++)
        {
#ifdef STERILE
            if(i == STERILE) printf("%.03f \t", Neff);
            else printf("1.015 \t");
#else
            printf("1.015 \t");
#endif
        }
        printf("\nCosmological information:\n");
        printf("Omega_m:                     \t%f\n", Omega);
        printf("Omega_cb:                    \t%f\n", Omega2);
        printf("Omega_L:                     \t%f\n", OmegaLambda);
        printf("Omega_nu at z=0:             \t%f\n", Omega_nu0_expan);
        printf("Omega_nu massless fiducial:  \t%f\n", Omega_nu0_frstr);
        printf("\nCosmology growth information:\n");
        pirntf("D11:                         \t%f\n", D11);
        printf("D10:                         \t%f\n", D10);
        printf("d11:                         \t%f\n", growth(1.0));
        printf("d10:                         \t%f\n", growth(a_ini));
    }
}

double cal_xi2(double xi3){
    double AA, BB, CC, DD, t12, t13, s13, s23, c23, s12, c12, c13, r23;
    double xi2, L3, L2;
    int i;
    
    s12 = sqrt(0.304);
    s23 = sqrt(0.51);
    s13 = sqrt(0.0219);
    c23 = sqrt(1. - 0.51);
    c12 = sqrt(1. - 0.304);
    c13 = sqrt(1. - 0.0219);
    t12 = s12 / c12;
    t13 = s13 / c13;
    
    AA = c23*((1.- t12*t12)*c23 - 2.*s13*s23*t12);
    BB = ((1.- t13*t13)*s23*s23 - t12*t13*t13*c23*(2.*s13*s23+t12*c23));
    CC = s23*((1.-t12*t12)*s23+2.*s13*c23*t12);
    DD = ((1.- t13*t13)*c23*c23 + t12*t13*t13*s23*(2.*s13*c23 - t12*s23));
    
    r23 = (DD - BB) / (AA - CC);

    L3 = xi3 * (xi3 * xi3 + M_PI * M_PI);
    L2 = r23 * L3;
    
    xi2 = L2 / (M_PI * M_PI);
    for(i=0;i<10;i++){
        xi2 = L2 / (xi2 * xi2 + M_PI * M_PI);
       //printf("xi2 %f\n", xi2);
    }
    
    return xi2;
}

double cal_xi1(double xi2, double xi3){
    double s13, s23, c23, s12, c12, c13, r23;
    double xi1, L3, L2, L1;
    int i;
    
    s12 = sqrt(0.304);
    s23 = sqrt(0.51);
    s13 = sqrt(0.0219);
    c23 = sqrt(1. - 0.51);
    c12 = sqrt(1. - 0.304);
    c13 = sqrt(1. - 0.0219);

    
    L3 = xi3 * (xi3 * xi3 + M_PI * M_PI);
    L2 = xi2 * (xi2 * xi2 + M_PI * M_PI);
    
    L1 = - (s13*s13*L3 + c13*c13*s12*s12*L2) / (c13*c13*c12*c12);
	//printf("L1: %f L2:%f L3: %f xi1: %f\n", L1, L2, L3, xi1);
    
    xi1 = L1 / (M_PI * M_PI);
    for(i=0;i<10;i++){
        xi1 = L1 / (xi1 * xi1 + M_PI * M_PI);
      //printf("xi1 %f\n", xi1);
    }
    
    return xi1;
}
