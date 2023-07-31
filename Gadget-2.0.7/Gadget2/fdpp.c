#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include <fftw3.h>
#include <time.h>
#include <mpi.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include "neutrino.h"
#include "fsvars.h"
#include "allvars.h"

struct fdpp_params {double x; double y; double z; double l;};

double perturb_fd_integrand(double pot, void *par){
    
    double part;
    double mass_n;
    
    struct fdpp_params * params = (struct fdpp_params *)par;
    double Tneu = (params->x);
    double phi = (params->y);
    double mass_nu = (params->z);
    double nu_xi = (params->l);
    
    mass_n = mass_nu/ktoev;
    part = pow(pot, 2) / (exp((pot) - nu_xi + phi * mass_n / Tneu / cc) + 1.0) + pow(pot, 2) / (exp((pot) + nu_xi + phi * mass_n / Tneu / cc) + 1.0);
    //part = pow(pot, 3) * sqrt(1 + pow(mass_n/(pot*Tneu), 2)) / (exp((pot) - nu_xi + phi * mass_n / Tneu / cc) + 1.0) + pow(pot, 3) * sqrt(1 + pow(mass_n/(pot*Tneu), 2)) / (exp((pot) + nu_xi + phi * mass_n / Tneu / cc) + 1.0);
    //printf("part %f\n", part);
    return part;
    
}



double perturb_fd_integration(double a, double phi, double m, double xi)
{
    double Tneu = All.Tneu0/a;
    struct fdpp_params alpha = {Tneu, phi, m, xi};
    
#define WORKSIZE2 100000
    gsl_function F2;
    gsl_integration_workspace *workspace2;
    
    double inte_result, inte_abserr;
    double roneu;
    
    workspace2 = gsl_integration_workspace_alloc(WORKSIZE2);
    
    F2.function = &perturb_fd_integrand;
    F2.params = &alpha;
    gsl_integration_qagiu(&F2, 0.0, 0, 1.0e-8, WORKSIZE2, workspace2, &inte_result, &inte_abserr);
    roneu = inte_result * All.unittrans * pow(Tneu, 3) / kb;
    //roneu = inte_result*unittrans*pow(Tneu, 4);
    gsl_integration_workspace_free(workspace2);
    //printf("roneu %f inte_result %f unittrans %f\n", roneu, inte_result, All.unittrans);
    return 3*roneu;
}




/*void phi_perturb_correction(double *wwww){
    int xi, yi, zi, xtemp, ytemp, ztemp;
    double k, kx, ky, kz;
    double average;
    //double Omega_nu0 = 0.007396;
    double  H0 = 100. * All.HubbleParam * 1000. / ((float)(1E6 * 3.0857E16));
    double rrp = 8.0*M_PI*Gr/(cc*3.0*H0*H0);
    double rocr = 1. /rrp;
    double mpc_to_m = 3.0857e22;
    
    average = (double)(NumPart) * P[1].Mass/ (double)(pow(PMGRID+1, 3));
    
    int nmid = PMGRID / 2;
    printf("check point 0\n");
    fftw_complex *fluc_density_field;
    fluc_density_field = (fftw_complex*) malloc((PMGRID)*(PMGRID)*(PMGRID) * sizeof(fftw_complex));
    //fluc_density_field = (fftw_complex*) fftw_malloc(PMGRID*PMGRID*(PMGRID) * sizeof(fftw_complex));
    fftw_complex *out;
    out = (fftw_complex*) malloc((PMGRID)*(PMGRID)*(PMGRID) * sizeof(fftw_complex));
    //out = (fftw_complex*) fftw_malloc((PMGRID)*(PMGRID)*(PMGRID) * sizeof(fftw_complex));
    
    fftw_complex *phik;
    fftw_complex *phi_real;
    
    fftw_complex *delta_ro;
    //delta_ro = (fftw_complex*) fftw_malloc((PMGRID)*(PMGRID)*(PMGRID) * sizeof(fftw_complex));
    delta_ro = (fftw_complex*) fftw_malloc((PMGRID+1)*(PMGRID+1)*(PMGRID+1) * sizeof(fftw_complex));
    
    
    //phik = (fftw_complex*) malloc((PMGRID)*(PMGRID)*(PMGRID) * sizeof(fftw_complex));
    //phi_real = (fftw_complex*) malloc((PMGRID)*(PMGRID)*(PMGRID) * sizeof(fftw_complex));
    
    phik = (fftw_complex*) malloc((PMGRID + 1)*(PMGRID + 1)*(PMGRID + 1) * sizeof(fftw_complex));
    phi_real = (fftw_complex*) malloc((PMGRID + 1)*(PMGRID + 1)*(PMGRID + 1) * sizeof(fftw_complex));
   
    fftwnd_plan fluc_density_field_plan, plan_of_phi;
    fluc_density_field_plan = rfftw3d_create_plan(PMGRID, PMGRID, PMGRID, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE);
    //fluc_density_field_plan = fftw_create_plan((PMGRID+1)*(PMGRID+1)*(PMGRID+1), FFTW_FORWARD, FFTW_ESTIMATE);
    plan_of_phi = fftw3d_create_plan(PMGRID, PMGRID, PMGRID, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_IN_PLACE);
    //fftwnd_mpi_plan fluc_density_field_plan, plan_of_phi;
    
    printf("check point 1 PMGRID %d average %f\n", PMGRID, average);
    
    for(xi=0;xi<=PMGRID;xi++){
            for(yi=0;yi<=PMGRID;yi++){
                for(zi=0;zi<=PMGRID;zi++){
                    //fluc_density_field[zi + (PMGRID+1)*(yi + (PMGRID+1)*xi)].re = (wwww[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)] - average)/average;
                    fluc_density_field[zi + (PMGRID+1)*(yi + (PMGRID+1)*xi)].re = (wwww[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)])/average;
                    fluc_density_field[zi + (PMGRID+1)*(yi + (PMGRID+1)*xi)].im = 0.;
                    out[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re = 0.;
                    out[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].im = 0.;
                    //printf("try %f  %f xi %d yi %d zi %d \n", fluc_density_field[zi + (PMGRID+1)*(yi + (PMGRID+1)*xi)].re, fluc_density_field[zi + (PMGRID+1)*(yi + (PMGRID+1)*xi)].im, xi, yi, zi);
                }
            }
        }
            printf("check point 1 PMGRID %d\n", PMGRID);
        //fftw_execute(fluc_density_field_plan);
    //fluc_density_field_plan = fftw3d_mpi_create_plan(MPI_COMM_WORLD, PMGRID, PMGRID, PMGRID, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_IN_PLACE);

   
    //fftwnd_one(fluc_density_field_plan, fluc_density_field, out);
    rfftwnd_one_real_to_complex(fluc_density_field_plan, wwww, out);
    //fftwnd_mpi(fluc_density_field_plan, 1, fluc_density_field, out, FFTW_TRANSPOSED_ORDER);

        for(xi=0;xi<=PMGRID;xi++){
            for(yi=0;yi<=PMGRID;yi++){
                for(zi=0;zi<=PMGRID;zi++){
                    
                    phik[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re = 0.;
                    phik[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].im = 0.;
                    phi_real[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re = 0.;
                    phi_real[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].im = 0.;
                    printf("out %f %d %d %d\n", out[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re, xi, yi, zi);
                }
            }
        }
        
    //for(xi=0;xi<129*129*129;xi++){
    //    printf("in re %lf im %lf\n", fluc_density_field[xi].re, fluc_density_field[xi].im);
     //   printf("out re %lf im %lf\n", out[xi].re, out[xi].im);
    //}
        printf("check point 2\n");
        for(xi=0;xi<=PMGRID;xi++){
            for(yi=0;yi<=PMGRID;yi++){
                for(zi=0;zi<=PMGRID;zi++){
                    
                    if(xi>nmid){
                        xtemp = PMGRID + 1 - xi;
                    }
                    else{
                        xtemp =  - xi;
                    }
                    
                    if(yi>nmid){
                        ytemp = PMGRID + 1 - yi;
                    }
                    else{
                        ytemp =  - yi;
                    }
                    
                    if(zi>nmid){
                        ztemp = PMGRID + 1 - zi;
                    }
                    else{
                        ztemp =  - zi;
                    }
                    
                    kx = (2*M_PI*1000/All.BoxSize) * xtemp;  //MPc ^ -1
                    ky = (2*M_PI*1000/All.BoxSize) * ytemp;
                    kz = (2*M_PI*1000/All.BoxSize) * ztemp;
                    
                    k = sqrt(kx*kx + ky*ky + kz*kz);
                    
                    if(k == 0){
                        phik[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re = 0.;
                        phik[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].im = 0.;
                    }
                    
                    else{
                        phik[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re = (0. - out[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re) * rocr / cc * All.Omega0 * 4. * M_PI * Gr / ((All.Time * All.Time * All.Time) * (k * k / mpc_to_m / mpc_to_m) * pow((PMGRID+1), 3));
                        phik[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].im = (0. - out[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].im) * rocr / cc * All.Omega0 * 4. * M_PI * Gr / ((All.Time * All.Time * All.Time) * (k * k / mpc_to_m / mpc_to_m) * pow((PMGRID+1), 3));
                        //printf("phi = %lf, k = %lf out = %lf norm %f\n", phik[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re, mpc_to_m * mpc_to_m / k / k / 1e40, out[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re * rocr * All.Omega0 * 4. * M_PI * Gr * 1e15, pow((PMGRID+1), 3));
                    }
                }
            }
        }
        
    
    //plan_of_phi = fftw3d_mpi_create_plan(MPI_COMM_WORLD, PMGRID, PMGRID, PMGRID,
      //                                               FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_IN_PLACE);
    
    //fftwnd_mpi(plan_of_phi, 1, phik, phi_real, FFTW_TRANSPOSED_ORDER);
    printf("check point 3\n");
    
    fftwnd_one(plan_of_phi, &phik[0], &phi_real[0]);
    fftwnd_destroy_plan(plan_of_phi);

        //fftw_execute(plan_of_phi);
        
 
        //for(xi=0;xi<=PMGRID;xi++){
         //   for(yi=0;yi<=PMGRID;yi++){
          //      for(zi=0;zi<=PMGRID;zi++){
                    
                    //phi_real[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re = phi_real[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re;
                    //phi_real[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].im = phi_real[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].im;
                    //phi_real[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].im = phi_real[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].im / pow((PMGRID+1), 3);
                    //printf("phi_real imaginary %f real %f phik %lf\n", phi_real[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].im, phi_real[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re, phik[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re);
                    //printf("phi/T %f phi %f\n", phi_real[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re * 0.1 / ktoev / cc / 1.7, phi_real[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re);
            //    }
          //  }
        //}
        
        double ev_to_kg = 1.78266191E-36;
        double mass_of_sun = 1.989E30;
        double average2;
        double rosum = 0.;
        average2 = rocr * All.Omega0 * pow((200 * mpc_to_m / h), 3) / cc / mass_of_sun / 1e10 * All.HubbleParam / 33.5;
        printf("check point 4\n");
        for(xi=0;xi<=PMGRID;xi++){
            for(yi=0;yi<=PMGRID;yi++){
                for(zi=0;zi<=PMGRID;zi++){
                    
                    delta_ro[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re = mass_neu * ev_to_kg * perturb_fd_integration(All.Time, phi_real[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re) * pow((mpc_to_m * All.BoxSize / 1000. / (PMGRID+1)), 3) / All.HubbleParam/ All.HubbleParam/ mass_of_sun / 1E10 / P[1].Mass;
                    
                    rosum = rosum + delta_ro[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re / average;
                    //delta_ro[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re = mass_neu * ev_to_kg * perturb_fd_integration(All.Time, 0.) * pow((mpc_to_m * All.BoxSize / 1000. / PMGRID), 3) * All.HubbleParam* All.HubbleParam/ mass_of_sun / 1E10 / P[1].Mass;
                    //delta_ro[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re = perturb_fd_integration(All.Time, 0.) * pow((mpc_to_m * All.BoxSize / 1000. / PMGRID), 3) * All.HubbleParam * All.HubbleParam / mass_of_sun / 1E10 / P[1].Mass / cc;
                    //double cdm = wwww[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)];
                    //double neuneu = mass_neu * ev_to_kg * perturb_fd_integration(All.Time, phi_real[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re) * cc;
                    
                    //if(cdm > 1){
                     //printf("delta_ro = %lf, cdm mass density %lf mass %lf average %lf ro %lf omega ratio %lf\n", delta_ro[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re / average, cdm, P[1].Mass, average, Omega_nu0);
                     //}
                    
                }
            }
        }
        rosum = rosum / pow(PMGRID+1, 3);
        printf("rosum/average %lf Omega_nu0 / All.Omega0 %lf\n", rosum, neutrino_integration(1.) / rocr / All.Omega0);
    printf("check point 5\n");
    int correction_nu_distribution = 2;
    for(xi=0;xi<=PMGRID;xi++){
        for(yi=0;yi<=PMGRID;yi++){
            for(zi=0;zi<=PMGRID;zi++){
                double roro = delta_ro[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)].re;
                if(correction_nu_distribution == 1){
                    //fluc_density_field[zi + (PMGRID+1)*(yi + (PMGRID+1)*xi)].re = (density_field[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)] * All.Omega0 / (All.Omega0 + Omega_nu0) + (roro + 1.) * average * Omega_nu0 / (All.Omega0 + Omega_nu0) - average)/average;
                    wwww[zi + (PMGRID+1)*(yi + (PMGRID+1)*xi)] = wwww[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)] + (roro + 1.) * average * Omega_nu0 / All.Omega0;
                }
                
                if(correction_nu_distribution == 2){
                    wwww[zi + (PMGRID+1)*(yi + (PMGRID+1)*xi)] = wwww[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)] * All.Omega0 / (All.Omega0 + Omega_nu0) + roro;
                }
                //if(fluc_density_field[zi + (PMGRID+1)*(yi + (PMGRID+1)*xi)] != 0.){
                
                //printf("dens = %f fluc = %f ro %lf ave = %f xyz = %d %d %d\n", density_field[zi+(PMGRID+1)*(yi + (PMGRID+1)*xi)], fluc_density_field[zi + (PMGRID+1)*(yi + (PMGRID+1)*xi)].re, roro, average, xi, yi, zi);
                //}
            }

        }
}
}*/
