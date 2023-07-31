#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include <time.h>

#include "allvars.h"
#include "proto.h"
#include "neutrino.h"
#include "fsvars.h"

/*! \file neutrino.c
 *  \brief neutrino free-streaming algorithm
 *  
 *  This file contains various function to initialize the neutrino and 
 *  calculate the neutrino overdensity field via free-streaming method.
 */

/*
 *! This fucntion performs the initial set-up of neutrino part, and also
 *  give serval table related to free-streaming method in order to speed
 *  up the code.
 */
struct int_params 
{ 
  double k; 
  int neutrinoindex; 
  gsl_spline *spline_rho; 
  gsl_interp_accel *acc_rho; 
  double sf;
};
struct neu_params {double Tnu; double m; double xi;};

double hubble(double a)
{
  double hub, rrp, rom, ror, rolambda, roneu, rototal;
  int i;
  rom = All.Omega2 * All.rocr / (a * a * a);
  rolambda = All.OmegaLambda * All.rocr;
  if(All.expan_on == 1)
    {
      roneu = 0.0;
	    for(i = 0; i < All.NNeutrino; i++)
        {
	        roneu += neutrino_integration(a, All.mass[i], All.xi[i], i);
	      }
    }
  else
    {
#ifdef STERILE
      roneu = neutrino_integration(a, 0., 0., -1) * (All.NNeutrino + All.neff - 4.046);
#else
      roneu = neutrino_integration(a, 0., 0., -1) * All.NNeutrino;
#endif    
    }

  roneu *= All.rocr;
    
  rototal = rom + roneu + rolambda;
  hub = sqrt(8 * M_PI * Gr * rototal / 3) / clight;

  return hub;
}
#ifndef PHIFORMULA
void Table_phi_init(int num_bins)
{
  int b, i;
  double *Phiarray0;
  double A_interval = 1.00207;
  if(ThisTask == 0) printf("Phi Table Initializing...\n");
  Phiarray0 = (double *) malloc((num_bins) * sizeof(double));
  Fs.Phiarrayk = (double *) malloc((num_bins + 1) * sizeof(double));
  Fs.Phiarray  = (double **) malloc(All.NNeutrino * sizeof(double *));
  Fs.Phiarray_local = (double **) malloc(All.NNeutrino * sizeof(double *));
  for(i = 0; i < All.NNeutrino; i++)
    {
      Fs.Phiarray[i] = (double *) malloc((num_bins) * sizeof(double));
      Fs.Phiarray_local[i] = (double *) malloc((num_bins) * sizeof(double));
      for(b = 0; b < num_bins; b++)
        {
          Fs.Phiarray[i][b] = 0.;
          Fs.Phiarray_local[i][b] = 0.;
        }
    }
  
  /* Initialize the bin arrary. */
  Phiarray0[0] = 1.0e-3;
  for(b = 1; b < (num_bins + 1); b++)
      Phiarray0[b] = Phiarray0[b - 1] * A_interval;
  
  for(b = 0; b < num_bins; b++)
      Fs.Phiarrayk[b] = sqrt(Phiarray0[b] * Phiarray0[b]);
  if(ThisTask == 0) printf("Phi Task has been initialized successfully!\n");
}

void Table_phi_print(int num_bins)
{
  int i, b; 

  if(ThisTask == 0) 
    {
      FILE *output;
      char path[500];
      sprintf(path, "%s/table_phi.txt", All.OutputDir);
      output = fopen(path, "w");
      for(b = 0; b < num_bins; b++)
        for(i = 0; i < All.NNeutrino; i++)
          {
            fprintf(output, "%d\t %f \t %.20f\n", i, Fs.Phiarrayk[b], Fs.Phiarray[i][b]);
          }
      fclose(output);
    } 
}

/* the integrand of Phi(q) upper part. */
double phi_up_integrand(double x, void *para)
{
  double xi = *(double *) para;
  return (x / (exp(x - xi) + 1.) + x / (exp(x + xi) + 1.));
}

void Table_phi_generate(int num_bins)
{
  #define WORKSIZE3 100000
  int PercentCompleted, b, i;
  double tmp_up, tmp_dn, A, xi, tmp_abserr;
  PercentCompleted = num_bins / 10;
  /* Generate the table for each neutrino type. */
  for(i = 0; i < All.NNeutrino; i++)
    {
      tmp_dn = All.numdens[i];
      xi = All.xi[i];
      b = ThisTask;
      do
        {
          if(b % PercentCompleted == 0 )
              if(ThisTask == 0) 
                  printf("Neutrino Index [%d]: %d0 percent completed...\n", i, b/PercentCompleted);
          A = Fs.Phiarrayk[b];
          gsl_integration_qawo_table* wf = gsl_integration_qawo_table_alloc(A, 1.0, GSL_INTEG_SINE, WORKSIZE3);
          gsl_integration_qawo_table_set(wf, A, 1.0, GSL_INTEG_SINE);
          gsl_function F2;
          gsl_integration_workspace *workspace0;
          gsl_integration_workspace *workspace1;
          workspace1 = gsl_integration_workspace_alloc(WORKSIZE3);
          workspace0 = gsl_integration_workspace_alloc(WORKSIZE3);
          F2.function = &phi_up_integrand;
          F2.params   = &xi;
          gsl_integration_qawf(&F2, 0.0, 1.0e-10, WORKSIZE3, workspace1, workspace0, wf, &tmp_up, &tmp_abserr);
          gsl_integration_workspace_free(workspace0);
          gsl_integration_workspace_free(workspace1);
          gsl_integration_qawo_table_free(wf);
#ifdef STERILE
          if(i == STERILE) tmp_up *= (All.neff - 3.046);
#endif
          Fs.Phiarray_local[i][b] = 1e13 * tmp_up / (A * tmp_dn);
          b += NTask;
        }
      while(b < num_bins);
      MPI_Allreduce(Fs.Phiarray_local[i], Fs.Phiarray[i], num_bins, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }
  for(i = 0; i < All.NNeutrino; i++)
    {
      free(Fs.Phiarray_local[i]);
      Fs.Phiarray_local[i] = NULL;
    }
  free(Fs.Phiarray_local);
  Fs.Phiarray_local = NULL;
  Table_phi_print(num_bins);
}

void Table_phi_free()
{
  if(ThisTask == 0)
    {
      int i;
      if(Fs.Phiarrayk != NULL) free(Fs.Phiarrayk);
      for(i = 0; i < All.NNeutrino; i++)
        {
          if(Fs.Phiarray[i] != NULL) free(Fs.Phiarray[i]);
        }
      if(Fs.Phiarray != NULL) free(Fs.Phiarray);
    }
}

void Table_phi_search(int num_bins)
{
  FILE *input;
  char path[500];
  sprintf(path, "%stable_phi.txt", All.OutputDir);
  int size, index;
  double data, k;
  size = 0;
  
  if(!(input = fopen(path, "r")))
    {
      if(ThisTask == 0) printf("Can't read the table in file '%s'\n", path);
      if(ThisTask == 0) printf("Table generating...\n");
      Table_phi_generate(num_bins);
    }
  else
    {
      if(ThisTask == 0) printf("Reading table of Phi(q) from %s ...\n", path);
      do
        {
          if(fscanf(input, "%d %lg %lg", &index, &k, &data) == 3)
            {
              Fs.Phiarray[index][size] = data;
              if(index == (All.NNeutrino - 1)) size++;
            }
          else break;
        }
      while(1);
      fclose(input);
      if(ThisTask == 0) printf("Finish reading, checking...\n");
      if(size != num_bins)
        {
          printf("the size of table phi is %d, but we actually need %d, please check it.\n", size, num_bins);
          endrun(1);
        }
      if(ThisTask == 0) printf("Finish checking.\n");
    }
}
#endif

void Table_s_init(int num_bins)
{
  if(ThisTask == 0) printf("Table s and hubble paramter start initializing...\n");
  int i;
  Fs.loga = (double *) malloc((num_bins + 1) * sizeof(double));
  Fs.s = (double *) malloc((num_bins + 1) * sizeof(double));
  Fs.h = (double *) malloc((num_bins + 1) * sizeof(double));
  double k = (0 - log(All.TimeBegin)) / (double) num_bins;
  Fs.loga[0] = log(All.TimeBegin);
  for(i = 0; i <= num_bins; i++)
    {
      Fs.loga[i] = i * k + Fs.loga[0];
    }
  if(ThisTask == 0) printf("Table s and hubble parameter have been initialized successfully!\n");
}

void Table_s_print(int num_bins)
{
  int i, b; 

  if(ThisTask == 0) 
    {
      FILE *output;
      char path[500];
      sprintf(path, "%stable_s.txt", All.OutputDir);
      output = fopen(path, "w");
      for(b = 0; b <= num_bins; b++)
        {
          fprintf(output, "%f\t %.20f %.26f\n", Fs.loga[b], Fs.s[b], Fs.h[b]);
        }
      fclose(output);
    } 
}

double s_integrand(double loga)
{
  double a = exp(loga);
  return 1./(a*a*hubble(a));
}

double S_integration(double loga)
{
  double inte_result, inte_abserr;
  #define WORKSIZE3 100000
  gsl_function S;
  gsl_integration_workspace *workspace3;
  workspace3 = gsl_integration_workspace_alloc(WORKSIZE3);
  S.function = &s_integrand;
  gsl_integration_qag(&S, log(All.TimeBegin), loga, 0, 1.0e-8, WORKSIZE3, GSL_INTEG_GAUSS61, workspace3, &inte_result, &inte_abserr);
  gsl_integration_workspace_free(workspace3);
  return inte_result;
}

void Table_s_generate(int num_bins)
{
  if(ThisTask == 0) printf("S table generating: Starting...\n");
  int i;
  int PercentCompleted;
  double *s_local, *h_local;
  s_local = (double *) malloc((num_bins+1) * sizeof(double));
  h_local = (double *) malloc((num_bins+1) * sizeof(double));
  for(i = 0; i <= num_bins; i++)
    {
      s_local[i] = 0.;
      h_local[i] = 0.;
    }
  Fs.s[0] = 0;
  PercentCompleted = num_bins / 10;
  i = ThisTask + 1;
  do
    {
      if((ThisTask == 0) && ((i % PercentCompleted) == 0)) printf("%d0 percent completed...\n", i/PercentCompleted);
      s_local[i] = S_integration(Fs.loga[i])/1e15;
      h_local[i] = hubble(exp(Fs.loga[i]))*1e18;
      i += NTask;
    }
  while(i <= num_bins);
  MPI_Allreduce(s_local, Fs.s, num_bins+1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(h_local, Fs.h, num_bins+1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  Fs.h[0] = hubble(All.TimeBegin)*1e18;
  free(s_local);
  free(h_local);
  s_local= NULL;
  h_local = NULL;
  Table_s_print(num_bins);
  if(ThisTask == 0) printf("S table generating: Finished.\n");
}

void Table_s_search(int num_bins)
{
  FILE *input;
  char path[500];
  sprintf(path, "%stable_s.txt", All.OutputDir);
  double loga, s, h;
  int size = 0;
  
  if(!(input = fopen(path, "r")))
    {
      if(ThisTask == 0) printf("Can't read the table s in file '%s'\n", path);
      MPI_Barrier(MPI_COMM_WORLD);
      if(ThisTask == 0) printf("Table s generating...\n");
      Table_s_generate(num_bins);
      size = num_bins + 1;
    }
  else
    {
      if(ThisTask == 0) printf("Reading table of s from %s ...\n", path);
      do
        {
          if(fscanf(input, "%lg %lg %lg", &loga, &s, &h) == 3)
            {
              Fs.loga[size] = loga;
              Fs.s[size] = s;
              Fs.h[size] = h;
              //if(ThisTask == 0) printf("size=%d, a=%f, hubble=%f", size, Fs.loga[size], Fs.h[size]);
              size++;
            }
          else break;
        }
      while(1);
      fclose(input);
      if(ThisTask == 0) printf("Finish reading, checking...\n");
      if(size != (num_bins+1))
        {
          printf("the size of table s and hubble is %d, but we actually need %d, please check it.\n", size, (num_bins+1));
          endrun(1);
        }
      if(ThisTask == 0) printf("Finish checking.\n");
    }
  Fs.spline_s = gsl_spline_alloc(gsl_interp_cspline, size);
  Fs.spline_h = gsl_spline_alloc(gsl_interp_cspline, size);
  gsl_spline_init(Fs.spline_s, Fs.loga, Fs.s, size);
  gsl_spline_init(Fs.spline_h, Fs.loga, Fs.h, size);
}

void Table_s_free(void)
{ 
  if(ThisTask == 0)
    {
      free(Fs.loga);
      free(Fs.s);
      free(Fs.h);
      gsl_interp_accel_free(Fs.acc_s);
      gsl_interp_accel_free(Fs.acc_h);
      gsl_spline_free(Fs.spline_h);
      gsl_spline_free(Fs.spline_s);
    }
}

void Table_rho_init(int num_kbins)
{
  int i;
  Fs.rho_tot = (double **) malloc((num_kbins) * sizeof(double*));
  for(i = 0; i <= num_kbins; i++)
    {
      Fs.rho_tot[i] = (double *) malloc((5001) * sizeof(double));
    }
  Fs.loga_for_rho = (double *) malloc(5001 * sizeof(double));
  Fs.count = 0;
}

void Table_rho_free(void)
{
  if(ThisTask == 0)
    {
      int i;
      for(i = 0; i < Fs.num_kbins; i++)
        {
          if(Fs.rho_tot[i] != NULL) free(Fs.rho_tot[i]);
        }
      if(Fs.rho_tot != NULL) free(Fs.rho_tot);
      if(Fs.loga_for_rho != NULL) free(Fs.loga_for_rho);
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
     //   printf("xi2 %f\n", xi2);
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
    xi1 = L1 / (M_PI * M_PI);
    
    for(i=0;i<10;i++){
        xi1 = L1 / (xi1 * xi1 + M_PI * M_PI);
      //  printf("xi1 %f\n", xi1);
    }
    
    return xi1;
}

double neutrino_partition(double pot, void *par)
{
  struct neu_params * params = (struct neu_params *)par;
    double part;
  double Tneu = (params->Tnu);
  double mass_nu = (params->m);
    double nu_xi = (params->xi);
    double mass_n;
    
  mass_n = mass_nu/ktoev;
  part = pow(pot, 3)*sqrt(1 + pow(mass_n/(pot*Tneu), 2))/(exp((pot) - nu_xi) + 1.0) + pow(pot, 3)*sqrt(1 + pow(mass_n/(pot*Tneu), 2))/(exp((pot) + nu_xi) + 1.0);
    //printf("massn %f mass nu %f, ktoev %f part %f Tneu %f\n", mass_n, mass_nu, ktoev, part,Tneu);
  
  return part;
}

double neutrino_integration(double a, double m, double xi, int neutrinoindex)
{
  #define WORKSIZE2 100000
  gsl_function F2;
  gsl_integration_workspace *workspace2;
  
  double inte_result, inte_abserr;
  double roneu;
  double Tneu;
  Tneu = All.Tneu0/a;
  struct neu_params alpha = {Tneu, m, xi};
  
  
  workspace2 = gsl_integration_workspace_alloc(WORKSIZE2);
  F2.function = &neutrino_partition;
  F2.params = &alpha;
  
  gsl_integration_qagiu(&F2, 0.0, 0, 1.0e-8, WORKSIZE2, workspace2, &inte_result, &inte_abserr);
  roneu = inte_result * All.unittrans * pow(Tneu, 4);
  roneu = roneu / All.rocr;
  
  //printf("mass %f xi %f a %f roneu %f rocr %f unittrans %f inte_result %f\n", m, xi, a, roneu, All.rocr *1e15, All.unittrans*1e15, inte_result);
  
  gsl_integration_workspace_free(workspace2);
#ifdef STERILE
  if(neutrinoindex == STERILE)
    {
      roneu *= (All.neff - 3.046);
    }
#endif
  return roneu;
}

double numdens_integrand(double x, void *par){
    
    double xi = *(double *) par;
    return x * x /(exp(x - xi) + 1.) + x * x /(exp(x + xi) + 1.);
    
}

/* Calculate the neutrino number density. */
double numdens(double xi, int neutrinoindex){
#define WORKSIZE2 100000
  gsl_function F;
  gsl_integration_workspace *workspace2;
  
  double inte_result, inte_abserr;
  
  workspace2 = gsl_integration_workspace_alloc(WORKSIZE2);
  F.function = &numdens_integrand;
  F.params = &xi;
  
  gsl_integration_qagiu(&F, 0.0, 0, 1.0e-8, WORKSIZE2, workspace2, &inte_result, &inte_abserr);
  gsl_integration_workspace_free(workspace2);
  
#ifdef STERILE
  if(neutrinoindex == STERILE)
    {
      inte_result *= (All.neff - 3.046);
    }
#endif
  return inte_result;
    
}

#ifdef PHIFORMULA
double Phi_formula(double q)
{
  double tmp;
  if(q <= 0.)
    tmp = 1.;
  else
    tmp = (1.+0.0168*pow(q,2)+0.0407*pow(q,4))/(1.+2.1734*pow(q,2)+1.6787*exp(4.1811*log(q))+0.1467*pow(q,8));
  return tmp;
}
#endif

double free_streaming_integrand(double loga, void *param)
{
  double sf, s_int, phi_int, rho_int, k, h;
  int neutrinoindex;
  gsl_interp_accel *acc_rho_int;
  gsl_spline *spline_rho_int;
  struct int_params *int_para = (struct int_params *) param;
  k = int_para->k;
  neutrinoindex = int_para->neutrinoindex;
  acc_rho_int = int_para->acc_rho;
  spline_rho_int = int_para->spline_rho;
  sf = int_para->sf;
  s_int = gsl_spline_eval(Fs.spline_s, loga, Fs.acc_s);
  //if(ThisTask == 0) printf("loga=%f\t s_int=%f\t", loga, s_int);
  rho_int = gsl_spline_eval(spline_rho_int, loga, acc_rho_int);
  //if(ThisTask == 0) printf("rho_int=%f\t", rho_int);
  h = gsl_spline_eval(Fs.spline_h, loga, Fs.acc_h) ;
  //if(ThisTask == 0) printf("loga=%f\t h=%f\t", loga, h);
  double q = k*(sf-s_int)*1e15* 1.945 * ktoev * All.HubbleParam * clight / ( mpc_to_m * All.mass[neutrinoindex]);
#ifdef PHIFORMULA
  phi_int = Phi_formula(q) * 1e13·;
#else
  //if(ThisTask == 0) printf("q=%f\t", q);
  if(q <= Fs.Phiarrayk[0]) phi_int = 1e13;
  else if(q >= Fs.Phiarrayk[9999]) phi_int = 0.;
  else phi_int = gsl_spline_eval(Fs.spline_phi, q, Fs.acc_phi);
  //if(ThisTask == 0) printf("phi_int=%f\t", phi_int);
#endif

  /* s has been divided by 1e15, so here times 1e15, phi has been divied by 1e18 and h divied by 
   * 1e13 so here we times 1e5.
   */
  double results = (4 * M_PI * Gr / cc) * exp(loga) * exp(loga) * (sf - s_int) * 1e15 * rho_int * phi_int * 1e5 / h;
  //if(ThisTask == 0) printf("result=%f\n", results);
  return  results;
}

double frstr(double k, double delta_nu_0, int neutrinoindex, int kindex)
{
  //clock_t time0, time1;
  //time0 = clock();
  gsl_interp_accel *acc_rho = gsl_interp_accel_alloc();
  gsl_spline *spline_rho;
  if(Fs.count < 3) 
    {
      spline_rho  = gsl_spline_alloc(gsl_interp_linear, Fs.count);
    }
  else 
    {
      spline_rho  = gsl_spline_alloc(gsl_interp_cspline, Fs.count);
    }
  gsl_spline_init(spline_rho, Fs.loga_for_rho, Fs.rho_tot[kindex], Fs.count);
  double phi, sf, delta_nu = 0.;

  sf  = gsl_spline_eval(Fs.spline_s, log(All.Time), Fs.acc_s);
  //if(ThisTask == 0) printf("1st term:sf=%f\t", sf);
  //if(ThisTask == 0) printf("a = %g\t s = %g\n", All.Time, sf);
  double q = k*sf*1e15* 1.945 * All.HubbleParam * ktoev * clight / (mpc_to_m * All.mass[neutrinoindex]);
#ifdef PHIFORMULA
  phi = Phi_formula(q)*1e13;
#else
  if(q <= Fs.Phiarrayk[0]) phi = 1e13;
  else if (q >= Fs.Phiarrayk[9999]) phi = 0.;
  else phi = gsl_spline_eval(Fs.spline_phi, q, Fs.acc_phi);
#endif
  delta_nu += phi * delta_nu_0 * (1 + sf*1e15 * All.TimeBegin * All.TimeBegin * hubble(All.TimeBegin))  / 1e13;
  //if(ThisTask == 0) printf("k = %f\t q = %f\t phi = %f \t delta_nu_0 = %g\t delta_nu = %g\n", k, q, phi, delta_nu_0, delta_nu);
  //if(ThisTask == 0) printf("k = %f\t delta_nu_0 = %g\t delta_nu = %g\t", k, delta_nu_0, delta_nu);

  #define WORKSIZE4 100000
  gsl_function F;
  gsl_integration_workspace *workspace4;
  double inte_result, inte_abserr;
  inte_result = 0.;
  struct int_params beta = {k, neutrinoindex, spline_rho, acc_rho, sf};
  workspace4 = gsl_integration_workspace_alloc(WORKSIZE4);
  F.function = &free_streaming_integrand;
  F.params = &beta;
  gsl_integration_qag(&F, log(All.TimeBegin), log(All.Time), 0, 1.0e-5, WORKSIZE4, GSL_INTEG_GAUSS61, workspace4, &inte_result, &inte_abserr);
  gsl_integration_workspace_free(workspace4);
  //if(ThisTask == 0) printf("\n");
  delta_nu += inte_result;
  //if(ThisTask == 0) printf("inte_result = %g\t delta_nu=%g\n", inte_result, delta_nu);
  gsl_interp_accel_free(acc_rho);
  gsl_spline_free(spline_rho);

  return delta_nu;
}

void Neutrino_init()
{
  int i;
  All.H0 = 100. * All.HubbleParam * 1000. / mpc_to_m;
  All.rocr = (cc * 3.0 * All.H0 * All.H0) / (8.0*M_PI*Gr);
  All.unittrans = pow(kb, 4)/((pow(hbar, 3))*(pow(clight, 3))*2*M_PI*M_PI);
  if(ThisTask == 0) printf("H0 = %g \t rocr = %g \n", All.H0, All.rocr);
  switch(All.lepton_asymmetry)
    {
      case 0:
        All.xi[1] = All.xi[2];
        All.xi[0] = All.xi[2];
        break;

      case 1:
        All.xi[1] = cal_xi2(All.xi[2]);
        All.xi[0] = cal_xi1(All.xi[1], All.xi[2]);
        break;

      default:
        break;
    }
  
  switch(All.mass_hierarchy)
    {
      case 0:
        /* normal */
        All.mass[1] = sqrt(All.mass[0]*All.mass[0] + 7.59e-5);
        All.mass[2] = sqrt(All.mass[1]*All.mass[1] + 2.32e-3);
        break;
      
      case 1:
        /* inverted */
        All.mass[1] = sqrt(All.mass[0]*All.mass[0] + 2.32e-3);
        All.mass[2] = sqrt(All.mass[1]*All.mass[1] + 7.59e-5);
        break;
      
      case 2:
        /* identical */
        All.mass[1] = All.mass[0];
        All.mass[2] = All.mass[0];
        break;
      
      default:
        break;
    }
    
  if(All.expan_on == 1)
    {
      All.Omega_nu0_expan = 0.0;
	    for(i = 0; i < All.NNeutrino; i++){
	      All.Omega_nu0_expan += neutrino_integration(1.0, All.mass[i], All.xi[i], i);
	    }
    } 
  else 
    {
#ifdef STERILE
      All.Omega_nu0_expan = neutrino_integration(1.0, 0., 0., -1) * (All.NNeutrino + All.neff - 4.046);
#else
      All.Omega_nu0_expan = neutrino_integration(1.0, 0., 0., -1) * All.NNeutrino;
#endif
    }
    
#ifdef STERILE
  All.Omega_nu0_frstr = neutrino_integration(1.0, 0., 0., -1) * (All.NNeutrino + All.neff - 4.046);
#else
  All.Omega_nu0_frstr = neutrino_integration(1.0, 0., 0., -1) * All.NNeutrino;
#endif
    
  if(All.deductfromDE == 1)
    {
      All.Omega2 = All.Omega0;
      All.OmegaLambda = All.OmegaLambda - All.Omega_nu0_expan;
      All.Omega0 = All.Omega0 + All.Omega_nu0_expan;
    }
  else
    {
      All.Omega2 = All.Omega0 - All.Omega_nu0_expan;
      All.OmegaLambda = All.OmegaLambda;
    }

  All.numdens0 = numdens(0., -1);
  for(i = 0; i < All.NNeutrino; i++)
    {
	    All.numdens[i] = numdens(All.xi[i], i);
    }

  if(ThisTask == 0)
    {
      for(i = 0; i < All.NNeutrino; i++)
        {
          printf("m%d=%f ", i, All.mass[i]);
        }
    
      for(i = 0; i < All.NNeutrino; i++)
        {
	        printf("xi%d=%f ", i, All.xi[i]);
        }
      printf("\n");
      printf("Omega0 = %f Omega2 = %f Omeganuexpan %f Omeganufrstr %f\t", All.Omega0, All.Omega2, All.Omega_nu0_expan, All.Omega_nu0_frstr);
      printf("Omega_lambda = %f\n", All.OmegaLambda);
      printf("numdens %f ", All.numdens0);
      for(i = 0; i < All.NNeutrino; i++)
        {
	        printf("%f ", All.numdens[i]);
        }
        printf("\n");
    }
#ifndef PHIFORMULA
  Table_phi_init(10000);
  Table_phi_search(10000);
#endif
  Table_s_init(10000);
  Table_s_search(10000);
}

void Neutrino_free()
{
  Table_rho_free();
  Table_s_free();
#ifndef PHIFORMULA
  Table_phi_free();
#endif
}
