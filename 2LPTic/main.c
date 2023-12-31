#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <drfftw_mpi.h>
#include <mpi.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>

#include "allvars.h"
#include "proto.h"
#include "neutrino.h"

#define ASSERT_ALLOC(cond) {                                                                                  \
	if(cond)                                                                                                   \
    {                                                                                                         \
    	if(ThisTask == 0)                                                                                       \
			printf("\nallocated %g Mbyte on Task %d\n", bytes / (1024.0 * 1024.0), ThisTask);                     \
    }                                                                                                         \
  	else                                                                                                        \
    {                                                                                                         \
      	printf("failed to allocate %g Mbyte on Task %d\n", bytes / (1024.0 * 1024.0), ThisTask);                \
      	printf("bailing out.\n");                                                                               \
     	FatalError(1);                                                                                          \
    }                                                                                                         \
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  MPI_Comm_size(MPI_COMM_WORLD, &NTask);

  if(argc < 2)
    {
      	if(ThisTask == 0)
		{
	  		fprintf(stdout, "\nParameters are missing.\n");
	  		fprintf(stdout, "Call with <ParameterFile>\n\n");
		}
      	MPI_Finalize();
    	exit(0);
    }

  	read_parameterfile(argv[1]);

	int i;
    H0 = 100. * HubbleParam * 1000. / ((float)(1E6 * 3.0857E16));
    unittrans = pow(kb, 4)/((pow(hbar, 3))*(pow(c, 3))*2*M_PI*M_PI);
    rocr = (cc * 3.0 * H0 * H0) / (8.0*M_PI*Gr);

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
	if (ThisTask == 0)
	{
		printf("Here are the xis ");
		for (i = 0; i < NNeutrino; i++)
		{
			printf("xi%d %f ", i, Xi[i]);
		}
		printf("\n");
		printf("Here are the masses: Hierarchy %d ", mass_hierarchy);
		for (i = 0; i < NNeutrino; i++)
		{
			printf("m%d %f ", i, Mass[i]);
		}
		printf("\n");
	}
	if (expan_on == 0)
	{
#ifdef STERILE
		Omega_nu0_expan = neutrino_integration(1.0, 0.0, 0.0, -1) * (NNeutrino + Neff - 4.046);
#else
		Omega_nu0_expan = neutrino_integration(1.0, 0.0, 0.0, -1) * NNeutrino;
#endif
	}

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

    if(deductfromDE == 1){
        OmegaLambda = OmegaLambda - Omega_nu0_expan;
        Omega2 = Omega;
        Omega = Omega + Omega_nu0_expan;
    }
    
    if(deductfromDE == 0){
        OmegaLambda = OmegaLambda;
        Omega2 = Omega - Omega_nu0_expan;
        Omega = Omega;
    }
    
	if (ThisTask == 0)
	{
		printf("omega = %f omega2 = %f omega_lambda = %f Omega_nu expan %f Omega_nu frstr %f\n", Omega, Omega2, OmegaLambda, Omega_nu0_expan, Omega_nu0_frstr);
	}

    double a00 = 1 / (1 + Redshift);
    
    if (ThisTask == 0)
	{
		printf("inittime %f\n", a00);
	}
    D11 = growth(1.0);
	printf("%f\n", a00);
    D10 = growth(a00);
	if (ThisTask == 0)
	{
		printf("D11 %f D10 %f d11 %f d10 %f\n", D11, D10, growth(1.0), growth(a00));
	}
    
    double theta = 2.7250 / 2.7;
    D11 *= (1 + 2.5 * 1e4 * (Omega) * HubbleParam * HubbleParam * pow(theta, -4));
    D11 *= 2.5 * (Omega);
    
    /*ronu_init = neutrino_integration(a00, mass_nu_frstr, xi_frstr);
    
    ro_init = (Omega / (a00*a00*a00)) + ronu_init + OmegaLambda;
    Omegam_init = (Omega / (a00*a00*a00)) / ro_init;
    Omeganu_init = ronu_init / ro_init;*/
    
    D10 *= (1 + 2.5 * 1e4 * (Omega) * HubbleParam * HubbleParam * pow(theta, -4));
    D10 *= 2.5 * (Omega);
    //printf("m %f nu %f\n", Omega / (a00*a00*a00), ronu_init);
    if (ThisTask == 0) 
	{
		printf("At initial z, Omegam %f Omeganu %f, ronuinit%f  D10 %f D11 %f\n", Omegam_init, Omeganu_init, ronu_init, D10, D11);
	}
    
  	set_units();

  	initialize_powerspectrum();

  	initialize_ffts();

  	read_glass(GlassFile);

  	displacement_fields();

  	write_particle_data();

  	if(NumPart)
    	free(P);

  	free_ffts();


  	if(ThisTask == 0)
    {
      printf("\nIC's generated.\n\n");
      printf("Initial scale factor = %g\n", InitTime);
      printf("\n");
    }

  	MPI_Barrier(MPI_COMM_WORLD);
  	print_spec();

  	MPI_Finalize();		/* clean up & finalize MPI */
  	exit(0);
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



void displacement_fields(void)
{
  	MPI_Request request;
  	MPI_Status status;
  	gsl_rng *random_generator;
  	int i, j, k, ii, jj, kk, axes;
  	int n;
  	int sendTask, recvTask;
  	double fac, vel_prefac, vel_prefac2;
  	double kvec[3], kmag, kmag2, p_of_k;
  	double delta, phase, ampl, hubble_a;
  	double u, v, w;
  	double f1, f2, f3, f4, f5, f6, f7, f8;
  	double dis, dis2, maxdisp, max_disp_glob;
  	unsigned int *seedtable;

  	unsigned int bytes, nmesh3;
  	int coord;
  	fftw_complex *(cdisp[3]), *(cdisp2[3]) ; /* ZA and 2nd order displacements */
  	fftw_real *(disp[3]), *(disp2[3]) ;

  	fftw_complex *(cdigrad[6]);
  	fftw_real *(digrad[6]);

    double roneu;
    int count = 0;
    
#ifdef CORRECT_CIC
  	double fx, fy, fz, ff, smth;
#endif

  	if(ThisTask == 0)
    {
      	printf("\nstart computing displacement fields...\n");
      	fflush(stdout);
    }
    
	roneu =	0.0;
    if(expan_on == 0){
#ifdef STERILE
		roneu = neutrino_integration(InitTime, 0., 0., -1) * (NNeutrino + Neff - 4.046);
#else
        roneu = neutrino_integration(InitTime, 0., 0., -1) * NNeutrino;
#endif
	}
    if(expan_on == 1){
		for (i = 0; i < NNeutrino; i++)
		{
			roneu += neutrino_integration(InitTime, Mass[i], Xi[i], i);
		}
    }
    
    //hubble_a =
    //Hubble * sqrt(Omega2 / pow(InitTime, 3) + (1 - Omega2 - OmegaLambda - Omega_nu0_expan) / pow(InitTime, 2) + OmegaLambda + roneu);
    
    hubble_a = Hubble * sqrt(Omega / pow(InitTime, 3) + (1 - Omega - OmegaLambda - Omega_nu0_expan) / pow(InitTime, 2) + OmegaLambda + roneu);

  	vel_prefac = InitTime * hubble_a * F_Omega(InitTime);
  	vel_prefac2 = InitTime * hubble_a * F2_Omega(InitTime);

  	vel_prefac /= sqrt(InitTime);	/* converts to Gadget velocity */
  	vel_prefac2 /= sqrt(InitTime);

  	if(ThisTask == 0)
    	printf("vel_prefac= %g, vel_prefac2= %g,  hubble_a=%g fom=%g \n", vel_prefac, vel_prefac2, hubble_a, F_Omega(InitTime));

  	fac = pow(2 * PI / Box, 1.5);

  	maxdisp = 0;

  	random_generator = gsl_rng_alloc(gsl_rng_ranlxd1);

  	gsl_rng_set(random_generator, Seed);

  	if(!(seedtable = malloc(Nmesh * Nmesh * sizeof(unsigned int))))
    	FatalError(4);

  	for(i = 0; i < Nmesh / 2; i++)
    {
      	for(j = 0; j < i; j++)
			seedtable[i * Nmesh + j] = 0x7fffffff * gsl_rng_uniform(random_generator);

      	for(j = 0; j < i + 1; j++)
			seedtable[j * Nmesh + i] = 0x7fffffff * gsl_rng_uniform(random_generator);

      	for(j = 0; j < i; j++)
			seedtable[(Nmesh - 1 - i) * Nmesh + j] = 0x7fffffff * gsl_rng_uniform(random_generator);

      	for(j = 0; j < i + 1; j++)
			seedtable[(Nmesh - 1 - j) * Nmesh + i] = 0x7fffffff * gsl_rng_uniform(random_generator);

      	for(j = 0; j < i; j++)
			seedtable[i * Nmesh + (Nmesh - 1 - j)] = 0x7fffffff * gsl_rng_uniform(random_generator);

      	for(j = 0; j < i + 1; j++)
			seedtable[j * Nmesh + (Nmesh - 1 - i)] = 0x7fffffff * gsl_rng_uniform(random_generator);

      	for(j = 0; j < i; j++)
			seedtable[(Nmesh - 1 - i) * Nmesh + (Nmesh - 1 - j)] = 0x7fffffff * gsl_rng_uniform(random_generator);

      	for(j = 0; j < i + 1; j++)
			seedtable[(Nmesh - 1 - j) * Nmesh + (Nmesh - 1 - i)] = 0x7fffffff * gsl_rng_uniform(random_generator);
    }

	for(axes=0,bytes=0; axes < 3; axes++)
    {
      	cdisp[axes] = (fftw_complex *) malloc(bytes += sizeof(fftw_real) * TotalSizePlusAdditional);
      	disp[axes] = (fftw_real *) cdisp[axes];
    }

  	ASSERT_ALLOC(cdisp[0] && cdisp[1] && cdisp[2]);
  	printf("checkpoint 1");

#if defined(MULTICOMPONENTGLASSFILE) && defined(DIFFERENT_TRANSFER_FUNC)
  	for(Type = MinType; Type <= MaxType; Type++)
#endif
    {
      	if(ThisTask == 0)
		{
	  		printf("\nstarting axes=%d...\n", axes);
	  		fflush(stdout);
		}

      	/* first, clean the array */
     	for(i = 0; i < Local_nx; i++)
			for(j = 0; j < Nmesh; j++)
	 	 		for(k = 0; k <= Nmesh / 2; k++)
	    			for(axes = 0; axes < 3; axes++)
	      			{
						cdisp[axes][(i * Nmesh + j) * (Nmesh / 2 + 1) + k].re = 0;
						cdisp[axes][(i * Nmesh + j) * (Nmesh / 2 + 1) + k].im = 0;
	      			}

      	for(i = 0; i < Nmesh; i++)
		{
	  		ii = Nmesh - i;
	  		if(ii == Nmesh)
	   	 		ii = 0;
	  		if((i >= Local_x_start && i < (Local_x_start + Local_nx)) || (ii >= Local_x_start && ii < (Local_x_start + Local_nx)))
	    	{
	      		for(j = 0; j < Nmesh; j++)
				{
		  			gsl_rng_set(random_generator, seedtable[i * Nmesh + j]);
		  
		  			for(k = 0; k < Nmesh / 2; k++)
		    		{
		      			phase = gsl_rng_uniform(random_generator) * 2 * PI;
		      			do
							ampl = gsl_rng_uniform(random_generator);
		      			while(ampl == 0);
		      
		      			if(i == Nmesh / 2 || j == Nmesh / 2 || k == Nmesh / 2)
							continue;
		      			if(i == 0 && j == 0 && k == 0)
							continue;
		      
		      			if(i < Nmesh / 2)
							kvec[0] = i * 2 * PI / Box;
		      			else
							kvec[0] = -(Nmesh - i) * 2 * PI / Box;
		      
		      			if(j < Nmesh / 2)
							kvec[1] = j * 2 * PI / Box;
		      			else
							kvec[1] = -(Nmesh - j) * 2 * PI / Box;
		      
		      			if(k < Nmesh / 2)
							kvec[2] = k * 2 * PI / Box;
		      			else
							kvec[2] = -(Nmesh - k) * 2 * PI / Box;
		      
		      				kmag2 = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
		      				kmag = sqrt(kmag2);
		      
		      			if(SphereMode == 1)
						{
			  				if(kmag * Box / (2 * PI) > Nsample / 2)	/* select a sphere in k-space */
			    				continue;
						}
		      			else
						{
			  				if(fabs(kvec[0]) * Box / (2 * PI) > Nsample / 2)
			    				continue;
			  				if(fabs(kvec[1]) * Box / (2 * PI) > Nsample / 2)
			    				continue;
			  				if(fabs(kvec[2]) * Box / (2 * PI) > Nsample / 2)
			    				continue;
						}
		      
		      			//p_of_k = PowerSpec(kmag, ICfrstr);
                		p_of_k = PowerSpec(kmag);
		      
		      			p_of_k *= -log(ampl);
		      
		     			delta = fac * sqrt(p_of_k) / Dplus;	/* scale back to starting redshift */
                
                		count++;
                		//printf("dplus %f\n", Dplus);
                
                		if(ICfrstr == 1)
						{
                    		double kdependent_transfer, kdependent_transfer0;
                    		kdependent_transfer = 0.;
                    		kdependent_transfer0 = 0.;
                    		kdependent_transfer0 = growth_nu(1.0, kmag*1e3, Omega, Omega_nu0_frstr);
                    		kdependent_transfer = growth_nu(InitTime, kmag*1e3, Omega, Omega_nu0_frstr);

                    		count++;
                    		delta = fac * sqrt(p_of_k) * kdependent_transfer / kdependent_transfer0;
                    		//printf("perturb k %lf ampl %lf pk %lf delta %lf index %d scale %lf\n", kmag, ampl, p_of_k, delta, ICfrstr, kdependent_transfer / kdependent_transfer0);
                		}
	
                		if(ICfrstr == 4)
						{
                    		double kdependent_transfer, kdependent_transfer0;
                    		kdependent_transfer = 0.;
                    		kdependent_transfer0 = 0.;
                    		kdependent_transfer0 = growth_nu(1.0, kmag*1e3, Omega, Omega_nu0_frstr);
                   		 	//kdependent_transfer = growth_nu(InitTime, kmag*1e3, Omegam_init, Omeganu_init);
                    		kdependent_transfer = growth_nu(InitTime, kmag*1e3, Omega, Omega_nu0_frstr);
	
                    		delta = fac * sqrt(p_of_k) * kdependent_transfer / kdependent_transfer0;
                		}
                
                		if(ICfrstr == 5)
						{
                    		delta = fac * sqrt(p_of_k);
                		}
                
                		//printf("ampl = %f pofk0 = %f pofk0 = %f delta = %f ijk = %d %d %d \n", ampl, PowerSpec(kmag), p_of_k, delta, i, j, k);
                		if(k > 0)
						{
			  				if(i >= Local_x_start && i < (Local_x_start + Local_nx))
			    				for(axes = 0; axes < 3; axes++)
			      				{
									cdisp[axes][((i - Local_x_start) * Nmesh + j) * (Nmesh / 2 + 1) + k].re = -kvec[axes] / kmag2 * delta * sin(phase);
									cdisp[axes][((i - Local_x_start) * Nmesh + j) * (Nmesh / 2 + 1) + k].im = kvec[axes] / kmag2 * delta * cos(phase);
			      				}
						}
		      			else	/* k=0 plane needs special treatment */
						{
			  				if(i == 0)
			    			{
			      				if(j >= Nmesh / 2)
									continue;
			      				else
								{
				  					if(i >= Local_x_start && i < (Local_x_start + Local_nx))
				    				{
				      					jj = Nmesh - j;	/* note: j!=0 surely holds at this point */
				      
				      					for(axes = 0; axes < 3; axes++)
										{
					  						cdisp[axes][((i - Local_x_start) * Nmesh + j) * (Nmesh / 2 + 1) + k].re = -kvec[axes] / kmag2 * delta * sin(phase);
					  						cdisp[axes][((i - Local_x_start) * Nmesh + j) * (Nmesh / 2 + 1) + k].im = kvec[axes] / kmag2 * delta * cos(phase);
					  
					  						cdisp[axes][((i - Local_x_start) * Nmesh + jj) * (Nmesh / 2 + 1) + k].re = -kvec[axes] / kmag2 * delta * sin(phase);
					  						cdisp[axes][((i - Local_x_start) * Nmesh + jj) * (Nmesh / 2 + 1) + k].im = -kvec[axes] / kmag2 * delta * cos(phase);
										}
				    				}
								}
			    			}
			  				else	/* here comes i!=0 : conjugate can be on other processor! */
			    			{
			      				if(i >= Nmesh / 2)
									continue;
			      				else
								{
				 			 		ii = Nmesh - i;
				  					if(ii == Nmesh)
				    					ii = 0;
				  					jj = Nmesh - j;
				  					if(jj == Nmesh)
				    					jj = 0;

				  					if(i >= Local_x_start && i < (Local_x_start + Local_nx))
				    					for(axes = 0; axes < 3; axes++)
				      					{
											cdisp[axes][((i - Local_x_start) * Nmesh + j) * (Nmesh / 2 + 1) + k].re = -kvec[axes] / kmag2 * delta * sin(phase);
											cdisp[axes][((i - Local_x_start) * Nmesh + j) * (Nmesh / 2 + 1) + k].im = kvec[axes] / kmag2 * delta * cos(phase);
				      					}
				  
				  					if(ii >= Local_x_start && ii < (Local_x_start + Local_nx))
				    					for(axes = 0; axes < 3; axes++)
				      					{
											cdisp[axes][((ii - Local_x_start) * Nmesh + jj) * (Nmesh / 2 + 1) + k].re = -kvec[axes] / kmag2 * delta * sin(phase);
											cdisp[axes][((ii - Local_x_start) * Nmesh + jj) * (Nmesh / 2 + 1) + k].im = -kvec[axes] / kmag2 * delta * cos(phase);
				      					}
								}
			    			}
						}
		    		}
				}
	   		}
		}
      


      	/* At this point, cdisp[axes] contains the complex Zeldovich displacement */

       	if(ThisTask == 0) printf("Done Zeldovich.\n");
      
      	/* Compute displacement gradient */

      	for(i = 0; i < 6; i++)
		{
	  		cdigrad[i] = (fftw_complex *) malloc(bytes = sizeof(fftw_real) * TotalSizePlusAdditional);
	  		digrad[i] = (fftw_real *) cdigrad[i];
	  		ASSERT_ALLOC(cdigrad[i]);
		}
      
      	for(i = 0; i < Local_nx; i++)
			for(j = 0; j < Nmesh; j++)
	  			for(k = 0; k <= Nmesh / 2; k++)
	    		{
	      			coord = (i * Nmesh + j) * (Nmesh / 2 + 1) + k;
	      			if((i + Local_x_start) < Nmesh / 2)
						kvec[0] = (i + Local_x_start) * 2 * PI / Box;
	      			else
						kvec[0] = -(Nmesh - (i + Local_x_start)) * 2 * PI / Box;
	      
	      			if(j < Nmesh / 2)
						kvec[1] = j * 2 * PI / Box;
	      			else
						kvec[1] = -(Nmesh - j) * 2 * PI / Box;
	      
	     		 	if(k < Nmesh / 2)
						kvec[2] = k * 2 * PI / Box;
	     			else
						kvec[2] = -(Nmesh - k) * 2 * PI / Box;
	      
	      			/* Derivatives of ZA displacement  */
	      			/* d(dis_i)/d(q_j)  -> sqrt(-1) k_j dis_i */
	      			cdigrad[0][coord].re = -cdisp[0][coord].im * kvec[0]; /* disp0,0 */
	      			cdigrad[0][coord].im = cdisp[0][coord].re * kvec[0];

	      			cdigrad[1][coord].re = -cdisp[0][coord].im * kvec[1]; /* disp0,1 */
	      			cdigrad[1][coord].im = cdisp[0][coord].re * kvec[1];

	      			cdigrad[2][coord].re = -cdisp[0][coord].im * kvec[2]; /* disp0,2 */
	      			cdigrad[2][coord].im = cdisp[0][coord].re * kvec[2];
	      
	      			cdigrad[3][coord].re = -cdisp[1][coord].im * kvec[1]; /* disp1,1 */
	      			cdigrad[3][coord].im = cdisp[1][coord].re * kvec[1];

	      			cdigrad[4][coord].re = -cdisp[1][coord].im * kvec[2]; /* disp1,2 */
	      			cdigrad[4][coord].im = cdisp[1][coord].re * kvec[2];

	      			cdigrad[5][coord].re = -cdisp[2][coord].im * kvec[2]; /* disp2,2 */
	     		 	cdigrad[5][coord].im = cdisp[2][coord].re * kvec[2];
	    		}


      	if(ThisTask == 0) printf("Fourier transforming displacement gradient...");
      		for(i = 0; i < 6; i++) rfftwnd_mpi(Inverse_plan, 1, digrad[i], Workspace, FFTW_NORMAL_ORDER);
      			if(ThisTask == 0) printf("Done.\n");

      	/* Compute second order source and store it in digrad[3]*/

      	for(i = 0; i < Local_nx; i++)
			for(j = 0; j < Nmesh; j++)
	 			 for(k = 0; k < Nmesh; k++)
	    		{
	      			coord = (i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k;

	      			digrad[3][coord] = digrad[0][coord]*(digrad[3][coord]+digrad[5][coord])+digrad[3][coord]*digrad[5][coord]
                						-digrad[1][coord]*digrad[1][coord]-digrad[2][coord]*digrad[2][coord]-digrad[4][coord]*digrad[4][coord];
	    		}

      	if(ThisTask == 0) printf("Fourier transforming second order source...");
      		rfftwnd_mpi(Forward_plan, 1, digrad[3], Workspace, FFTW_NORMAL_ORDER);
      	if(ThisTask == 0) printf("Done.\n");
      
      	/* The memory allocated for cdigrad[0], [1], and [2] will be used for 2nd order displacements */
      	/* Freeing the rest. cdigrad[3] still has 2nd order displacement source, free later */

     	for(axes = 0; axes < 3; axes++) 
		{
	  		cdisp2[axes] = cdigrad[axes]; 
	  		disp2[axes] = (fftw_real *) cdisp2[axes];
		}

      	free(cdigrad[4]); free(cdigrad[5]); 

      	/* Solve Poisson eq. and calculate 2nd order displacements */

      	for(i = 0; i < Local_nx; i++)
			for(j = 0; j < Nmesh; j++)
	  			for(k = 0; k <= Nmesh / 2; k++)
	    		{
	      			coord = (i * Nmesh + j) * (Nmesh / 2 + 1) + k;
	      			if((i + Local_x_start) < Nmesh / 2)
						kvec[0] = (i + Local_x_start) * 2 * PI / Box;
	     			else
						kvec[0] = -(Nmesh - (i + Local_x_start)) * 2 * PI / Box;
	      
	      			if(j < Nmesh / 2)
						kvec[1] = j * 2 * PI / Box;
	     	 		else
						kvec[1] = -(Nmesh - j) * 2 * PI / Box;
	      
	      			if(k < Nmesh / 2)
						kvec[2] = k * 2 * PI / Box;
	      			else
						kvec[2] = -(Nmesh - k) * 2 * PI / Box;

	      				kmag2 = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
#ifdef CORRECT_CIC
	      			/* calculate smooth factor for deconvolution of CIC interpolation */
	      			fx = fy = fz = 1;
	      			if(kvec[0] != 0)
					{
		  				fx = (kvec[0] * Box / 2) / Nmesh;
		  				fx = sin(fx) / fx;
					}
	      			if(kvec[1] != 0)
					{
		  				fy = (kvec[1] * Box / 2) / Nmesh;
		  				fy = sin(fy) / fy;
					}
	      			if(kvec[2] != 0)
					{
		  				fz = (kvec[2] * Box / 2) / Nmesh;
		  				fz = sin(fz) / fz;
					}
	      			ff = 1 / (fx * fy * fz);
	      			smth = ff * ff;
	      			/*  */
#endif

	    /* cdisp2 = source * k / (sqrt(-1) k^2) */
	    for(axes = 0; axes < 3; axes++)
		{
		 	if(kmag2 > 0.0) 
		    {
		      	cdisp2[axes][coord].re = cdigrad[3][coord].im * kvec[axes] / kmag2;
		      	cdisp2[axes][coord].im = -cdigrad[3][coord].re * kvec[axes] / kmag2;
		    }
		  	else 
			  	cdisp2[axes][coord].re = cdisp2[axes][coord].im = 0.0;
#ifdef CORRECT_CIC
		  	cdisp[axes][coord].re *= smth;   cdisp[axes][coord].im *= smth;
		  	cdisp2[axes][coord].re *= smth;  cdisp2[axes][coord].im *= smth;
#endif
		}
	}
      
      	/* Free cdigrad[3] */
      	free(cdigrad[3]);

      
      	/* Now, both cdisp, and cdisp2 have the ZA and 2nd order displacements */

      	for(axes = 0; axes < 3; axes++)
		{
          	if(ThisTask == 0) printf("Fourier transforming displacements, axis %d.\n",axes);

	  		rfftwnd_mpi(Inverse_plan, 1, disp[axes], Workspace, FFTW_NORMAL_ORDER);
	  		rfftwnd_mpi(Inverse_plan, 1, disp2[axes], Workspace, FFTW_NORMAL_ORDER);

	  		/* now get the plane on the right side from neighbour on the right, 
	     	and send the left plane */
      
	 		 recvTask = ThisTask;
	  		do
	    	{
	      		recvTask--;
	      		if(recvTask < 0)
					recvTask = NTask - 1;
	    	}
	  		while(Local_nx_table[recvTask] == 0);
      
	  		sendTask = ThisTask;
	  		do
	    	{
	    		sendTask++;
	      		if(sendTask >= NTask)
					sendTask = 0;
	    	}
	  		while(Local_nx_table[sendTask] == 0);
      
	  		/* use non-blocking send */
      
	  		if(Local_nx > 0)
	    	{
	      		/* send ZA disp */
	      		MPI_Isend(&(disp[axes][0]),
				sizeof(fftw_real) * Nmesh * (2 * (Nmesh / 2 + 1)),
				MPI_BYTE, recvTask, 10, MPI_COMM_WORLD, &request);
	      
	     		MPI_Recv(&(disp[axes][(Local_nx * Nmesh) * (2 * (Nmesh / 2 + 1))]),
		       	sizeof(fftw_real) * Nmesh * (2 * (Nmesh / 2 + 1)),
		       	MPI_BYTE, sendTask, 10, MPI_COMM_WORLD, &status);
	      
	      		MPI_Wait(&request, &status);

	      		/* send 2nd order disp */
	     		MPI_Isend(&(disp2[axes][0]),
				sizeof(fftw_real) * Nmesh * (2 * (Nmesh / 2 + 1)),
				MPI_BYTE, recvTask, 10, MPI_COMM_WORLD, &request);
	      
	      		MPI_Recv(&(disp2[axes][(Local_nx * Nmesh) * (2 * (Nmesh / 2 + 1))]),
		       	sizeof(fftw_real) * Nmesh * (2 * (Nmesh / 2 + 1)),
		       	MPI_BYTE, sendTask, 10, MPI_COMM_WORLD, &status);
	      
	      		MPI_Wait(&request, &status);
	    	}
		}
      
      	/* read-out displacements */

	  	nmesh3 = ((unsigned int ) Nmesh ) * ((unsigned int) Nmesh) *  ((unsigned int) Nmesh);
      
      	for(n = 0; n < NumPart; n++)
		{
#if defined(MULTICOMPONENTGLASSFILE) && defined(DIFFERENT_TRANSFER_FUNC)
	  		if(P[n].Type == Type)
#endif
	    	{
	      		u = P[n].Pos[0] / Box * Nmesh;
	      		v = P[n].Pos[1] / Box * Nmesh;
	     	 	w = P[n].Pos[2] / Box * Nmesh;
	      
	      		i = (int) u;
	      		j = (int) v;
	      		k = (int) w;
	      
	      		if(i == (Local_x_start + Local_nx))
					i = (Local_x_start + Local_nx) - 1;
	      		if(i < Local_x_start)
					i = Local_x_start;
	      		if(j == Nmesh)
					j = Nmesh - 1;
	      		if(k == Nmesh)
					k = Nmesh - 1;
	      
	      		u -= i;
	      		v -= j;
	      		w -= k;
	      
	      		i -= Local_x_start;
	      		ii = i + 1;
	      		jj = j + 1;
	      		kk = k + 1;
	      
	      		if(jj >= Nmesh)
					jj -= Nmesh;
	      		if(kk >= Nmesh)
					kk -= Nmesh;
	      
	      		f1 = (1 - u) * (1 - v) * (1 - w);
	      		f2 = (1 - u) * (1 - v) * (w);
	      		f3 = (1 - u) * (v) * (1 - w);
	      		f4 = (1 - u) * (v) * (w);
	      		f5 = (u) * (1 - v) * (1 - w);
	      		f6 = (u) * (1 - v) * (w); 
	      		f7 = (u) * (v) * (1 - w);
	      		f8 = (u) * (v) * (w);
	     
	      		for(axes = 0; axes < 3; axes++)
				{
		  			dis = disp[axes][(i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k] * f1 +
		    		disp[axes][(i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + kk] * f2 +
		    		disp[axes][(i * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + k] * f3 +
		    		disp[axes][(i * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + kk] * f4 +
		    		disp[axes][(ii * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k] * f5 +
		    		disp[axes][(ii * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + kk] * f6 +
		    		disp[axes][(ii * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + k] * f7 +
		    		disp[axes][(ii * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + kk] * f8;

		  			dis2 = disp2[axes][(i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k] * f1 +
					disp2[axes][(i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + kk] * f2 +
		    		disp2[axes][(i * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + k] * f3 +
		    		disp2[axes][(i * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + kk] * f4 +
		    		disp2[axes][(ii * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k] * f5 +
		   	 		disp2[axes][(ii * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + kk] * f6 +
		    		disp2[axes][(ii * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + k] * f7 +
		    		disp2[axes][(ii * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + kk] * f8;
		  			dis2 /= (float) nmesh3;
#ifdef ONLY_ZA
		  			P[n].Pos[axes] += dis;
		  			P[n].Vel[axes] = dis * vel_prefac;
#else
		  			P[n].Pos[axes] += dis - 3./7. * dis2;
		  			P[n].Vel[axes] = dis * vel_prefac - 3./7. * dis2 * vel_prefac2;
#endif

		  			P[n].Pos[axes] = periodic_wrap(P[n].Pos[axes]);

		  			if(dis - 3./7. * dis2 > maxdisp)
		    		maxdisp = dis;
				}
	    	}
		}
    }
  

  	for(axes = 0; axes < 3; axes++) free(cdisp[axes]);
  	for(axes = 0; axes < 3; axes++) free(cdisp2[axes]);

 	gsl_rng_free(random_generator);

  	MPI_Reduce(&maxdisp, &max_disp_glob, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  	if(ThisTask == 0)
    {
      	printf("\nMaximum displacement: %g kpc/h, in units of the part-spacing= %g\n", max_disp_glob, max_disp_glob / (Box / Nmesh));
    }
}



double periodic_wrap(double x)
{
  while(x >= Box)
    x -= Box;

  while(x < 0)
    x += Box;

  return x;
}


void set_units(void)		/* ... set some units */
{
  UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s;

  G = GRAVITY / pow(UnitLength_in_cm, 3) * UnitMass_in_g * pow(UnitTime_in_s, 2);
  Hubble = HUBBLE * UnitTime_in_s;
}



void initialize_ffts(void)
{
  int total_size, i, additional;
  int local_ny_after_transpose, local_y_start_after_transpose;
  int *slab_to_task_local;
  size_t bytes;


  Inverse_plan = rfftw3d_mpi_create_plan(MPI_COMM_WORLD,
					 Nmesh, Nmesh, Nmesh, FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE);

  Forward_plan = rfftw3d_mpi_create_plan(MPI_COMM_WORLD,
					 Nmesh, Nmesh, Nmesh, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE);

  rfftwnd_mpi_local_sizes(Forward_plan, &Local_nx, &Local_x_start,
			  &local_ny_after_transpose, &local_y_start_after_transpose, &total_size);

  Local_nx_table = malloc(sizeof(int) * NTask);
  MPI_Allgather(&Local_nx, 1, MPI_INT, Local_nx_table, 1, MPI_INT, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      for(i = 0; i < NTask; i++)
	printf("Task=%d Local_nx=%d\n", i, Local_nx_table[i]);
      fflush(stdout);
    }


  Slab_to_task = malloc(sizeof(int) * Nmesh);
  slab_to_task_local = malloc(sizeof(int) * Nmesh);

  for(i = 0; i < Nmesh; i++)
    slab_to_task_local[i] = 0;

  for(i = 0; i < Local_nx; i++)
    slab_to_task_local[Local_x_start + i] = ThisTask;

  MPI_Allreduce(slab_to_task_local, Slab_to_task, Nmesh, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  free(slab_to_task_local);



  additional = (Nmesh) * (2 * (Nmesh / 2 + 1));	/* additional plane on the right side */

  TotalSizePlusAdditional = total_size + additional;

  //Disp = (fftw_real *) malloc(bytes = sizeof(fftw_real) * (total_size + additional));

  Workspace = (fftw_real *) malloc(bytes = sizeof(fftw_real) * total_size);

  ASSERT_ALLOC(Workspace)

  //Cdata = (fftw_complex *) Disp;	/* transformed array */
}



void free_ffts(void)
{
  free(Workspace);
  //free(Disp);
  free(Slab_to_task);
  rfftwnd_mpi_destroy_plan(Inverse_plan);
  rfftwnd_mpi_destroy_plan(Forward_plan);
}


int FatalError(int errnum)
{
  printf("FatalError called with number=%d\n", errnum);
  fflush(stdout);
  MPI_Abort(MPI_COMM_WORLD, errnum);
  exit(0);
}




static double A, B, alpha, beta, V, gf;

double fnl(double x)		/* Peacock & Dodds formula */
{
  return x * pow((1 + B * beta * x + pow(A * x, alpha * beta)) /
		 (1 + pow(pow(A * x, alpha) * gf * gf * gf / (V * sqrt(x)), beta)), 1 / beta);
}

void print_spec(void)
{
  	double k, knl, po, dl, dnl, neff, kf, kstart, kend, po2, po1, DDD;
  	char buf[1000];
  	FILE *fd;

  	if(ThisTask == 0)
    {
      	sprintf(buf, "%s/inputspec_%s.txt", OutputDir, FileBase);

      	fd = fopen(buf, "w");

      	gf = GrowthFactor(0.001, 1.0) / (1.0 / 0.001);

      	DDD = GrowthFactor(1.0 / (Redshift + 1), 1.0);

      	fprintf(fd, "%12g %12g\n", Redshift, DDD);	/* print actual starting redshift and linear growth factor for this cosmology */

      	kstart = 2 * PI / (1000.0 * (3.085678e24 / UnitLength_in_cm));	/* 1000 Mpc/h */
      	kend = 2 * PI / (0.001 * (3.085678e24 / UnitLength_in_cm));	/* 0.001 Mpc/h */

      	for(k = kstart; k < kend; k *= 1.025)
		{
	  		po = PowerSpec(k);
	  		dl = 4.0 * PI * k * k * k * po;

	  		kf = 0.5;

	  		po2 = PowerSpec(1.001 * k * kf);
	  		po1 = PowerSpec(k * kf);

	  		if(po != 0 && po1 != 0 && po2 != 0)
	    	{
	      		neff = (log(po2) - log(po1)) / (log(1.001 * k * kf) - log(k * kf));

	      		if(1 + neff / 3 > 0)
				{
		  			A = 0.482 * pow(1 + neff / 3, -0.947);
		  			B = 0.226 * pow(1 + neff / 3, -1.778);
		  			alpha = 3.310 * pow(1 + neff / 3, -0.244);
		  			beta = 0.862 * pow(1 + neff / 3, -0.287);
		  			V = 11.55 * pow(1 + neff / 3, -0.423) * 1.2;

		  			dnl = fnl(dl);
		  			knl = k * pow(1 + dnl, 1.0 / 3);
				}
	     		else
				{
		  			dnl = 0;
		  			knl = 0;
				}
	    	}
	  		else
	    	{
	      		dnl = 0;
	      		knl = 0;
	    	}

	  		fprintf(fd, "%12g %12g    %12g %12g\n", k, dl, knl, dnl);
		}
      	fclose(fd);
    }
}
