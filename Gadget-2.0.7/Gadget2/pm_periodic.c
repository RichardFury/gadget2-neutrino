#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <mpi.h>
#include <time.h>

/*! \file pm_periodic.c
 *  \brief routines for periodic PM-force computation
 */

#ifdef PMGRID
#ifdef PERIODIC

#ifdef NOTYPEPREFIX_FFTW
#include        <rfftw_mpi.h>
#else
#ifdef DOUBLEPRECISION_FFTW
#include     <drfftw_mpi.h>	/* double precision FFTW */
#else
#include     <srfftw_mpi.h>
#endif
#endif


#include "allvars.h"
#include "proto.h"
#include "fsvars.h"
#include "neutrino.h"

#define  PMGRID2 (2*(PMGRID/2 + 1))

static rfftwnd_mpi_plan fft_forward_plan, fft_inverse_plan;

static int slab_to_task[PMGRID];
static int *slabs_per_task;
static int *first_slab_of_task;
static int *meshmin_list, *meshmax_list;

static int slabstart_x, nslab_x, slabstart_y, nslab_y, smallest_slab;

static int fftsize, maxfftsize;

static double k_interval = 1.005;
/* neutrino power spectrum correction */
static fftw_real *rhogrid, *forcegrid, *workspace, *workspace1, *deltagrid;
static fftw_complex *fft_of_rhogrid, *fft_of_deltagrid;

static double *delta_cb, *delta_nu, *delta_nu_0;
static double *rd_array_k, *rd_array_pk;
static double *output_time_array;

static double *count_b, *count_b_local;
static double *k_array, *k_array0;

static int num_kbins;
static int rd_size_count;
static int output_time_size;

static FLOAT to_slab_fac;

void find_output_time()
{
	FILE *outtimetxt;
  double time_a;
  int output_no, i;
  if(!(outtimetxt = fopen(All.OutputListFilename, "r")))
    {
      printf("can't read input spectrum in file '%s' on task %d\n", All.OutputListFilename, ThisTask);
    }

  do
    {
      if(fscanf(outtimetxt, "%lg", &time_a) == 1)
        {
          output_no++;
        }
      else
        break;
    }
  while(1);
  fclose(outtimetxt);
	output_time_array = (double*) malloc((output_no) * sizeof(double));
	output_no = 0;
  if(!(outtimetxt = fopen(All.OutputListFilename, "r")))
    {
      printf("can't read input spectrum in file '%s' on task %d\n", All.OutputListFilename, ThisTask);
    }

  do
    {
      if(fscanf(outtimetxt, "%lg", &time_a) == 1)
        {
          output_time_array[output_no] = time_a;
          output_no++;
        }
      else
        break;
    }
  while(1);
  fclose(outtimetxt);

	output_time_size = output_no;
  if(ThisTask == 0)
    {
      printf("finished output time array reading \n");
      for(i = 0; i < output_time_size; i++)
        {
          printf("Time No.%03d: %f\n", i, output_time_array[i]);
        }
    }
}


int read_ratio()
{
  FILE *input;
  double k, p;
  int rd_size;
  rd_size = 0;
  if(ThisTask == 0) printf("Ratio array reading: Starting... \n");

  if(!(input = fopen(All.ratio_nu_cdm_txt, "r")))
    {
      printf("Error: Can't read the input spectrum in file '%s' on task %d\n", All.ratio_nu_cdm_txt, ThisTask);
    }
  do
    {
      if(fscanf(input, "%lg %lg", &k, &p) == 2)
        {
          rd_size++;
        }
      else
        break;
    }
  while(1);

  rd_array_k = (double *) malloc((rd_size + 1) * sizeof(double));
  rd_array_pk = (double *) malloc((rd_size + 1) * sizeof(double));
  rd_size = 0;

  if(!(input = fopen(All.ratio_nu_cdm_txt, "r")))
    {
      printf("Error: Can't read the input spectrum in file '%s' on task %d\n", All.ratio_nu_cdm_txt, ThisTask);
    }
  do
    {
      if(fscanf(input, "%lg %lg", &k, &p) == 2)
        {
          rd_array_k[rd_size] = k;
          rd_array_pk[rd_size] = p;
          rd_size++;
        }
      else
        break;
    }
  while(1);
  if(ThisTask == 0) printf("Ratio array reading: Finished.\n");
  return(rd_size);
}

void output_ratio(int num_kbins, int OutputIndex)
{
  FILE *output;
  char output_txt[300];
  int oi, j;
  sprintf(output_txt, "%s_%03d", All.nu_pk_txt, OutputIndex);
  output = fopen(output_txt, "w");
  printf("Nu_pk.txt No.%03d printing: Starting...\n", OutputIndex);

  double nu_temp;
  for(oi = 0; oi < num_kbins; oi++)
    {
      if(delta_cb[oi] > 1e-7)
        {
          nu_temp = 0;
          for(j = 0; j < All.NNeutrino; j++)
            {
              //printf("Fs.Omega_nu_temp[%d] = %g\t delta_nu = %g\n", j, Fs.Omega_nu_temp[j], delta_nu[oi*All.NNeutrino+j]);
              nu_temp += Fs.Omega_nu_temp[j] * delta_nu[oi*All.NNeutrino+j] / Fs.Omega_nu_temp_total;
            }
          fprintf(output, "%f\t %g\n", k_array[oi], (nu_temp * nu_temp) / (delta_cb[oi] * delta_cb[oi]));
          //printf("%f\t %g\t %g\t %g\n", k_array[oi], Fs.Omega_nu_temp_total, nu_temp, (delta_cb[oi] * delta_cb[oi]));
        }
    }
  fclose(output);
  printf("Nu_pk.txt No.%03d printing: Finished.\n", OutputIndex);
}

/*! This routines generates the FFTW-plans to carry out the parallel FFTs
 *  later on. Some auxiliary variables are also initialized.
 */
void pm_init_periodic(void)
{
  int i;
  int slab_to_task_local[PMGRID];

  All.Asmth[0] = ASMTH * All.BoxSize / PMGRID;
  All.Rcut[0] = RCUT * All.Asmth[0];

  /* Set up the FFTW plan files. */

  fft_forward_plan = rfftw3d_mpi_create_plan(MPI_COMM_WORLD, PMGRID, PMGRID, PMGRID,
					     FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE | FFTW_IN_PLACE);
  fft_inverse_plan = rfftw3d_mpi_create_plan(MPI_COMM_WORLD, PMGRID, PMGRID, PMGRID,
					     FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE | FFTW_IN_PLACE);

  /* Workspace out the ranges on each processor. */

  rfftwnd_mpi_local_sizes(fft_forward_plan, &nslab_x, &slabstart_x, &nslab_y, &slabstart_y, &fftsize);
    printf("nslab_x %d slabstart_x %d nslab_y %d slabstart_y %d fftsize %d maxfftsize %d\n", nslab_x, slabstart_x, nslab_y, slabstart_y, fftsize, maxfftsize);
  for(i = 0; i < PMGRID; i++)
    slab_to_task_local[i] = 0;

  for(i = 0; i < nslab_x; i++)
    slab_to_task_local[slabstart_x + i] = ThisTask;

  MPI_Allreduce(slab_to_task_local, slab_to_task, PMGRID, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  MPI_Allreduce(&nslab_x, &smallest_slab, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  slabs_per_task = malloc(NTask * sizeof(int));
  MPI_Allgather(&nslab_x, 1, MPI_INT, slabs_per_task, 1, MPI_INT, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      for(i = 0; i < NTask; i++)
	      printf("Task=%d  FFT-Slabs=%d\n", i, slabs_per_task[i]);
    }

  first_slab_of_task = malloc(NTask * sizeof(int));
  MPI_Allgather(&slabstart_x, 1, MPI_INT, first_slab_of_task, 1, MPI_INT, MPI_COMM_WORLD);

  meshmin_list = malloc(3 * NTask * sizeof(int));
  meshmax_list = malloc(3 * NTask * sizeof(int));


  to_slab_fac = PMGRID / All.BoxSize;

  MPI_Allreduce(&fftsize, &maxfftsize, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
}

/* initialize the neutrino stuff. */
void delta_init()
{
	Fs.num_kbins = num_kbins;
	Fs.fnu = (double*) malloc((All.NNeutrino)*sizeof(double));
  Fs.Omega_nu_temp = (double*) malloc((All.NNeutrino)*sizeof(double));
  delta_cb  = (double *) malloc((num_kbins) * sizeof(double));
  delta_nu  = (double *) malloc((num_kbins * All.NNeutrino) * sizeof(double));
  delta_nu_0= (double *) malloc((num_kbins * All.NNeutrino) * sizeof(double));
  count_b   = (double *) malloc((num_kbins) * sizeof(double));
  count_b_local = (double *) malloc((num_kbins) * sizeof(double));
  k_array0  = (double *) malloc((num_kbins+1) * sizeof(double));
  k_array   = (double *) malloc((num_kbins) * sizeof(double));
  Fs.acc_s    = gsl_interp_accel_alloc();
  Fs.acc_h    = gsl_interp_accel_alloc();
  int i, b;
  double start_k = 2. * M_PI * 0.95 / (All. BoxSize / 1e3);
  k_array0[0] = start_k;
  for(b = 1; b < (num_kbins+1); b++)
    {
      k_array0[b] = k_array0[b-1] * k_interval;
    }
  for(b = 0; b < num_kbins; b++)
    {
      delta_cb[b] = 0.;
      for(i = 0; i < All.NNeutrino; i++)
        {
          delta_nu[b*All.NNeutrino+i] = 0.;
          delta_nu_0[b*All.NNeutrino+i] = 0.;
        }
			k_array[b] = sqrt(k_array0[b] * k_array0[b+1]);
    }
	rd_size_count = read_ratio();
}

/*! This function allocates the memory neeed to compute the long-range PM
 *  force. Three fields are used, one to hold the density (and its FFT, and
 *  then the real-space potential), one to hold the force field obtained by
 *  finite differencing, and finally a workspace field, which is used both as
 *  workspace for the parallel FFT, and as buffer for the communication
 *  algorithm used in the force computation.
 */
void pm_init_periodic_allocate(int dimprod)
{
  static int first_alloc = 1;
  int dimprodmax;
  double bytes_tot = 0;
  size_t bytes;

  MPI_Allreduce(&dimprod, &dimprodmax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  /* allocate the memory to hold the FFT fields */

  if(!(rhogrid = (fftw_real *) malloc(bytes = fftsize * sizeof(fftw_real))))
    {
      printf("failed to allocate memory for `FFT-rhogrid' (%g MB).\n", bytes / (1024.0 * 1024.0));
      endrun(1);
    }
  bytes_tot += bytes;


  if(!(forcegrid = (fftw_real *) malloc(bytes = imax(fftsize, dimprodmax) * sizeof(fftw_real))))
    {
      printf("failed to allocate memory for `FFT-forcegrid' (%g MB).\n", bytes / (1024.0 * 1024.0));
      endrun(1);
    }
  bytes_tot += bytes;

  if(!(workspace = (fftw_real *) malloc(bytes = imax(maxfftsize, dimprodmax) * sizeof(fftw_real))))
    {
      printf("failed to allocate memory for `FFT-workspace' (%g MB).\n", bytes / (1024.0 * 1024.0));
      endrun(1);
    }
  /* neutrino power spectrum correction */
  if(!(workspace1 = (fftw_real *) malloc(bytes = imax(maxfftsize, dimprodmax) * sizeof(fftw_real))))
    {
      printf("failed to allocate memory for `FFT-workspace-neutrino` (%g MB).\n", bytes / (1024.0 * 1024.0));
      endrun(1);
    }
  
  if(!(deltagrid = (fftw_real *) malloc(bytes = fftsize * sizeof(fftw_real))))
    {
      printf("failed to allocate memory for `FFT-deltagrid` (%g MB).\n", bytes / (1024.0 * 1024.0));
      endrun(1);
    }
  bytes_tot += bytes;

  if(first_alloc == 1)
    {
      first_alloc = 0;
      if(ThisTask == 0)
	printf("\nAllocated %g MByte for FFT data.\n\n", bytes_tot / (1024.0 * 1024.0));
    }

  fft_of_rhogrid = (fftw_complex *) & rhogrid[0];
  /* neutrino power spectrum correction */
  fft_of_deltagrid = (fftw_complex *) & deltagrid[0];
    
  num_kbins = (int) (log(sqrt(3.) * PMGRID / 0.95) / log(k_interval));
    
  if(All.NumCurrentTiStep == 0) delta_init();
}

/*! This routine frees the space allocated for the parallel FFT algorithm.
 */
void pm_init_periodic_free(void)
{
  /* allocate the memory to hold the FFT fields */
  free(workspace);
  free(forcegrid);
  free(rhogrid);
  /* neutrino power spectrum correction */
  free(workspace1);
  free(deltagrid);
}



/*! Calculates the long-range periodic force given the particle positions
 *  using the PM method.  The force is Gaussian filtered with Asmth, given in
 *  mesh-cell units. We carry out a CIC charge assignment, and compute the
 *  potenial by Fourier transform methods. The potential is finite differenced
 *  using a 4-point finite differencing formula, and the forces are
 *  interpolated tri-linearly to the particle positions. The CIC kernel is
 *  deconvolved. Note that the particle distribution is not in the slab
 *  decomposition that is used for the FFT. Instead, overlapping patches
 *  between local domains and FFT slabs are communicated as needed.
 */
void pmforce_periodic(void)
{
  double k2, kx, ky, kz, smth;
  double dx, dy, dz;
  double fx, fy, fz, ff;
  double asmth2, fac, acc_dim;
  int i, j, slab, level, sendTask, recvTask;
  int x, y, z, xl, yl, zl, xr, yr, zr, xll, yll, zll, xrr, yrr, zrr, ip, dim;
  int slab_x, slab_y, slab_z;
  int slab_xx, slab_yy, slab_zz;
  int meshmin[3], meshmax[3], sendmin, sendmax, recvmin, recvmax;
  int rep, ncont, cont_sendmin[2], cont_sendmax[2], cont_recvmin[2], cont_recvmax[2];
  int dimx, dimy, dimz, recv_dimx, recv_dimy, recv_dimz;
  MPI_Status status;

  if(ThisTask == 0)
    {
      printf("Starting periodic PM calculation.\n");
      fflush(stdout);
    }

  force_treefree();

  asmth2 = (2 * M_PI) * All.Asmth[0] / All.BoxSize;
  asmth2 *= asmth2;

  fac = All.G / (M_PI * All.BoxSize);	/* to get potential */
  fac *= 1 / (2 * All.BoxSize / PMGRID);	/* for finite differencing */

  /* first, establish the extension of the local patch in the PMGRID  */

  for(j = 0; j < 3; j++)
    {
      meshmin[j] = PMGRID;
      meshmax[j] = 0;
    }

  for(i = 0; i < NumPart; i++)
    {
      for(j = 0; j < 3; j++)
	      {
	        slab = to_slab_fac * P[i].Pos[j];
	        if(slab >= PMGRID)
	          slab = PMGRID - 1;

	        if(slab < meshmin[j])
	          meshmin[j] = slab;

	        if(slab > meshmax[j])
	          meshmax[j] = slab;
	      }
    }

  MPI_Allgather(meshmin, 3, MPI_INT, meshmin_list, 3, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(meshmax, 3, MPI_INT, meshmax_list, 3, MPI_INT, MPI_COMM_WORLD);

  dimx = meshmax[0] - meshmin[0] + 2;
  dimy = meshmax[1] - meshmin[1] + 2;
  dimz = meshmax[2] - meshmin[2] + 2;

  pm_init_periodic_allocate((dimx + 4) * (dimy + 4) * (dimz + 4));

  for(i = 0; i < dimx * dimy * dimz; i++)
    workspace[i] = 0;

  if(ThisTask == 0)
    {
      printf("All.Time %f, All.Ti_current %d All.Timebase_interval %f\n", All.Time, All.Ti_Current, All.Timebase_interval);
    }

  for(i = 0; i < NumPart; i++)
    {
      slab_x = to_slab_fac * P[i].Pos[0];
      if(slab_x >= PMGRID)
	      slab_x = PMGRID - 1;
      dx = to_slab_fac * P[i].Pos[0] - slab_x;
      slab_x -= meshmin[0];
      slab_xx = slab_x + 1;

      slab_y = to_slab_fac * P[i].Pos[1];
      if(slab_y >= PMGRID)
	      slab_y = PMGRID - 1;
      dy = to_slab_fac * P[i].Pos[1] - slab_y;
      slab_y -= meshmin[1];
      slab_yy = slab_y + 1;

      slab_z = to_slab_fac * P[i].Pos[2];
      if(slab_z >= PMGRID)
	      slab_z = PMGRID - 1;
      dz = to_slab_fac * P[i].Pos[2] - slab_z;
      slab_z -= meshmin[2];
      slab_zz = slab_z + 1;

      workspace[(slab_x * dimy + slab_y) * dimz + slab_z] += P[i].Mass * (1.0 - dx) * (1.0 - dy) * (1.0 - dz);
      workspace[(slab_x * dimy + slab_yy) * dimz + slab_z] += P[i].Mass * (1.0 - dx) * dy * (1.0 - dz);
      workspace[(slab_x * dimy + slab_y) * dimz + slab_zz] += P[i].Mass * (1.0 - dx) * (1.0 - dy) * dz;
      workspace[(slab_x * dimy + slab_yy) * dimz + slab_zz] += P[i].Mass * (1.0 - dx) * dy * dz;

      workspace[(slab_xx * dimy + slab_y) * dimz + slab_z] += P[i].Mass * (dx) * (1.0 - dy) * (1.0 - dz);
      workspace[(slab_xx * dimy + slab_yy) * dimz + slab_z] += P[i].Mass * (dx) * dy * (1.0 - dz);
      workspace[(slab_xx * dimy + slab_y) * dimz + slab_zz] += P[i].Mass * (dx) * (1.0 - dy) * dz;
      workspace[(slab_xx * dimy + slab_yy) * dimz + slab_zz] += P[i].Mass * (dx) * dy * dz;
    }
  /* neutrino power spectrum correction */
  for(i = 0; i < fftsize; i++)	/* clear local density field */
    {
      rhogrid[i] = 0;
      deltagrid[i] = 0;
    }

  for(level = 0; level < (1 << PTask); level++)	/* note: for level=0, target is the same task */
    {
      sendTask = ThisTask;
      recvTask = ThisTask ^ level;
      if(recvTask < NTask)
	      {
	        /* check how much we have to send */
	        sendmin = 2 * PMGRID;
	        sendmax = -1;
	        for(slab_x = meshmin[0]; slab_x < meshmax[0] + 2; slab_x++)
	          if(slab_to_task[slab_x % PMGRID] == recvTask)
	            {
		            if(slab_x < sendmin)
		              sendmin = slab_x;
		            if(slab_x > sendmax)
		              sendmax = slab_x;
	            }
	        if(sendmax == -1)
	          sendmin = 0;

	        /* check how much we have to receive */
	        recvmin = 2 * PMGRID;
	        recvmax = -1;
	        for(slab_x = meshmin_list[3 * recvTask]; slab_x < meshmax_list[3 * recvTask] + 2; slab_x++)
	          if(slab_to_task[slab_x % PMGRID] == sendTask)
	            {
		            if(slab_x < recvmin)
		              recvmin = slab_x;
		            if(slab_x > recvmax)
		              recvmax = slab_x;
	            }
	        if(recvmax == -1)
	          recvmin = 0;

	        if((recvmax - recvmin) >= 0 || (sendmax - sendmin) >= 0)	/* ok, we have a contribution to the slab */
	          {
	            recv_dimx = meshmax_list[3 * recvTask + 0] - meshmin_list[3 * recvTask + 0] + 2;
	            recv_dimy = meshmax_list[3 * recvTask + 1] - meshmin_list[3 * recvTask + 1] + 2;
	            recv_dimz = meshmax_list[3 * recvTask + 2] - meshmin_list[3 * recvTask + 2] + 2;

	            if(level > 0)
		            {
		              MPI_Sendrecv(workspace + (sendmin - meshmin[0]) * dimy * dimz,
			              (sendmax - sendmin + 1) * dimy * dimz * sizeof(fftw_real), MPI_BYTE, recvTask,
			              TAG_PERIODIC_A, forcegrid,
			              (recvmax - recvmin + 1) * recv_dimy * recv_dimz * sizeof(fftw_real), MPI_BYTE,
			              recvTask, TAG_PERIODIC_A, MPI_COMM_WORLD, &status);
		            }
	            else
		            {
		              memcpy(forcegrid, workspace + (sendmin - meshmin[0]) * dimy * dimz,
			              (sendmax - sendmin + 1) * dimy * dimz * sizeof(fftw_real));
		            }

	            for(slab_x = recvmin; slab_x <= recvmax; slab_x++)
		            {
		              slab_xx = (slab_x % PMGRID) - first_slab_of_task[ThisTask];

		              if(slab_xx >= 0 && slab_xx < slabs_per_task[ThisTask])
		                {
		                  for(slab_y = meshmin_list[3 * recvTask + 1];
			                  slab_y <= meshmax_list[3 * recvTask + 1] + 1; slab_y++)
			                  {
			                    slab_yy = slab_y;
			                    if(slab_yy >= PMGRID)
			                      slab_yy -= PMGRID;

			                    for(slab_z = meshmin_list[3 * recvTask + 2];
			                      slab_z <= meshmax_list[3 * recvTask + 2] + 1; slab_z++)
			                      {
			                        slab_zz = slab_z;
			                        if(slab_zz >= PMGRID)
				                        slab_zz -= PMGRID;

			                        rhogrid[PMGRID * PMGRID2 * slab_xx + PMGRID2 * slab_yy + slab_zz] +=
				                        forcegrid[((slab_x - recvmin) * recv_dimy +
					                        (slab_y - meshmin_list[3 * recvTask + 1])) * recv_dimz +
					                        (slab_z - meshmin_list[3 * recvTask + 2])];
			                      }
			                  }
		                }
		            }
	          }
	      }
    }
  /* Do the FFT of the density field */

    rfftwnd_mpi(fft_forward_plan, 1, rhogrid, workspace, FFTW_TRANSPOSED_ORDER);
    memcpy(deltagrid, rhogrid, fftsize* sizeof(fftw_real));
    
  /* multiply with Green's function for the potential */

  for(y = slabstart_y; y < slabstart_y + nslab_y; y++)
    for(x = 0; x < PMGRID; x++)
      for(z = 0; z < PMGRID / 2 + 1; z++)
	      {
	        if(x > PMGRID / 2)
	          kx = x - PMGRID;
	        else
	          kx = x;
	        if(y > PMGRID / 2)
	          ky = y - PMGRID;
	        else
	          ky = y;
	        if(z > PMGRID / 2)
	          kz = z - PMGRID;
	        else
	          kz = z;

	        k2 = kx * kx + ky * ky + kz * kz;

	        if(k2 > 0)
	          {
	            smth = -exp(-k2 * asmth2) / k2;

	            /* do deconvolution */

	            fx = fy = fz = 1;
	            if(kx != 0)
		            {
		              fx = (M_PI * kx) / PMGRID;
		              fx = sin(fx) / fx;
		            }
	            if(ky != 0)
		            {
		              fy = (M_PI * ky) / PMGRID;
		              fy = sin(fy) / fy;
		            }
	            if(kz != 0)
		            {
		              fz = (M_PI * kz) / PMGRID;
		              fz = sin(fz) / fz;
		            }
	            ff = 1 / (fx * fy * fz);
	            smth *= ff * ff * ff * ff;

	            /* end deconvolution */
	            ip = PMGRID * (PMGRID / 2 + 1) * (y - slabstart_y) + (PMGRID / 2 + 1) * x + z;
	            fft_of_rhogrid[ip].re *= smth;
	            fft_of_rhogrid[ip].im *= smth;
              /* neutrino power spectrum correction */
              fft_of_deltagrid[ip].re *= ff * ff;
              fft_of_deltagrid[ip].im *= ff * ff;    
            }
	      }
  
  if(All.neutrino_scheme == 4.0)
    {     
			int b;        
      double kk;
      /* TODO: recalibrate the mass density of particles, by supporting multiple types. */
      All.temp_rho_mean /= pow(PMGRID, 3);
      
      double *delta_cb_local;
      delta_cb_local = (double*) malloc((num_kbins) * sizeof(double));
			/* pre calculate the neutrino fraction first. */

			if(All.NumCurrentTiStep == 0) find_output_time();
      for(i = 0; i < output_time_size; i++)
        {
          if(All.Time >= output_time_array[i] && All.a_last_pm_step < output_time_array[i])
            {
              if(ThisTask == 0)
                {
                  output_ratio(num_kbins, i);
                }
            }
        }
        
      for(b = 0; b < num_kbins; b++)
        {
          delta_cb_local[b] = 0.;
          count_b[b] = 0.;
          count_b_local[b] = 0.;
        }  
			/* Calculate the neutrino fraction at this time. */
	    Fs.Omega_nu_temp = (double*) malloc((All.NNeutrino)*sizeof(double));
      Fs.Omega_nu_temp_total = 0.0;
      Fs.fnu_total = 0.0;
	    for(i = 0; i < All.NNeutrino; i++)
        {
		      Fs.Omega_nu_temp[i] = neutrino_integration(All.Time, All.mass[i], All.xi[i], i);
		      Fs.Omega_nu_temp_total += Fs.Omega_nu_temp[i];
	      }
	    for(i = 0; i < All.NNeutrino; i++)
        {
	 	      Fs.fnu[i] = Fs.Omega_nu_temp[i] / (Fs.Omega_nu_temp_total + (All.Omega0 - All.Omega_nu0_expan) / pow(All.Time, 3));
      	  Fs.fnu_total += Fs.fnu[i];
	      }
      
      /* Calculate the power spectrum of CDM and baryon. */
      for(y = slabstart_y; y < slabstart_y + nslab_y; y++)
        for(x = 0; x < PMGRID; x++)
          for(z = 0; z < PMGRID / 2 + 1; z++)
            {
              if(x > PMGRID / 2)
                kx = x - PMGRID;
              else
                kx = x;
              if(y > PMGRID / 2)
                ky = y - PMGRID;
              else
                ky = y;
              if(z > PMGRID / 2)
                kz = z - PMGRID;
              else
                kz = z;
              
              k2 = kx * kx + ky * ky + kz * kz;
              kk = pow(k2, 0.5) * 2. * M_PI * 1e3 / All.BoxSize;
              
              ip = PMGRID * (PMGRID / 2 + 1) * (y - slabstart_y) + (PMGRID / 2 + 1) * x + z;

              for(b = 0; b < num_kbins; b++)
                {
                  if(kk >= k_array0[b] && kk < k_array0[b+1])
                    {
                      count_b_local[b] = count_b_local[b] + 1.;
                      /* neutrino overdensity correction because the data stored in deltagrid is density but not overdensity. */
                      delta_cb_local[b] += (fft_of_deltagrid[ip].re * fft_of_deltagrid[ip].re + fft_of_deltagrid[ip].im * fft_of_deltagrid[ip].im) / pow(All.temp_rho_mean, 2);
                    }
                }
            }
      MPI_Allreduce(delta_cb_local, delta_cb, num_kbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(count_b_local, count_b, num_kbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      /* for the first timestep, gsl interpolation to get the delta_nu_0 */
      if(All.NumCurrentTiStep == 0)
        {
          double pre_estimate, ratio_temp = 1.;
          /* This is a estimate, assumuting delta_nu is proportional to fnu. */
          pre_estimate = 0.;
          for(i = 0; i < All.NNeutrino; i++)
            {
              pre_estimate += (Fs.Omega_nu_temp[i]*Fs.Omega_nu_temp[i]);
            }
          if(ThisTask == 0)
            {
              for(i = 0; i < All.NNeutrino; i++)
                {
                  printf("Omega_nu[%d]=%g\t", i, Fs.Omega_nu_temp[i]);
                }
              printf("pre_estimate=%g\t Omega_nu_total=%g\t prefact1=%g\n", pre_estimate, (double)(Fs.Omega_nu_temp_total, Fs.Omega_nu_temp_total * Fs.Omega_nu_temp[i] / (double)pre_estimate));
            }
          Table_rho_init(num_kbins);
					if(ThisTask == 0)printf("The first interpolation: Starting...\n");
          gsl_interp_accel *acc = gsl_interp_accel_alloc();
          gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, rd_size_count);
          gsl_spline_init(spline, rd_array_k, rd_array_pk, rd_size_count);
					Fs.count = 0;
          Fs.loga_for_rho[Fs.count] = log(All.Time);
          for(b = 0; b < num_kbins; b++)
            {
							Fs.rho_tot[b][Fs.count] = 0.;
              if(count_b[b] > 0)
                {
                  delta_cb[b] = sqrt(delta_cb[b] / count_b[b]);
                  ratio_temp  = gsl_spline_eval(spline, k_array[b], acc);
                  //if(ThisTask == 0) printf("k=%g\t ratio= %g\t detla_cb= %g\t prefact= %g\t", k_array[b], sqrt(ratio_temp), delta_cb[b], Fs.Omega_nu_temp_total * Fs.Omega_nu_temp[i] / pre_estimate);
                  for(i = 0; i < All.NNeutrino; i++)
                    {
                      delta_nu[b*All.NNeutrino+i] = delta_cb[b] * sqrt(ratio_temp) * Fs.Omega_nu_temp_total * Fs.Omega_nu_temp[i] / pre_estimate;
                      delta_nu_0[b*All.NNeutrino+i] = delta_nu[b*All.NNeutrino+i];
                      //if(ThisTask == 0) printf("delta_nu[%d]=%g\t", i, delta_nu[b*All.NNeutrino+i]);
                    }
                  //if(ThisTask == 0) printf("\n");
                  Fs.rho_tot[b][Fs.count] = All.rocr * delta_cb[b] * ((All.Omega0 - All.Omega_nu0_expan) / pow(All.Time, 3) + Fs.Omega_nu_temp_total * sqrt(ratio_temp));
                }
            }
					Fs.count++;
          gsl_interp_accel_free(acc);
          gsl_spline_free(spline);
					if(ThisTask == 0)printf("The first interpolation: Finished.\n");
        }
      /* after first timestep, the neutrino will be evolved. */
      if(All.NumCurrentTiStep > 0)
        {
          /* parallel running for free-streaming method. */ 
          double *delta_nu_local;
          delta_nu_local = (double *) malloc((All.NNeutrino*num_kbins) * sizeof(double));
          for(b = 0; b < num_kbins; b++)
            {
              for(i = 0; i < All.NNeutrino; i++) delta_nu_local[b*All.NNeutrino+i] = 0.;
            }
					Fs.loga_for_rho[Fs.count] = log(All.Time);
          Fs.h[Fs.count]    = hubble(All.Time);
          for(b = 0; b < num_kbins; b++)
            {
							Fs.rho_tot[b][Fs.count] = 0.;
              if(count_b[b] > 0)
                {
                  delta_cb[b] = sqrt(delta_cb[b] / count_b[b]);
									/* estimate the rho_tot in current time. */
									Fs.rho_tot[b][Fs.count] = delta_cb[b] * (All.Omega0 - All.Omega_nu0_expan) / pow(All.Time, 3);
									for(i = 0; i < All.NNeutrino; i++)
										{
											Fs.rho_tot[b][Fs.count] += Fs.Omega_nu_temp[i] * delta_nu[b*All.NNeutrino+i];
										}
									Fs.rho_tot[b][Fs.count] *= All.rocr;
                }
            }
					Fs.count++;
					for(i = 0; i < All.NNeutrino; i++)
						{
#ifndef PHIFORMULA
							Fs.spline_phi = gsl_spline_alloc(gsl_interp_cspline, 10000);
							Fs.acc_phi = gsl_interp_accel_alloc();
							gsl_spline_init(Fs.spline_phi, Fs.Phiarrayk, Fs.Phiarray[i], 10000);
#endif
              //if(ThisTask == 0) printf("k=%g\t", k_array[b]);
              b = ThisTask;
              do
                {
                  delta_nu_local[b*All.NNeutrino+i] = frstr(k_array[b], delta_nu_0[b*All.NNeutrino+i], i, b);
                  b += NTask;
                }
              while(b < num_kbins);
#ifndef PHIFORMULA
							gsl_interp_accel_free(Fs.acc_phi);
							gsl_spline_free(Fs.spline_phi);
#endif
							//printf("Finishing interpolation for phi on Task %d\n", ThisTask);
						}
          MPI_Allreduce(delta_nu_local, delta_nu, All.NNeutrino*num_kbins, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
          free(delta_nu_local);
          if(delta_nu_local != NULL) delta_nu_local = NULL;

          //printf("fnu %f xi %f--------------\n", fnu, All.xi_1);
          //if(ThisTask == 0) printf("### k\t ratio_nu_1\t ratio_nu_2\t ratio_nu_3\n");
          for(y = slabstart_y; y < slabstart_y + nslab_y; y++)
              for(x = 0; x < PMGRID; x++)
                  for(z = 0; z < PMGRID / 2 + 1; z++)
                    {
                        if(x > PMGRID / 2)
                            kx = x - PMGRID;
                        else
                            kx = x;
                        if(y > PMGRID / 2)
                            ky = y - PMGRID;
                        else
                            ky = y;
                        if(z > PMGRID / 2)
                            kz = z - PMGRID;
                        else
                            kz = z;
                        
                        k2 = kx * kx + ky * ky + kz * kz;
                        kk = pow(k2, 0.5) * 2. * M_PI * 1e3 / All.BoxSize;
                        ip = PMGRID * (PMGRID / 2 + 1) * (y - slabstart_y) + (PMGRID / 2 + 1) * x + z;
                        
                        for(b = 0; b < num_kbins; b++)
                          {
                            if(kk >= k_array0[b] && kk < k_array0[b+1])
                              {
                                //if(ThisTask == 0) printf("k=%g\t pk_b=%g\t", kk, pk_b[b]);
                                //if(ThisTask == 0) printf("%g\t", kk);
				                        double sum=0.;
				                        for(i = 0; i < All.NNeutrino; i++)
                                  {
				                            sum += Fs.fnu[i]*delta_nu[b*All.NNeutrino+i];
                                    //if(ThisTask == 0)printf("%g\t", pk_nub[b*All.NNeutrino+i]/pk_b[b]);
				                          }
                                //if(ThisTask == 0) printf("\n");
                                /* Modify the bug: */
                                fft_of_rhogrid[ip].re = fft_of_rhogrid[ip].re + fft_of_rhogrid[ip].re * fabs(sum / delta_cb[b]) / (1. - Fs.fnu_total);
                                fft_of_rhogrid[ip].im = fft_of_rhogrid[ip].im + fft_of_rhogrid[ip].im * fabs(sum / delta_cb[b]) / (1. - Fs.fnu_total);
                              }
                          }
                    }
					/* Save the rho_tot data. */
					if(ThisTask == 0)
						{
							for(b = 0; b < num_kbins; b++)
								{
									Fs.rho_tot[b][Fs.count-1] = delta_cb[b] * (All.Omega0 - All.Omega_nu0_expan) / pow(All.Time, 3);
									for(i = 0; i < All.NNeutrino; i++)
										{
											Fs.rho_tot[b][Fs.count-1] += Fs.Omega_nu_temp[i] * delta_nu[i];
										}
                  Fs.rho_tot[b][Fs.count-1] *= All.rocr;
								}
						}
          All.a_last_pm_step = All.Time;
        }
     
      if(ThisTask == 0)
        {
          printf("time now %f time max %f\n", All.Time, All.TimeMax);
        }
      if(fabs(All.Time - All.TimeMax) < 1e-6)
        {
          if(ThisTask == 0)
            {
              output_ratio(num_kbins, output_time_size);
            }
        }
			//printf("Starting cleaning up the cache on Task: %d\n", ThisTask);
      //if(delta_cb_local != NULL) free(delta_cb_local);
			//printf("Finish free delta_cb_local on Task %d", ThisTask);
			//printf("Clean up the cache on Task:%d", ThisTask);
    }
    


  if(All.neutrino_scheme > 1.5 && All.Time > All.TimeBegin && ThisTask == 0)
    {
      printf("here done the %.1f correction step %d\n", All.neutrino_scheme, All.NumCurrentTiStep);
    }
  if(slabstart_y == 0)
    {
      fft_of_rhogrid[0].re = fft_of_rhogrid[0].im = 0.0;
    }
  /* Do the FFT to get the potential */

  rfftwnd_mpi(fft_inverse_plan, 1, rhogrid, workspace, FFTW_TRANSPOSED_ORDER);
 
  /* Now rhogrid holds the potential */
  /* construct the potential for the local patch */


  dimx = meshmax[0] - meshmin[0] + 6;
  dimy = meshmax[1] - meshmin[1] + 6;
  dimz = meshmax[2] - meshmin[2] + 6;

  for(level = 0; level < (1 << PTask); level++)	/* note: for level=0, target is the same task */
    {
      sendTask = ThisTask;
      recvTask = ThisTask ^ level;

      if(recvTask < NTask)
	{

	  /* check how much we have to send */
	  sendmin = 2 * PMGRID;
	  sendmax = -PMGRID;
	  for(slab_x = meshmin_list[3 * recvTask] - 2; slab_x < meshmax_list[3 * recvTask] + 4; slab_x++)
	    if(slab_to_task[(slab_x + PMGRID) % PMGRID] == sendTask)
	      {
		if(slab_x < sendmin)
		  sendmin = slab_x;
		if(slab_x > sendmax)
		  sendmax = slab_x;
	      }
	  if(sendmax == -PMGRID)
	    sendmin = sendmax + 1;


	  /* check how much we have to receive */
	  recvmin = 2 * PMGRID;
	  recvmax = -PMGRID;
	  for(slab_x = meshmin[0] - 2; slab_x < meshmax[0] + 4; slab_x++)
	    if(slab_to_task[(slab_x + PMGRID) % PMGRID] == recvTask)
	      {
		if(slab_x < recvmin)
		  recvmin = slab_x;
		if(slab_x > recvmax)
		  recvmax = slab_x;
	      }
	  if(recvmax == -PMGRID)
	    recvmin = recvmax + 1;

	  if((recvmax - recvmin) >= 0 || (sendmax - sendmin) >= 0)	/* ok, we have a contribution to the slab */
	    {
	      recv_dimx = meshmax_list[3 * recvTask + 0] - meshmin_list[3 * recvTask + 0] + 6;
	      recv_dimy = meshmax_list[3 * recvTask + 1] - meshmin_list[3 * recvTask + 1] + 6;
	      recv_dimz = meshmax_list[3 * recvTask + 2] - meshmin_list[3 * recvTask + 2] + 6;

	      ncont = 1;
	      cont_sendmin[0] = sendmin;
	      cont_sendmax[0] = sendmax;
	      cont_sendmin[1] = sendmax + 1;
	      cont_sendmax[1] = sendmax;

	      cont_recvmin[0] = recvmin;
	      cont_recvmax[0] = recvmax;
	      cont_recvmin[1] = recvmax + 1;
	      cont_recvmax[1] = recvmax;

	      for(slab_x = sendmin; slab_x <= sendmax; slab_x++)
		{
		  if(slab_to_task[(slab_x + PMGRID) % PMGRID] != ThisTask)
		    {
		      /* non-contiguous */
		      cont_sendmax[0] = slab_x - 1;
		      while(slab_to_task[(slab_x + PMGRID) % PMGRID] != ThisTask)
			slab_x++;
		      cont_sendmin[1] = slab_x;
		      ncont++;
		    }
		}

	      for(slab_x = recvmin; slab_x <= recvmax; slab_x++)
		{
		  if(slab_to_task[(slab_x + PMGRID) % PMGRID] != recvTask)
		    {
		      /* non-contiguous */
		      cont_recvmax[0] = slab_x - 1;
		      while(slab_to_task[(slab_x + PMGRID) % PMGRID] != recvTask)
			slab_x++;
		      cont_recvmin[1] = slab_x;
		      if(ncont == 1)
			ncont++;
		    }
		}


	      for(rep = 0; rep < ncont; rep++)
		{
		  sendmin = cont_sendmin[rep];
		  sendmax = cont_sendmax[rep];
		  recvmin = cont_recvmin[rep];
		  recvmax = cont_recvmax[rep];

		  /* prepare what we want to send */
		  if(sendmax - sendmin >= 0)
		    {
		      for(slab_x = sendmin; slab_x <= sendmax; slab_x++)
			{
			  slab_xx = ((slab_x + PMGRID) % PMGRID) - first_slab_of_task[ThisTask];

			  for(slab_y = meshmin_list[3 * recvTask + 1] - 2;
			      slab_y < meshmax_list[3 * recvTask + 1] + 4; slab_y++)
			    {
			      slab_yy = (slab_y + PMGRID) % PMGRID;

			      for(slab_z = meshmin_list[3 * recvTask + 2] - 2;
				  slab_z < meshmax_list[3 * recvTask + 2] + 4; slab_z++)
				{
				  slab_zz = (slab_z + PMGRID) % PMGRID;

				  forcegrid[((slab_x - sendmin) * recv_dimy +
					     (slab_y - (meshmin_list[3 * recvTask + 1] - 2))) * recv_dimz +
					    slab_z - (meshmin_list[3 * recvTask + 2] - 2)] =
				    rhogrid[PMGRID * PMGRID2 * slab_xx + PMGRID2 * slab_yy + slab_zz];
				}
			    }
			}
		    }

		  if(level > 0)
		    {
		      MPI_Sendrecv(forcegrid,
				   (sendmax - sendmin + 1) * recv_dimy * recv_dimz * sizeof(fftw_real),
				   MPI_BYTE, recvTask, TAG_PERIODIC_B,
				   workspace + (recvmin - (meshmin[0] - 2)) * dimy * dimz,
				   (recvmax - recvmin + 1) * dimy * dimz * sizeof(fftw_real), MPI_BYTE,
				   recvTask, TAG_PERIODIC_B, MPI_COMM_WORLD, &status);
		    }
		  else
		    {
		      memcpy(workspace + (recvmin - (meshmin[0] - 2)) * dimy * dimz,
			     forcegrid, (recvmax - recvmin + 1) * dimy * dimz * sizeof(fftw_real));
		    }
		}
	    }
	}
    }


  dimx = meshmax[0] - meshmin[0] + 2;
  dimy = meshmax[1] - meshmin[1] + 2;
  dimz = meshmax[2] - meshmin[2] + 2;

  recv_dimx = meshmax[0] - meshmin[0] + 6;
  recv_dimy = meshmax[1] - meshmin[1] + 6;
  recv_dimz = meshmax[2] - meshmin[2] + 6;


  for(dim = 0; dim < 3; dim++)	/* Calculate each component of the force. */
    {
      /* get the force component by finite differencing the potential */
      /* note: "workspace" now contains the potential for the local patch, plus a suffiently large buffer region */

      for(x = 0; x < meshmax[0] - meshmin[0] + 2; x++)
	for(y = 0; y < meshmax[1] - meshmin[1] + 2; y++)
	  for(z = 0; z < meshmax[2] - meshmin[2] + 2; z++)
	    {
	      xrr = xll = xr = xl = x;
	      yrr = yll = yr = yl = y;
	      zrr = zll = zr = zl = z;

	      switch (dim)
		{
		case 0:
		  xr = x + 1;
		  xrr = x + 2;
		  xl = x - 1;
		  xll = x - 2;
		  break;
		case 1:
		  yr = y + 1;
		  yl = y - 1;
		  yrr = y + 2;
		  yll = y - 2;
		  break;
		case 2:
		  zr = z + 1;
		  zl = z - 1;
		  zrr = z + 2;
		  zll = z - 2;
		  break;
		}

	      forcegrid[(x * dimy + y) * dimz + z]
		=
		fac * ((4.0 / 3) *
		       (workspace[((xl + 2) * recv_dimy + (yl + 2)) * recv_dimz + (zl + 2)]
			- workspace[((xr + 2) * recv_dimy + (yr + 2)) * recv_dimz + (zr + 2)]) -
		       (1.0 / 6) *
		       (workspace[((xll + 2) * recv_dimy + (yll + 2)) * recv_dimz + (zll + 2)] -
			workspace[((xrr + 2) * recv_dimy + (yrr + 2)) * recv_dimz + (zrr + 2)]));
	    }

      /* read out the forces */

      for(i = 0; i < NumPart; i++)
	{
	  slab_x = to_slab_fac * P[i].Pos[0];
	  if(slab_x >= PMGRID)
	    slab_x = PMGRID - 1;
	  dx = to_slab_fac * P[i].Pos[0] - slab_x;
	  slab_x -= meshmin[0];
	  slab_xx = slab_x + 1;

	  slab_y = to_slab_fac * P[i].Pos[1];
	  if(slab_y >= PMGRID)
	    slab_y = PMGRID - 1;
	  dy = to_slab_fac * P[i].Pos[1] - slab_y;
	  slab_y -= meshmin[1];
	  slab_yy = slab_y + 1;

	  slab_z = to_slab_fac * P[i].Pos[2];
	  if(slab_z >= PMGRID)
	    slab_z = PMGRID - 1;
	  dz = to_slab_fac * P[i].Pos[2] - slab_z;
	  slab_z -= meshmin[2];
	  slab_zz = slab_z + 1;

	  acc_dim =
	    forcegrid[(slab_x * dimy + slab_y) * dimz + slab_z] * (1.0 - dx) * (1.0 - dy) * (1.0 - dz);
	  acc_dim += forcegrid[(slab_x * dimy + slab_yy) * dimz + slab_z] * (1.0 - dx) * dy * (1.0 - dz);
	  acc_dim += forcegrid[(slab_x * dimy + slab_y) * dimz + slab_zz] * (1.0 - dx) * (1.0 - dy) * dz;
	  acc_dim += forcegrid[(slab_x * dimy + slab_yy) * dimz + slab_zz] * (1.0 - dx) * dy * dz;

	  acc_dim += forcegrid[(slab_xx * dimy + slab_y) * dimz + slab_z] * (dx) * (1.0 - dy) * (1.0 - dz);
	  acc_dim += forcegrid[(slab_xx * dimy + slab_yy) * dimz + slab_z] * (dx) * dy * (1.0 - dz);
	  acc_dim += forcegrid[(slab_xx * dimy + slab_y) * dimz + slab_zz] * (dx) * (1.0 - dy) * dz;
	  acc_dim += forcegrid[(slab_xx * dimy + slab_yy) * dimz + slab_zz] * (dx) * dy * dz;

	  P[i].GravPM[dim] = acc_dim;
	}
    }
    
  pm_init_periodic_free();
  force_treeallocate(All.TreeAllocFactor * All.MaxPart, All.MaxPart);

  All.NumForcesSinceLastDomainDecomp = 1 + All.TotNumPart * All.TreeDomainUpdateFrequency;

  if(ThisTask == 0)
    {
      printf("done PM.\n");
      fflush(stdout);
    }
}


/*! Calculates the long-range potential using the PM method.  The potential is
 *  Gaussian filtered with Asmth, given in mesh-cell units. We carry out a CIC
 *  charge assignment, and compute the potenial by Fourier transform
 *  methods. The CIC kernel is deconvolved.
 */
void pmpotential_periodic(void)
{
  double k2, kx, ky, kz, smth;
  double dx, dy, dz;
  double fx, fy, fz, ff;
  double asmth2, fac;
  int i, j, slab, level, sendTask, recvTask;
  int x, y, z, ip;
  int slab_x, slab_y, slab_z;
  int slab_xx, slab_yy, slab_zz;
  int meshmin[3], meshmax[3], sendmin, sendmax, recvmin, recvmax;
  int rep, ncont, cont_sendmin[2], cont_sendmax[2], cont_recvmin[2], cont_recvmax[2];
  int dimx, dimy, dimz, recv_dimx, recv_dimy, recv_dimz;
  MPI_Status status;

  if(ThisTask == 0)
    {
      printf("Starting periodic PM calculation.\n");
      fflush(stdout);
    }

  asmth2 = (2 * M_PI) * All.Asmth[0] / All.BoxSize;
  asmth2 *= asmth2;

  fac = All.G / (M_PI * All.BoxSize);	/* to get potential */

  force_treefree();

  /* first, establish the extension of the local patch in the PMGRID  */

  for(j = 0; j < 3; j++)
    {
      meshmin[j] = PMGRID;
      meshmax[j] = 0;
    }

  for(i = 0; i < NumPart; i++)
    {
      for(j = 0; j < 3; j++)
	{
	  slab = to_slab_fac * P[i].Pos[j];
	  if(slab >= PMGRID)
	    slab = PMGRID - 1;

	  if(slab < meshmin[j])
	    meshmin[j] = slab;

	  if(slab > meshmax[j])
	    meshmax[j] = slab;
	}
    }

  MPI_Allgather(meshmin, 3, MPI_INT, meshmin_list, 3, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(meshmax, 3, MPI_INT, meshmax_list, 3, MPI_INT, MPI_COMM_WORLD);

  dimx = meshmax[0] - meshmin[0] + 2;
  dimy = meshmax[1] - meshmin[1] + 2;
  dimz = meshmax[2] - meshmin[2] + 2;

  pm_init_periodic_allocate((dimx + 4) * (dimy + 4) * (dimz + 4));

  for(i = 0; i < dimx * dimy * dimz; i++)
    workspace[i] = 0;

  for(i = 0; i < NumPart; i++)
    {
      slab_x = to_slab_fac * P[i].Pos[0];
      if(slab_x >= PMGRID)
	slab_x = PMGRID - 1;
      dx = to_slab_fac * P[i].Pos[0] - slab_x;
      slab_x -= meshmin[0];
      slab_xx = slab_x + 1;

      slab_y = to_slab_fac * P[i].Pos[1];
      if(slab_y >= PMGRID)
	slab_y = PMGRID - 1;
      dy = to_slab_fac * P[i].Pos[1] - slab_y;
      slab_y -= meshmin[1];
      slab_yy = slab_y + 1;

      slab_z = to_slab_fac * P[i].Pos[2];
      if(slab_z >= PMGRID)
	slab_z = PMGRID - 1;
      dz = to_slab_fac * P[i].Pos[2] - slab_z;
      slab_z -= meshmin[2];
      slab_zz = slab_z + 1;

      workspace[(slab_x * dimy + slab_y) * dimz + slab_z] += P[i].Mass * (1.0 - dx) * (1.0 - dy) * (1.0 - dz);
      workspace[(slab_x * dimy + slab_yy) * dimz + slab_z] += P[i].Mass * (1.0 - dx) * dy * (1.0 - dz);
      workspace[(slab_x * dimy + slab_y) * dimz + slab_zz] += P[i].Mass * (1.0 - dx) * (1.0 - dy) * dz;
      workspace[(slab_x * dimy + slab_yy) * dimz + slab_zz] += P[i].Mass * (1.0 - dx) * dy * dz;

      workspace[(slab_xx * dimy + slab_y) * dimz + slab_z] += P[i].Mass * (dx) * (1.0 - dy) * (1.0 - dz);
      workspace[(slab_xx * dimy + slab_yy) * dimz + slab_z] += P[i].Mass * (dx) * dy * (1.0 - dz);
      workspace[(slab_xx * dimy + slab_y) * dimz + slab_zz] += P[i].Mass * (dx) * (1.0 - dy) * dz;
      workspace[(slab_xx * dimy + slab_yy) * dimz + slab_zz] += P[i].Mass * (dx) * dy * dz;
    }


  for(i = 0; i < fftsize; i++)	/* clear local density field */
    rhogrid[i] = 0;

  for(level = 0; level < (1 << PTask); level++)	/* note: for level=0, target is the same task */
    {
      sendTask = ThisTask;
      recvTask = ThisTask ^ level;
      if(recvTask < NTask)
	{
	  /* check how much we have to send */
	  sendmin = 2 * PMGRID;
	  sendmax = -1;
	  for(slab_x = meshmin[0]; slab_x < meshmax[0] + 2; slab_x++)
	    if(slab_to_task[slab_x % PMGRID] == recvTask)
	      {
		if(slab_x < sendmin)
		  sendmin = slab_x;
		if(slab_x > sendmax)
		  sendmax = slab_x;
	      }
	  if(sendmax == -1)
	    sendmin = 0;

	  /* check how much we have to receive */
	  recvmin = 2 * PMGRID;
	  recvmax = -1;
	  for(slab_x = meshmin_list[3 * recvTask]; slab_x < meshmax_list[3 * recvTask] + 2; slab_x++)
	    if(slab_to_task[slab_x % PMGRID] == sendTask)
	      {
		if(slab_x < recvmin)
		  recvmin = slab_x;
		if(slab_x > recvmax)
		  recvmax = slab_x;
	      }
	  if(recvmax == -1)
	    recvmin = 0;


	  if((recvmax - recvmin) >= 0 || (sendmax - sendmin) >= 0)	/* ok, we have a contribution to the slab */
	    {
	      recv_dimx = meshmax_list[3 * recvTask + 0] - meshmin_list[3 * recvTask + 0] + 2;
	      recv_dimy = meshmax_list[3 * recvTask + 1] - meshmin_list[3 * recvTask + 1] + 2;
	      recv_dimz = meshmax_list[3 * recvTask + 2] - meshmin_list[3 * recvTask + 2] + 2;

	      if(level > 0)
		{
		  MPI_Sendrecv(workspace + (sendmin - meshmin[0]) * dimy * dimz,
			       (sendmax - sendmin + 1) * dimy * dimz * sizeof(fftw_real), MPI_BYTE, recvTask,
			       TAG_PERIODIC_C, forcegrid,
			       (recvmax - recvmin + 1) * recv_dimy * recv_dimz * sizeof(fftw_real), MPI_BYTE,
			       recvTask, TAG_PERIODIC_C, MPI_COMM_WORLD, &status);
		}
	      else
		{
		  memcpy(forcegrid, workspace + (sendmin - meshmin[0]) * dimy * dimz,
			 (sendmax - sendmin + 1) * dimy * dimz * sizeof(fftw_real));
		}

	      for(slab_x = recvmin; slab_x <= recvmax; slab_x++)
		{
		  slab_xx = (slab_x % PMGRID) - first_slab_of_task[ThisTask];

		  if(slab_xx >= 0 && slab_xx < slabs_per_task[ThisTask])
		    {
		      for(slab_y = meshmin_list[3 * recvTask + 1];
			  slab_y <= meshmax_list[3 * recvTask + 1] + 1; slab_y++)
			{
			  slab_yy = slab_y;
			  if(slab_yy >= PMGRID)
			    slab_yy -= PMGRID;

			  for(slab_z = meshmin_list[3 * recvTask + 2];
			      slab_z <= meshmax_list[3 * recvTask + 2] + 1; slab_z++)
			    {
			      slab_zz = slab_z;
			      if(slab_zz >= PMGRID)
				slab_zz -= PMGRID;

			      rhogrid[PMGRID * PMGRID2 * slab_xx + PMGRID2 * slab_yy + slab_zz] +=
				forcegrid[((slab_x - recvmin) * recv_dimy +
					   (slab_y - meshmin_list[3 * recvTask + 1])) * recv_dimz +
					  (slab_z - meshmin_list[3 * recvTask + 2])];
			    }
			}
		    }
		}
	    }
	}
    }



  /* Do the FFT of the density field */

  rfftwnd_mpi(fft_forward_plan, 1, rhogrid, workspace, FFTW_TRANSPOSED_ORDER);

  /* multiply with Green's function for the potential */

  for(y = slabstart_y; y < slabstart_y + nslab_y; y++)
    for(x = 0; x < PMGRID; x++)
      for(z = 0; z < PMGRID / 2 + 1; z++)
	{
	  if(x > PMGRID / 2)
	    kx = x - PMGRID;
	  else
	    kx = x;
	  if(y > PMGRID / 2)
	    ky = y - PMGRID;
	  else
	    ky = y;
	  if(z > PMGRID / 2)
	    kz = z - PMGRID;
	  else
	    kz = z;

	  k2 = kx * kx + ky * ky + kz * kz;

	  if(k2 > 0)
	    {
	      smth = -exp(-k2 * asmth2) / k2 * fac;
	      /* do deconvolution */
	      fx = fy = fz = 1;
	      if(kx != 0)
		{
		  fx = (M_PI * kx) / PMGRID;
		  fx = sin(fx) / fx;
		}
	      if(ky != 0)
		{
		  fy = (M_PI * ky) / PMGRID;
		  fy = sin(fy) / fy;
		}
	      if(kz != 0)
		{
		  fz = (M_PI * kz) / PMGRID;
		  fz = sin(fz) / fz;
		}
	      ff = 1 / (fx * fy * fz);
	      smth *= ff * ff * ff * ff;
	      /* end deconvolution */

	      ip = PMGRID * (PMGRID / 2 + 1) * (y - slabstart_y) + (PMGRID / 2 + 1) * x + z;
	      fft_of_rhogrid[ip].re *= smth;
	      fft_of_rhogrid[ip].im *= smth;
	    }
	}

  if(slabstart_y == 0)
    fft_of_rhogrid[0].re = fft_of_rhogrid[0].im = 0.0;

  /* Do the FFT to get the potential */

  rfftwnd_mpi(fft_inverse_plan, 1, rhogrid, workspace, FFTW_TRANSPOSED_ORDER);

  /* note: "rhogrid" now contains the potential */



  dimx = meshmax[0] - meshmin[0] + 6;
  dimy = meshmax[1] - meshmin[1] + 6;
  dimz = meshmax[2] - meshmin[2] + 6;

  for(level = 0; level < (1 << PTask); level++)	/* note: for level=0, target is the same task */
    {
      sendTask = ThisTask;
      recvTask = ThisTask ^ level;

      if(recvTask < NTask)
	{

	  /* check how much we have to send */
	  sendmin = 2 * PMGRID;
	  sendmax = -PMGRID;
	  for(slab_x = meshmin_list[3 * recvTask] - 2; slab_x < meshmax_list[3 * recvTask] + 4; slab_x++)
	    if(slab_to_task[(slab_x + PMGRID) % PMGRID] == sendTask)
	      {
		if(slab_x < sendmin)
		  sendmin = slab_x;
		if(slab_x > sendmax)
		  sendmax = slab_x;
	      }
	  if(sendmax == -PMGRID)
	    sendmin = sendmax + 1;


	  /* check how much we have to receive */
	  recvmin = 2 * PMGRID;
	  recvmax = -PMGRID;
	  for(slab_x = meshmin[0] - 2; slab_x < meshmax[0] + 4; slab_x++)
	    if(slab_to_task[(slab_x + PMGRID) % PMGRID] == recvTask)
	      {
		if(slab_x < recvmin)
		  recvmin = slab_x;
		if(slab_x > recvmax)
		  recvmax = slab_x;
	      }
	  if(recvmax == -PMGRID)
	    recvmin = recvmax + 1;

	  if((recvmax - recvmin) >= 0 || (sendmax - sendmin) >= 0)	/* ok, we have a contribution to the slab */
	    {
	      recv_dimx = meshmax_list[3 * recvTask + 0] - meshmin_list[3 * recvTask + 0] + 6;
	      recv_dimy = meshmax_list[3 * recvTask + 1] - meshmin_list[3 * recvTask + 1] + 6;
	      recv_dimz = meshmax_list[3 * recvTask + 2] - meshmin_list[3 * recvTask + 2] + 6;

	      ncont = 1;
	      cont_sendmin[0] = sendmin;
	      cont_sendmax[0] = sendmax;
	      cont_sendmin[1] = sendmax + 1;
	      cont_sendmax[1] = sendmax;

	      cont_recvmin[0] = recvmin;
	      cont_recvmax[0] = recvmax;
	      cont_recvmin[1] = recvmax + 1;
	      cont_recvmax[1] = recvmax;

	      for(slab_x = sendmin; slab_x <= sendmax; slab_x++)
		{
		  if(slab_to_task[(slab_x + PMGRID) % PMGRID] != ThisTask)
		    {
		      /* non-contiguous */
		      cont_sendmax[0] = slab_x - 1;
		      while(slab_to_task[(slab_x + PMGRID) % PMGRID] != ThisTask)
			slab_x++;
		      cont_sendmin[1] = slab_x;
		      ncont++;
		    }
		}

	      for(slab_x = recvmin; slab_x <= recvmax; slab_x++)
		{
		  if(slab_to_task[(slab_x + PMGRID) % PMGRID] != recvTask)
		    {
		      /* non-contiguous */
		      cont_recvmax[0] = slab_x - 1;
		      while(slab_to_task[(slab_x + PMGRID) % PMGRID] != recvTask)
			slab_x++;
		      cont_recvmin[1] = slab_x;
		      if(ncont == 1)
			ncont++;
		    }
		}


	      for(rep = 0; rep < ncont; rep++)
		{
		  sendmin = cont_sendmin[rep];
		  sendmax = cont_sendmax[rep];
		  recvmin = cont_recvmin[rep];
		  recvmax = cont_recvmax[rep];

		  /* prepare what we want to send */
		  if(sendmax - sendmin >= 0)
		    {
		      for(slab_x = sendmin; slab_x <= sendmax; slab_x++)
			{
			  slab_xx = ((slab_x + PMGRID) % PMGRID) - first_slab_of_task[ThisTask];

			  for(slab_y = meshmin_list[3 * recvTask + 1] - 2;
			      slab_y < meshmax_list[3 * recvTask + 1] + 4; slab_y++)
			    {
			      slab_yy = (slab_y + PMGRID) % PMGRID;

			      for(slab_z = meshmin_list[3 * recvTask + 2] - 2;
				  slab_z < meshmax_list[3 * recvTask + 2] + 4; slab_z++)
				{
				  slab_zz = (slab_z + PMGRID) % PMGRID;

				  forcegrid[((slab_x - sendmin) * recv_dimy +
					     (slab_y - (meshmin_list[3 * recvTask + 1] - 2))) * recv_dimz +
					    slab_z - (meshmin_list[3 * recvTask + 2] - 2)] =
				    rhogrid[PMGRID * PMGRID2 * slab_xx + PMGRID2 * slab_yy + slab_zz];
				}
			    }
			}
		    }

		  if(level > 0)
		    {
		      MPI_Sendrecv(forcegrid,
				   (sendmax - sendmin + 1) * recv_dimy * recv_dimz * sizeof(fftw_real),
				   MPI_BYTE, recvTask, TAG_PERIODIC_D,
				   workspace + (recvmin - (meshmin[0] - 2)) * dimy * dimz,
				   (recvmax - recvmin + 1) * dimy * dimz * sizeof(fftw_real), MPI_BYTE,
				   recvTask, TAG_PERIODIC_D, MPI_COMM_WORLD, &status);
		    }
		  else
		    {
		      memcpy(workspace + (recvmin - (meshmin[0] - 2)) * dimy * dimz,
			     forcegrid, (recvmax - recvmin + 1) * dimy * dimz * sizeof(fftw_real));
		    }
		}
	    }
	}
    }


  dimx = meshmax[0] - meshmin[0] + 2;
  dimy = meshmax[1] - meshmin[1] + 2;
  dimz = meshmax[2] - meshmin[2] + 2;

  recv_dimx = meshmax[0] - meshmin[0] + 6;
  recv_dimy = meshmax[1] - meshmin[1] + 6;
  recv_dimz = meshmax[2] - meshmin[2] + 6;



  for(x = 0; x < meshmax[0] - meshmin[0] + 2; x++)
    for(y = 0; y < meshmax[1] - meshmin[1] + 2; y++)
      for(z = 0; z < meshmax[2] - meshmin[2] + 2; z++)
	{
	  forcegrid[(x * dimy + y) * dimz + z] =
	    workspace[((x + 2) * recv_dimy + (y + 2)) * recv_dimz + (z + 2)];
	}


  /* read out the potential */

  for(i = 0; i < NumPart; i++)
    {
      slab_x = to_slab_fac * P[i].Pos[0];
      if(slab_x >= PMGRID)
	slab_x = PMGRID - 1;
      dx = to_slab_fac * P[i].Pos[0] - slab_x;
      slab_x -= meshmin[0];
      slab_xx = slab_x + 1;

      slab_y = to_slab_fac * P[i].Pos[1];
      if(slab_y >= PMGRID)
	slab_y = PMGRID - 1;
      dy = to_slab_fac * P[i].Pos[1] - slab_y;
      slab_y -= meshmin[1];
      slab_yy = slab_y + 1;

      slab_z = to_slab_fac * P[i].Pos[2];
      if(slab_z >= PMGRID)
	slab_z = PMGRID - 1;
      dz = to_slab_fac * P[i].Pos[2] - slab_z;
      slab_z -= meshmin[2];
      slab_zz = slab_z + 1;

      P[i].Potential +=
	forcegrid[(slab_x * dimy + slab_y) * dimz + slab_z] * (1.0 - dx) * (1.0 - dy) * (1.0 - dz);
      P[i].Potential += forcegrid[(slab_x * dimy + slab_yy) * dimz + slab_z] * (1.0 - dx) * dy * (1.0 - dz);
      P[i].Potential += forcegrid[(slab_x * dimy + slab_y) * dimz + slab_zz] * (1.0 - dx) * (1.0 - dy) * dz;
      P[i].Potential += forcegrid[(slab_x * dimy + slab_yy) * dimz + slab_zz] * (1.0 - dx) * dy * dz;

      P[i].Potential += forcegrid[(slab_xx * dimy + slab_y) * dimz + slab_z] * (dx) * (1.0 - dy) * (1.0 - dz);
      P[i].Potential += forcegrid[(slab_xx * dimy + slab_yy) * dimz + slab_z] * (dx) * dy * (1.0 - dz);
      P[i].Potential += forcegrid[(slab_xx * dimy + slab_y) * dimz + slab_zz] * (dx) * (1.0 - dy) * dz;
      P[i].Potential += forcegrid[(slab_xx * dimy + slab_yy) * dimz + slab_zz] * (dx) * dy * dz;
    }

  pm_init_periodic_free();
  force_treeallocate(All.TreeAllocFactor * All.MaxPart, All.MaxPart);

  All.NumForcesSinceLastDomainDecomp = 1 + All.TotNumPart * All.TreeDomainUpdateFrequency;

  if(ThisTask == 0)
    {
      printf("done PM-Potential.\n");
      fflush(stdout);
    }
}

#endif
#endif
