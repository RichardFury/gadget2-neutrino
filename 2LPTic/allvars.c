#include "allvars.h"


struct io_header_1 header1, header;

int WhichSpectrum;


int SphereMode;
int *Local_nx_table;

FILE *FdTmp, *FdTmpInput;

int Nmesh, Nsample;

long long IDStart;



char GlassFile[500];
char FileWithInputSpectrum[500];

int GlassTileFac;

double Box;
int Seed;

long long TotNumPart;

int NumPart;

int *Slab_to_task;

int NTaskWithN;

struct part_data *P;

int Nglass;

double InitTime;
double Redshift;
double MassTable[6];


char OutputDir[100], FileBase[100];
int NumFilesWrittenInParallel;


int ThisTask, NTask;

int Local_nx, Local_x_start;

int IdStart;

rfftwnd_mpi_plan Inverse_plan;
rfftwnd_mpi_plan Forward_plan;
unsigned int TotalSizePlusAdditional;
//fftw_real *Disp;
fftw_real *Workspace;
//fftw_complex *Cdata;


double UnitTime_in_s, UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s;
double InputSpectrum_UnitLength_in_cm;
double G, Hubble;
double RhoCrit;

double Omega, OmegaLambda, OmegaDM_2ndSpecies, Sigma8, Omega2;
double OmegaBaryon, HubbleParam;
double ShapeGamma;
double PrimordialIndex;
double Dplus;			/* growth factor */

double Omega_nu0_expan;
double Omega_nu0_frstr;
double Mass[5];
double Xi[5];
double mass_1;
double mass_2;
double mass_3;
double xi_1;
double xi_2;
double xi_3;
double Tneu0;
double unittrans;
double rocr;

int deductfromDE;
int ReNormalizeInputSpectrum;
int ICfrstr;
int expan_on;
int mass_hierarchy;
int lepton_asymmetry;
int NNeutrino;
#ifdef STERILE
double Neff;
#endif
double H0;
double D10;
double D11;

double ro_init;
double Omegam_init;
double Omeganu_init;
double ronu_init;

#ifdef DIFFERENT_TRANSFER_FUNC
int Type, MinType, MaxType;
#endif

int WDM_On;
int WDM_Vtherm_On;
double WDM_PartMass_in_kev;
