#include <drfftw_mpi.h>

#define  PI          3.14159265358979323846 
#define  GRAVITY     6.672e-8
#define  HUBBLE      3.2407789e-18   /* in h/sec */


double PowerSpec(double kmag);
double GrowthFactor(double astart, double aend);
double growth_nu(double a, double k, double Omega_m, double Omega_nu0_frstr_0);
double F_Omega(double a);
double F2_Omega(double a);
int    read_parameter_file(char *fname);
double PowerSpec_EH(double k);
double PowerSpec_Efstathiou(double k);
double cal_xi2(double xi3);
double cal_xi1(double xi2, double xi3);


#ifdef T3E
typedef short int int4byte;	/* Note: int has 8 Bytes on the T3E ! */
typedef unsigned short int uint4byte;	/* Note: int has 8 Bytes on the T3E ! */
#else
typedef int int4byte;
typedef unsigned int uint4byte;
#endif



extern struct io_header_1
{
  uint4byte npart[6];      /*!< npart[1] gives the number of particles in the present file, other particle types are ignored */
  double mass[6];          /*!< mass[1] gives the particle mass */
  double time;             /*!< time (=cosmological scale factor) of snapshot */
  double redshift;         /*!< redshift of snapshot */
  int4byte flag_sfr;       /*!< flags whether star formation is used (not available in L-Gadget2) */
  int4byte flag_feedback;  /*!< flags whether feedback from star formation is included */
  uint4byte npartTotal[6]; /*!< npart[1] gives the total number of particles in the run. If this number exceeds 2^32, the npartTotal[2] stores
                                the result of a division of the particle number by 2^32, while npartTotal[1] holds the remainder. */
  int4byte flag_cooling;   /*!< flags whether radiative cooling is included */
  int4byte num_files;      /*!< determines the number of files that are used for a snapshot */
  double BoxSize;          /*!< Simulation box size (in code units) */
  double Omega0;           /*!< matter density */
  double OmegaLambda;      /*!< vacuum energy density */
  double HubbleParam;      /*!< little 'h' */
  int4byte flag_stellarage;     /*!< flags whether the age of newly formed stars is recorded and saved */
  int4byte flag_metals;         /*!< flags whether metal enrichment is included */
  int4byte hashtabsize;         /*!< gives the size of the hashtable belonging to this snapshot file */
  char fill[84];		/*!< fills to 256 Bytes */
}
header, header1;


extern int      Nglass;
extern int      *Local_nx_table;
extern int      WhichSpectrum;


extern FILE     *FdTmp, *FdTmpInput;

extern int      Nmesh, Nsample;

extern int      SphereMode;

extern long long IDStart;


extern char     GlassFile[500]; 
extern char     FileWithInputSpectrum[500];

extern int      GlassTileFac; 

extern double   Box;
extern int Seed;

extern long long TotNumPart;

extern int      NumPart;

extern int      NTaskWithN;


extern int      *Slab_to_task;


extern struct part_data 
{
  float Pos[3];
  float Vel[3];
#ifdef  MULTICOMPONENTGLASSFILE                      
  int   Type;
#endif
  long long ID;
} *P;


extern double InitTime;
extern double Redshift;
extern double MassTable[6];


extern char OutputDir[100], FileBase[100];
extern int  NumFilesWrittenInParallel;


extern int      ThisTask, NTask;

extern int      Local_nx, Local_x_start;

extern int  IdStart;

extern unsigned int TotalSizePlusAdditional;
extern rfftwnd_mpi_plan Inverse_plan;
extern rfftwnd_mpi_plan Forward_plan;
//extern fftw_real        *Disp;
extern fftw_real        *Workspace;
//extern fftw_complex     *Cdata;


extern double UnitTime_in_s, UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s;
extern double InputSpectrum_UnitLength_in_cm;
extern double G, Hubble;
extern double RhoCrit;

extern double Omega, OmegaLambda, OmegaDM_2ndSpecies, Sigma8, Omega2;
extern double OmegaBaryon, HubbleParam;
extern double PrimordialIndex;
extern double ShapeGamma;

extern double Dplus; /* growth factor */


extern int    ReNormalizeInputSpectrum;
extern int    deductfromDE;
extern int    expan_on;
extern int    mass_hierarchy;
extern int    lepton_asymmetry;
extern int    ICfrstr;
extern int    NNeutrino;
#ifdef STERILE
extern double Neff;
#endif
extern double H0;
extern double Omega_nu0_expan;
extern double Omega_nu0_frstr;
extern double Mass[5];
extern double Xi[5];
extern double Tneu0;
extern double unittrans;
extern double rocr;
extern double D10;
extern double D11;

extern double ro_init;
extern double Omegam_init;
extern double Omeganu_init;
extern double ronu_init;

#ifdef DIFFERENT_TRANSFER_FUNC
extern int Type, MinType, MaxType;
#endif

extern int    WDM_On;
extern int    WDM_Vtherm_On;
extern double WDM_PartMass_in_kev;
