#include <math.h>
#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <stdlib.h>
#include "allvars.h"
#include "proto.h"
#include "neutrino.h"

#define WORKSIZE 100000

static double R8;
static double r_tophat;

static double AA, BB, CC;
static double nu;
static double Norm;
struct tophatparam
{
  int index
};

static float sqrarg;
#define SQR(a) ((sqrarg = (a)) == 0.0 ? 0.0 : sqrarg * sqrarg)

static int NPowerTable, NPowernuTable;

static struct pow_table
{
  double logk, logD;
}
    *PowerTable,
    *PowernuTable;

double PowerSpec(double k)
{
  double power, alpha, Tf;

  switch (WhichSpectrum)
  {
  case 1:
    power = PowerSpec_EH(k);
    break;

  case 2:
    power = PowerSpec_Tabulated(k);
    break;

  default:
    power = PowerSpec_Efstathiou(k);
    break;
  }

  if (WDM_On == 1)
  {
    /* Eqn. (A9) in Bode, Ostriker & Turok (2001), assuming gX=1.5  */
    alpha = 0.048 * pow((Omega - OmegaBaryon) / 0.4, 0.15) * pow(HubbleParam / 0.65, 1.3) * pow(1.0 / WDM_PartMass_in_kev, 1.15);
    Tf = pow(1 + pow(alpha * k * (3.085678e24 / UnitLength_in_cm), 2 * 1.2), -5.0 / 1.2);
    power *= Tf * Tf;
  }

#if defined(MULTICOMPONENTGLASSFILE) && defined(DIFFERENT_TRANSFER_FUNC)

  if (Type == 2)
  {
    power = PowerSpec_DM_2ndSpecies(k);
  }

#endif

  power *= pow(k, PrimordialIndex - 1.0);

  return power;
}

double PowerSpec_nu(double k)
{
  double power;
  power = PowerSpecNu_Tabulated(k);
  power *= pow(k, PrimordialIndex - 1.0);
  return power;
}

double PowerSpec_DM_2ndSpecies(double k)
{
  /* at the moment, we simply call the Eistenstein & Hu spectrum
   * for the second DM species, but this could be replaced with
   * something more physical, say for neutrinos
   */

  double power;

  power = Norm * k * pow(tk_eh(k), 2);

  return power;
}

void read_power_table(void)
{
  FILE *fd;
  char buf[500];
  double k, p;

  sprintf(buf, FileWithInputSpectrum);

  if (!(fd = fopen(buf, "r")))
  {
    printf("can't read input spectrum in file '%s' on task %d\n", buf, ThisTask);
    FatalError(17);
  }

  NPowerTable = 0;
  do
  {
    if (fscanf(fd, " %lg %lg ", &k, &p) == 2)
      NPowerTable++;
    else
      break;
  } while (1);

  fclose(fd);

  if (ThisTask == 0)
  {
    printf("found %d pairs of values in input spectrum table\n", NPowerTable);
    fflush(stdout);
  }

  PowerTable = malloc(NPowerTable * sizeof(struct pow_table));

  sprintf(buf, FileWithInputSpectrum);

  if (!(fd = fopen(buf, "r")))
  {
    printf("can't read input spectrum in file '%s' on task %d\n", buf, ThisTask);
    FatalError(18);
  }

  NPowerTable = 0;
  do
  {
    if (fscanf(fd, " %lg %lg ", &k, &p) == 2)
    {
      PowerTable[NPowerTable].logk = log10(k); // zzc modified to be log10
      PowerTable[NPowerTable].logD = log10(p);
      // printf("PK %lf N %d k %f\n", pow(10,PowerTable[NPowerTable].logD), NPowerTable, pow(10,PowerTable[NPowerTable].logk));
      NPowerTable++;
    }
    else
      break;
  } while (1);

  fclose(fd);

  qsort(PowerTable, NPowerTable, sizeof(struct pow_table), compare_logk);
}

#ifdef PARTICLENU
void read_powernu_table(void)
{
  FILE *fd;
  char buf[500];
  double k, p;

  sprintf(buf, FileWithNuSpectrum);

  if (!(fd = fopen(buf, "r")))
  {
    printf("can't read input spectrum in file '%s' on task %d\n", buf, ThisTask);
    FatalError(17);
  }

  NPowernuTable = 0;
  do
  {
    if (fscanf(fd, " %lg %lg ", &k, &p) == 2)
      NPowernuTable++;
    else
      break;
  } while (1);

  fclose(fd);

  if (ThisTask == 0)
  {
    printf("found %d pairs of values in input spectrum table for neutrinos\n", NPowernuTable);
    fflush(stdout);
  }

  PowernuTable = malloc(NPowernuTable * sizeof(struct pow_table));

  sprintf(buf, FileWithNuSpectrum);

  if (!(fd = fopen(buf, "r")))
  {
    printf("can't read input spectrum in file '%s' on task %d\n", buf, ThisTask);
    FatalError(18);
  }

  NPowernuTable = 0;
  do
  {
    if (fscanf(fd, " %lg %lg ", &k, &p) == 2)
    {
      PowernuTable[NPowernuTable].logk = log10(k); // zzc modified to be log10
      PowernuTable[NPowernuTable].logD = log10(p);
      // printf("PK %lf N %d k %f\n", pow(10,PowerTable[NPowerTable].logD), NPowerTable, pow(10,PowerTable[NPowerTable].logk));
      NPowernuTable++;
    }
    else
      break;
  } while (1);

  fclose(fd);

  qsort(PowernuTable, NPowernuTable, sizeof(struct pow_table), compare_logk);
}
#endif

int compare_logk(const void *a, const void *b)
{
  if (((struct pow_table *)a)->logk < (((struct pow_table *)b)->logk))
    return -1;

  if (((struct pow_table *)a)->logk > (((struct pow_table *)b)->logk))
    return +1;

  return 0;
}

void initialize_powerspectrum(void)
{
  double res;

  InitTime = 1 / (1 + Redshift);

  AA = 6.4 / ShapeGamma * (3.085678e24 / UnitLength_in_cm);
  BB = 3.0 / ShapeGamma * (3.085678e24 / UnitLength_in_cm);
  CC = 1.7 / ShapeGamma * (3.085678e24 / UnitLength_in_cm);
  nu = 1.13;

  R8 = 8 * (3.085678e24 / UnitLength_in_cm); /* 8 Mpc/h */

  if (WhichSpectrum == 2)
    read_power_table();

#ifdef PARTICLENU
  read_powernu_table();
#endif
#ifdef DIFFERENT_TRANSFER_FUNC
  Type = 1;
#endif

  if (ReNormalizeInputSpectrum == 0 && WhichSpectrum == 2)
  {
    Norm = 1.0;
    /* tabulated file is already at the initial redshift */
    Dplus = GrowthFactor(InitTime, 1.0);
  }
  else
  {
    Norm = 1.0;
    res = TopHatSigma2(R8);
    printf("res0 = %.20lf res1 = %.20lf\n", res, TopHatSigma2(R8));

    if (ThisTask == 0 && WhichSpectrum == 2)
      printf("\nNormalization of spectrum in file:  Sigma8 = %g\n", sqrt(res));

    Norm = Sigma8 * Sigma8 / res;
    printf("sigma8^2 = %f res = %.20lf norm = %f\n", Sigma8 * Sigma8, res, Norm);

    if (ThisTask == 0 && WhichSpectrum == 2)
      printf("Normalization adjusted to  Sigma8=%g   (Normfac=%g)\n\n", Sigma8, Norm);

    Dplus = GrowthFactor(InitTime, 1.0);
  }
}

double PowerSpec_Tabulated(double k)
{
  double logk, logD, P, kold, u, dlogk, Delta2;
  int binlow, binhigh, binmid;

  kold = k;

  k *= (InputSpectrum_UnitLength_in_cm / UnitLength_in_cm); /* convert to h/Mpc */

  logk = log10(k);

  if (logk < PowerTable[0].logk || logk > PowerTable[NPowerTable - 1].logk)
    return 0;

  binlow = 0;
  binhigh = NPowerTable - 1;

  while (binhigh - binlow > 1)
  {
    binmid = (binhigh + binlow) / 2;
    if (logk < PowerTable[binmid].logk)
      binhigh = binmid;
    else
      binlow = binmid;
  }

  dlogk = PowerTable[binhigh].logk - PowerTable[binlow].logk;

  if (dlogk == 0)
    FatalError(777);

  u = (logk - PowerTable[binlow].logk) / dlogk;

  logD = (1 - u) * PowerTable[binlow].logD + u * PowerTable[binhigh].logD;

  Delta2 = pow(10.0, logD);

  P = Delta2 * 1e9 / (8. * PI * PI * PI); // zzc: no need to divide k^3

  return P;
}

#ifdef PARTICLENU
double PowerSpecNu_Tabulated(double k)
{
  double logk, logD, P, kold, u, dlogk, Delta2;
  int binlow, binhigh, binmid;

  kold = k;

  k *= (InputSpectrum_UnitLength_in_cm / UnitLength_in_cm); /* convert to h/Mpc */

  logk = log10(k);

  if (logk < PowernuTable[0].logk || logk > PowernuTable[NPowerTable - 1].logk)
    return 0;

  binlow = 0;
  binhigh = NPowernuTable - 1;

  while (binhigh - binlow > 1)
  {
    binmid = (binhigh + binlow) / 2;
    if (logk < PowernuTable[binmid].logk)
      binhigh = binmid;
    else
      binlow = binmid;
  }

  dlogk = PowernuTable[binhigh].logk - PowernuTable[binlow].logk;

  if (dlogk == 0)
    FatalError(777);

  u = (logk - PowernuTable[binlow].logk) / dlogk;

  logD = (1 - u) * PowernuTable[binlow].logD + u * PowernuTable[binhigh].logD;

  Delta2 = pow(10.0, logD);

  P = Delta2 * 1e9 / (8. * PI * PI * PI); // zzc: no need to divide k^3

  return P;
}
#endif

double PowerSpec_Efstathiou(double k)
{
  return Norm * k / pow(1 + pow(AA * k + pow(BB * k, 1.5) + CC * CC * k * k, nu), 2 / nu);
}

double PowerSpec_EH(double k) /* Eisenstein & Hu */
{
  return Norm * k * pow(tk_eh(k), 2);
}

double tk_eh(double k) /* from Martin White */
{
  double q, theta, ommh2, a, s, gamma, L0, C0;
  double tmp;
  double omegam, ombh2, hubble;

  /* other input parameters */
  hubble = HubbleParam;

  omegam = Omega;
  ombh2 = OmegaBaryon * HubbleParam * HubbleParam;

  if (OmegaBaryon == 0)
    ombh2 = 0.04 * HubbleParam * HubbleParam;

  k *= (3.085678e24 / UnitLength_in_cm); /* convert to h/Mpc */

  theta = 2.728 / 2.7;
  ommh2 = omegam * hubble * hubble;
  s = 44.5 * log(9.83 / ommh2) / sqrt(1. + 10. * exp(0.75 * log(ombh2))) * hubble;
  a = 1. - 0.328 * log(431. * ommh2) * ombh2 / ommh2 + 0.380 * log(22.3 * ommh2) * (ombh2 / ommh2) * (ombh2 / ommh2);
  gamma = a + (1. - a) / (1. + exp(4 * log(0.43 * k * s)));
  gamma *= omegam * hubble;
  q = k * theta * theta / gamma;
  L0 = log(2. * exp(1.) + 1.8 * q);
  C0 = 14.2 + 731. / (1. + 62.5 * q);
  tmp = L0 / (L0 + C0 * q * q);
  return (tmp);
}

/*
double TopHatSigma2(double R)
{
  r_tophat = R;

  return qromb(sigma2_int, 0, 500.0 * 1 / R);	// note: 500/R is here chosen as
               integration boundary (infinity)
}


double sigma2_int(double k)
{
  double kr, kr3, kr2, w, x;

  kr = r_tophat * k;
  kr2 = kr * kr;
  kr3 = kr2 * kr;

  if(kr < 1e-8)
    return 0;

  w = 3 * (sin(kr) / kr3 - cos(kr) / kr2);
  x = 4 * PI * k * k * w * w * PowerSpec(k);

  return x;
}*/

double TopHatSigma2(double R)
{
  double result, abserr;
  gsl_integration_workspace *workspace;
  gsl_function F;
  workspace = gsl_integration_workspace_alloc(WORKSIZE);

  F.function = &sigma2_int;

  r_tophat = R;

  gsl_integration_qag(&F, 0, 500.0 * 1 / R, 0, 1.0e-6, WORKSIZE, GSL_INTEG_GAUSS41, workspace, &result, &abserr);

  gsl_integration_workspace_free(workspace);
  // printf("result = %.20lf r = %f norm = %f\n", result, R, Norm);
  return result;

  /* note: 500/R is here chosen as (effectively) infinity integration boundary */
}

double sigma2_int(double k)
{
  double kr, kr3, kr2, w, x;

  kr = r_tophat * k;
  kr2 = kr * kr;
  kr3 = kr2 * kr;

  if (kr < 1e-8)
    return 0;

  w = 3 * (sin(kr) / kr3 - cos(kr) / kr2);
  x = 4 * PI * k * k * w * w * PowerSpec(k);
  // printf("sigma2_int Pk = %.12lf k = %lf\n", PowerSpec(k, 1), k);
  return x;
}

double GrowthFactor(double astart, double aend)
{
  return growth(aend) / growth(astart);
}

/*
double growth(double a)
{
  double hubble_a;
    double roneu;
    roneu = neutrino_integration(a, mass_nu_expan, xi_expan);
    printf("check growth 2 %f\n", roneu);
  hubble_a = sqrt(Omega / (a * a * a) + (1 - Omega - OmegaLambda - Omega_nu0_expan) / (a * a) + OmegaLambda + roneu);

  return hubble_a * qromb(growth_int, 0, a);
}


double growth_int(double a)
{
    double roneu;
    roneu = neutrino_integration(a, mass_nu_expan, xi_expan);
    printf("check growth int 2 %f\n", roneu);
    return pow(a / (Omega2 + (1 - Omega2 - OmegaLambda - Omega_nu0_expan) * a + OmegaLambda * a * a * a + roneu * a * a * a), 1.5);
}*/

double growth(double a)
{
  double hubble_a;
  double roneu;

  roneu = 0.0;
  int i;
  if (expan_on == 0)
  {
#ifdef STERILE
    roneu = neutrino_integration(a, 0.0, 0.0, -1) * (NNeutrino + Neff - 4.046);
#else
    roneu = neutrino_integration(a, 0.0, 0.0, -1) * NNeutrino;
#endif
  }
  if (expan_on == 1)
  {
    for (i = 0; i < NNeutrino; i++)
    {
      roneu += neutrino_integration(a, Mass[i], Xi[i], i);
    }
  }

  // hubble_a = sqrt(Omega / (a * a * a) + (1 - Omega - OmegaLambda - Omega_nu0_expan) / (a * a) + OmegaLambda + roneu);
  hubble_a = sqrt(Omega2 / (a * a * a) + (1 - Omega2 - OmegaLambda - Omega_nu0_expan) / (a * a) + OmegaLambda + roneu);

  double result, abserr;
  gsl_integration_workspace *workspace;
  gsl_function F;

  workspace = gsl_integration_workspace_alloc(WORKSIZE);

  F.function = &growth_int;

  gsl_integration_qag(&F, 0, a, 0, 1.0e-8, WORKSIZE, GSL_INTEG_GAUSS41, workspace, &result, &abserr);

  gsl_integration_workspace_free(workspace);

  return hubble_a * result;
}

double growth_int(double a)
{
  double roneu, x;

  roneu = 0.0;
  int i;
  if (expan_on == 0)
  {
#ifdef STERILE
    roneu = neutrino_integration(a, 0.0, 0.0, -1) * (NNeutrino + Neff - 4.046);
#else
    roneu = neutrino_integration(a, 0.0, 0.0, -1) * NNeutrino;
#endif
  }
  if (expan_on == 1)
  {
    for (i = 0; i < NNeutrino; i++)
    {
      roneu += neutrino_integration(a, Mass[i], Xi[i], i);
    }
  }

  return pow(a / (Omega2 + (1 - Omega2 - OmegaLambda - Omega_nu0_expan) * a + OmegaLambda * a * a * a + roneu * a * a * a), 1.5);
}

double F_Omega(double a)
{
  double omega_a;
  double roneu;

  roneu = 0.0;
  int i;
  if (expan_on == 0)
  {
#ifdef STERILE
    roneu = neutrino_integration(a, 0.0, 0.0, -1) * (NNeutrino + Neff - 4.046);
#else
    roneu = neutrino_integration(a, 0.0, 0.0, -1) * NNeutrino;
#endif
  }
  if (expan_on == 1)
  {
    for (i = 0; i < NNeutrino; i++)
    {
      roneu += neutrino_integration(a, Mass[i], Xi[i], i);
    }
  }

  omega_a = Omega2 / (Omega2 + a * (1 - Omega2 - OmegaLambda - Omega_nu0_expan) + a * a * a * OmegaLambda + a * a * a * roneu);
  // omega_a = Omega / (Omega + a * (1 - Omega - OmegaLambda- Omega_nu0_expan) + a * a * a * OmegaLambda + a * a * a * roneu);

  return pow(omega_a, 0.6);
}

double F2_Omega(double a)
{
  double omega_a;
  double roneu;

  roneu = 0.0;
  int i;
  if (expan_on == 0)
  {
#ifdef STERILE
    roneu = neutrino_integration(a, 0.0, 0.0, -1) * (NNeutrino + Neff - 4.046);
#else
    roneu = neutrino_integration(a, 0.0, 0.0, -1) * NNeutrino;
#endif
  }
  if (expan_on == 1)
  {
    for (i = 0; i < NNeutrino; i++)
    {
      roneu += neutrino_integration(a, Mass[i], Xi[i], i);
    }
  }

  omega_a = Omega2 / (Omega2 + a * (1 - Omega2 - OmegaLambda - Omega_nu0_expan) + a * a * a * OmegaLambda + a * a * a * roneu);
  // omega_a = Omega / (Omega + a * (1 - Omega - OmegaLambda- Omega_nu0_expan) + a * a * a * OmegaLambda + a * a * a * roneu);

  return 2 * pow(omega_a, 4. / 7.);
}

double growth_nu(double a, double k, double Omega_m, double Omega_nu0_frstr_0)
{
  double fcb, fnu, pcb, yfs, D1, q, theta, Dcbnu, kprime; // From Eisenstein Hu 9710252

  // printf("kprime %f k %f unit %f\n", kprime, k, (3.085678e24 / UnitLength_in_cm));
  theta = 2.7250 / 2.7;
  q = k * theta * theta / ((Omega_m)*HubbleParam * HubbleParam);

  /* fcb = Omega_m / (Omega_m + Omega_nu0_frstr_0);
    fnu = Omega_nu0_frstr_0 / (Omega_m + Omega_nu0_frstr_0);*/

  fcb = (Omega_m - Omega_nu0_frstr_0) / (Omega_m);
  fnu = Omega_nu0_frstr_0 / (Omega_m);

  pcb = 0.25 * (5. - sqrt(1. + 24. * fcb));
  yfs = 17.2 * fnu * (1. + 0.488 * pow(fnu, -7. / 6.)) * (3. * q / fnu) * (3. * q / fnu);
  // D1 = growth(a);
  if (a == 1.)
  {
    D1 = D11;
  }
  if (a == InitTime)
  {
    D1 = D10;
  }

  if (a != 1. && a != InitTime)
  {
    printf("your stupid bet is done bro\n");
  }
  Dcbnu = pow((pow(fcb, 0.7 / pcb) + pow(D1 / (1. + yfs), 0.7)), pcb / 0.7) * pow(D1, 1. - pcb);

  // printf("k %f, Dcbnu %f a %f fcb %f pcb %f D1 %f yfs %f\n", k, Dcbnu, a, fcb, pcb, D1, yfs);
  return Dcbnu;
}

/*  Here comes the stuff to compute the thermal WDM velocity distribution */

#define LENGTH_FERMI_DIRAC_TABLE 2000
#define MAX_FERMI_DIRAC 20.0

double fermi_dirac_vel[LENGTH_FERMI_DIRAC_TABLE];
double fermi_dirac_cumprob[LENGTH_FERMI_DIRAC_TABLE];

double WDM_V0 = 0;

double fermi_dirac_kernel(double x)
{
  return x * x / (exp(x) + 1);
}

void fermi_dirac_init(void)
{
  int i;

  for (i = 0; i < LENGTH_FERMI_DIRAC_TABLE; i++)
  {
    fermi_dirac_vel[i] = MAX_FERMI_DIRAC * i / (LENGTH_FERMI_DIRAC_TABLE - 1.0);
    fermi_dirac_cumprob[i] = qromb(fermi_dirac_kernel, 0, fermi_dirac_vel[i]);
  }

  for (i = 0; i < LENGTH_FERMI_DIRAC_TABLE; i++)
    fermi_dirac_cumprob[i] /= fermi_dirac_cumprob[LENGTH_FERMI_DIRAC_TABLE - 1];

  WDM_V0 = 0.012 * (1 + Redshift) * pow((Omega - OmegaBaryon) / 0.3, 1.0 / 3) * pow(HubbleParam / 0.65, 2.0 / 3) * pow(1.0 / WDM_PartMass_in_kev, 4.0 / 3);

  if (ThisTask == 0)
    printf("\nWarm dark matter rms velocity dispersion at starting redshift = %g km/sec\n\n",
           3.59714 * WDM_V0);

  WDM_V0 *= 1.0e5 / UnitVelocity_in_cm_per_s;

  /* convert from peculiar velocity to gadget's cosmological velocity */
  WDM_V0 *= sqrt(1 + Redshift);
}

double get_fermi_dirac_vel(void)
{
  int i;
  double p, u;

  p = drand48();
  i = 0;

  while (i < LENGTH_FERMI_DIRAC_TABLE - 2)
    if (p > fermi_dirac_cumprob[i + 1])
      i++;
    else
      break;

  u = (p - fermi_dirac_cumprob[i]) / (fermi_dirac_cumprob[i + 1] - fermi_dirac_cumprob[i]);

  return fermi_dirac_vel[i] * (1 - u) + fermi_dirac_vel[i + 1] * u;
}

void add_WDM_thermal_speeds(float *vel)
{
  double v, phi, theta, vx, vy, vz;

  if (WDM_V0 == 0)
    fermi_dirac_init();

  v = WDM_V0 * get_fermi_dirac_vel();

  phi = 2 * M_PI * drand48();
  theta = acos(2 * drand48() - 1);

  vx = v * sin(theta) * cos(phi);
  vy = v * sin(theta) * sin(phi);
  vz = v * cos(theta);

  vel[0] += vx;
  vel[1] += vy;
  vel[2] += vz;
}

void add_nu_thermal_speeds(float *vel)
{
  double v, phi, theta, vx, vy, vz;

  if (NU_V0 == 0)
    fermi_dirac_nu_init();

  v = NU_V0 * get_fermi_dirac_vel_nu();

  phi = 2 * M_PI * drand48();
  theta = acos(2 * drand48() - 1);

  vx = v * sin(theta) * cos(phi);
  vy = v * sin(theta) * sin(phi);
  vz = v * cos(theta);

  vel[0] += vx;
  vel[1] += vy;
  vel[2] += vz;
}

double get_fermi_dirac_vel_nu(void)
{
  int i;
  double p, u;

  p = drand48();
  i = 0;

  while (i < LENGTH_FERMI_DIRAC_TABLE - 2)
    if (p > fermi_dirac_cumprob[i + 1])
      i++;
    else
      break;

  u = (p - fermi_dirac_cumprob[i]) / (fermi_dirac_cumprob[i + 1] - fermi_dirac_cumprob[i]);

  return fermi_dirac_vel[i] * (1 - u) + fermi_dirac_vel[i + 1] * u;
}

void fermi_dirac_init(void)
{
  int i;

  for (i = 0; i < LENGTH_FERMI_DIRAC_TABLE; i++)
  {
    fermi_dirac_vel[i] = MAX_FERMI_DIRAC * i / (LENGTH_FERMI_DIRAC_TABLE - 1.0);
    fermi_dirac_cumprob[i] = qromb(fermi_dirac_kernel, 0, fermi_dirac_vel[i]);
  }

  for (i = 0; i < LENGTH_FERMI_DIRAC_TABLE; i++)
    fermi_dirac_cumprob[i] /= fermi_dirac_cumprob[LENGTH_FERMI_DIRAC_TABLE - 1];

  NU_V0 = 0.012 * (1 + Redshift) * pow((Omega - OmegaBaryon) / 0.3, 1.0 / 3) * pow(HubbleParam / 0.65, 2.0 / 3) * pow(1.0 / WDM_PartMass_in_kev, 4.0 / 3);

  if (ThisTask == 0)
    printf("\nneutrino rms velocity dispersion at starting redshift = %g km/sec\n\n",
           3.59714 * WDM_V0);

  NU_V0 *= 1.0e5 / UnitVelocity_in_cm_per_s;

  /* convert from peculiar velocity to gadget's cosmological velocity */
  NU_V0 *= sqrt(1 + Redshift);
}