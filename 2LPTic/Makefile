EXEC   = 2LPTic

OBJS   = main.o power.o allvars.o save.o read_param.o  read_glass.o  neutrino.o\
         nrsrc/nrutil.o nrsrc/qromb.o nrsrc/polint.o nrsrc/trapzd.o

INCL   = allvars.h proto.h neutrino.h nrsrc/nrutil.h  Makefile



#OPT   +=  -DPRODUCEGAS   # Set this to automatically produce gas particles 
                         # for a single DM species in the input file by interleaved by a half a grid spacing


#OPT   +=  -DMULTICOMPONENTGLASSFILE  # set this if the initial glass file contains multiple components

#OPT   +=  -DDIFFERENT_TRANSFER_FUNC  # set this if you want to implement a transfer function that depends on
                                     # particle type

OPT   +=  -DNO64BITID    # switch this on if you want normal 32-bit IDs
OPT   +=  -DCORRECT_CIC  # only switch this on if particles start from a glass (as opposed to grid)

#OPT += -DONLY_ZA # swith this on if you want ZA initial conditions (2LPT otherwise)

OPT  += -DSTERILE=3 # sterile neutrino paramter


OPTIONS =  $(OPT)

CC       =  mpicc   
OPTIMIZE =  -O3 -Wall -std=c99
GSL_INCL = -I/home/rui/library/gsl/include
GSL_LIBS = -L/home/rui/library/gsl/lib
FFTW_INCL = -I/home/rui/library/fftw2/include
FFTW_LIBS = -L/home/rui/library/fftw2/lib

MPICHLIB =  -L/home/rui/library/openmpi/lib

FFTW_LIB =  $(FFTW_LIBS) -ldrfftw_mpi -ldfftw_mpi -ldrfftw -ldfftw

LIBS   =   -lm  $(MPICHLIB)  $(FFTW_LIB)  $(GSL_LIBS)  -lgsl -lgslcblas

CFLAGS =   $(OPTIONS)  $(OPTIMIZE)  $(FFTW_INCL) $(GSL_INCL)

$(EXEC): $(OBJS) 
	$(CC) $(OPTIMIZE) $(OBJS) $(LIBS)   -o  $(EXEC)  

$(OBJS): $(INCL) 


.PHONY : clean
clean:
	rm -f $(OBJS) $(EXEC)



