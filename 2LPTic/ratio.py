from itertools import tee
import numpy as np 
import sys

#base = "/ds/rui/workspace/NuGrid/de2_deduction/2LPTic/rui/Rui_twotypes"

#txt_cdm = base + "/cmbonly_m000-xi000-cdm-49-twotypes.txt"
#txt_nu = base + "/cmbonly_m000-xi000-nu-49-twotypes.txt"

#txt_ratio = "/ds/rui/workspace/Gadget_output/Rui_twotypes-mnu003-xi010/m000-xi000/ratio_cmbonly_m000-xi000-twotypes.txt"
base = str(sys.argv[1])
txt_cdm = base + "/" + str(sys.argv[2])
txt_nu = base + "/" + str(sys.argv[3])

txt_ratio = str(sys.argv[4])

k, pk_nu = np.loadtxt(txt_nu, usecols=(0, 1), unpack=True)
k, pk_cdm = np.loadtxt(txt_cdm, usecols=(0, 1), unpack=True)

ratio = pk_nu / pk_cdm

np.savetxt(txt_ratio, np.transpose([k, ratio]))

