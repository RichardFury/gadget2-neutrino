import numpy as np
import sys

def read_transfer(fname):
  """Read the transfer function data from CLASS file"""
  tk = np.loadtxt(fname, skiprows=11)
  return tk

def read_tot_powerspec(fname):
  """Read the power spectrum from CLASS output file"""
  pk_tot = np.loadtxt(fname, skiprows=4)
  return pk_tot

def calculate_pk_nu(tk, pk):
  """Calculate the power spectrum for specific type"""
  pk_nu[:, 0] = pk[:, 0]
  pk_nu[:, 1] = pk[:, 1] * (tk[:, 20]/tk[:, 21])
  return pk_nu

if __name__ == "__main__":
  path_tk = sys.argv[1]
  path_pk = sys.argv[2]
  tk = read_tot_powerspec(path_pk)
  pk = read_transfer(path_tk)
  pk_nu = calculate_pk_nu(tk, pk)
  np.savetxt("./initial_pk_nu.txt", pk_nu)
