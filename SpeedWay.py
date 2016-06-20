#!/usr/bin/python

##imports:
import sys
import subprocess
import os
import numpy as np
import glob
import time
import pickle
import scipy.stats as sst

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


from itertools import product
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import minimize

from minimization import Minimize as MoogMe_Minimize


sys.path.append('tmcalc_cython/')
import tmcalc_module as tm
file_lines='input_list.dat'
moog_driver='batch.par'
N_MOOG_SLIPLINES = 5

## My functions:

def get_tmcalc_teff_feh(filename_ares):
  (teff,erteff,erteff2,erteff3,nout,nindout) = tm.get_temperature_py('tmcalc_cython/ratios_list.dat', filename_ares)
  (fehout, erfehout, nout) = tm.get_feh_py('tmcalc_cython/feh_calib_lines.dat', filename_ares, teff, erteff, erteff2, erteff3)
  return (teff,fehout)


def interpolator(coords, data, point) :
  """
  >>> point = np.array([12.3,-4.2, 500.5, 2.5])
  >>> interpolator((lats, lons, alts, time), data, point)
  http://stackoverflow.com/questions/14119892/python-4d-linear-interpolation-on-a-rectangular-grid
  """
  dims = len(point)
  indices = []
  sub_coords = []
  for j in xrange(dims) :
    idx = np.digitize([point[j]], coords[j])[0]
    indices += [[idx - 1, idx]]
    sub_coords += [coords[j][indices[-1]]]
  indices = np.array([j for j in product(*indices)])
  sub_coords = np.array([j for j in product(*sub_coords)])
  sub_data = data[list(np.swapaxes(indices, 0, 1))]
  li = LinearNDInterpolator(sub_coords, sub_data)
  return li([point])[0]

def get_interpolated_ew(teff,logg,feh,vtur, teffv, loggv, fehv, vturv, ewdatai):
  """
  Get a point (EW) interpolated given the parameters for the models
  """
  point = np.array([teff, logg, feh, vtur])
  ewint = interpolator((teffv, loggv, fehv, vturv), ewdatai, point)
  return ewintv


def get_interpolated_ews(teff,logg,feh,vtur, teffv, loggv, fehv, vturv, ewdata):
  """
  Get the list of EWs interpolated given the parameters for the models
  """
  ewintv = [get_interpolated_ew(teff,logg,feh,vtur, teffv, loggv, fehv, vturv, ewdata[:,:,:,:,i]) for i in range(ewdata.shape[-1])]
  return np.array(ewintv)



# Evaluation functions

def func_min(x, teffv, loggv, fehv, vturv, ewdata, ew_obs):
  """
  The Function for the minimization.
  This one just use the simple chisquare
  """
  ewints = get_interpolated_ews(x[0],x[1],x[2],x[3], teffv, loggv, fehv, vturv, ewdata)
  chi, p = sst.chisquare(ewints, f_exp=ew_obs)
  return chi


def func_eval(llv, epv, ionv, ew_model, ew_obs):
  """
  Evaluation function based on the slopes of EP and RW
  and differences between FeI and FeII
  """
  indexes_feh1 = np.where(ionv == 0)[0]
  indexes_feh2 = np.where(ionv == 1)[0]
  x1 = epv[indexes_feh1]
  x2 = np.log(ew_model[indexes_feh1]/llv[indexes_feh1])
  y = ew_obs[indexes_feh1]-ew_model[indexes_feh1]
  slope1, intercept, r_value, p_value, std_err = sst.linregress(x1, y/ew_model[indexes_feh1])
  slope2, intercept, r_value, p_value, std_err = sst.linregress(x2, y/ew_model[indexes_feh1])
#  slope1, intercept = np.polyfit(x1, y, 1)
#  slope2, intercept = np.polyfit(x2, y, 1)
  diff_feh = np.mean(y) - np.mean(ew_obs[indexes_feh2]-ew_model[indexes_feh2])

#  func_val = 61.25 * slope1**2. + 8.45 * slope2**2. + diff_feh**2.
#  func_val = 80. * slope1**2. + 10. * slope2**2. + diff_feh**2.
#  func_val = 80. * slope1**2. + 40. * slope2**2. + diff_feh**2.
  func_val = 80. * slope1**2. + 40. * slope2**2. + diff_feh**2.
  return func_val

def func_eval_fast(x1, x2, yI, yII):
  slope1, intercept, r_value, p_value, std_err = sst.linregress(x1, yI)
  slope2, intercept, r_value, p_value, std_err = sst.linregress(x2, yI)
#  slope1, intercept = np.polyfit(x1, y, 1)
#  slope2, intercept = np.polyfit(x2, y, 1)
  diff_feh = np.mean(yI) - np.mean(yII)
  func_val = 61.25 * slope1**2. + 8.45 * slope2**2. + diff_feh**2.
#  func_val = 10 * slope1+ 40* slope2 + diff_feh
  return func_val

def func_min_physics(x, teffv, loggv, fehv, vturv, ewdata, llv, epv, ionv, ew_obs):
  """
  The Function for the minimization.
  This one resambe
  """
  ew_model = get_interpolated_ews(x[0],x[1],x[2],x[3], teffv, loggv, fehv, vturv, ewdata)
  fval = func_eval(llv, epv, ionv, ew_model, ew_obs)
  print x, fval
  return fval

def dEW(EW, SNR, dl=0.1):
  """
  Calculate the error on EW from Caryl 1988
  """
  # EW and dl have to be the same unit
  return 1.6*np.sqrt(dl*EW*1000)/SNR


def noise(EW, SNR, dl=0.1):
  """
  Calculate and extract a value from a normal distribution with
  center at a give EW, and with the width from the equation above
  """
  sigma = dEW(EW, SNR, dl)
  while True:
      newEW = np.random.normal(loc=EW, scale=sigma)
      if newEW > 0:
          return newEW


def read_ares_file_old(path,fileares):
  """
  Read the ares old output file (without error column) into a numpy array
  with the respective names in the columns
  """
  data = np.loadtxt(path+fileares, dtype={'names': ('lambda_rest', 'ngauss', 'depth', \
    'fwhm', 'ew', 'c1','c2','c3' ),'formats': ('f4', 'f4', 'f4', 'f4','f4', 'f4','f4','f4')})
  return data


def read_ares_file(path,fileares):
  """
  Read the ares new output file (with ew error column) into a numpy array
  with the respective names in the columns
  """
  data = np.loadtxt(path+fileares, dtype={'names': ('lambda_rest', 'ngauss', 'depth', \
    'fwhm', 'ew', 'ew_er','c1','c2','c3' ),'formats': ('f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4','f4')})
  return data


def test_plot(y,x,Z):
  """
  Color plot to see chisquare function on 2D.
  """
#  NOTE the axis exchange
  X, Y = np.meshgrid(x, y)
  fig, ax1 = plt.subplots(figsize=(10,6))
#  im = plt.imshow(Z, interpolation='bilinear', cmap=cm.RdYlGn,
  im = plt.imshow(Z, interpolation='bilinear', cmap=cm.spectral,
                  origin='lower', aspect='auto', extent=[0,len(x),0,len(y)],
                  vmax=abs(Z).max(), vmin=Z.min())
#                  norm=LogNorm())
#  im = plt.imshow(Z, extent=[0,len(x),0,len(y)])
  if x[0] > 100:
    strx = ["%4d" % (value) for value in x]
  else:
    strx = [str(value) for value in x]
  stry = [str(value) for value in y]
  ax1.set_xticks(range(len(x)))
  ax1.set_yticks(range(len(y)))
  xtickNames = plt.setp(ax1, xticklabels=strx)
  ytickNames = plt.setp(ax1, yticklabels=stry)
#  plt.setp(xtickNames)
#  plt.setp(ytickNames)

#  im = plt.imshow(Z,aspect='auto',extent=[x.min(),x.max(),y.min(),y.max()])
  plt.colorbar()
  plt.show()



def compute_quisquare_mat(teffv, loggv, fehv, vturv, llv, epv, loggfv, ewdata, ew_test_data):
  """
  Compute the chisquare mat for all the grid
  """
  chi_mat = np.empty([len(teffv), len(loggv), len(fehv), len(vturv)])
  p_mat = np.empty([len(teffv), len(loggv), len(fehv), len(vturv)])
#  fileout = open('chi_file.temp','w')
  for i in range(len(teffv)):
    for j in range(len(loggv)):
      for k in range(len(fehv)):
        for l in range(len(vturv)):
          chi_mat[i,j,k,l],p_mat[i,j,k,l] = sst.chisquare(ewdata[i,j,k,l,:], f_exp=ew_test_data)
#          fileout.write("%4d %6.2f %6.2f %6.2f %15.4f %15.4f\n" % (teffv[i],loggv[j],fehv[k],vturv[l],chi_mat[i,j,k,l],p_mat[i,j,k,l]))
#  fileout.close()
  return (chi_mat, p_mat)


def compute_funcphy_mat(teffv, loggv, fehv, vturv, llv, epv, loggfv, ionv, ewdata, ew_test_data):
  """
  Compute the evaluation function mat for all the grid
  using the evaluation function
  """
  funcphy_mat = np.empty([len(teffv), len(loggv), len(fehv), len(vturv)])
#  fileout = open('chi_file.temp','w')
#  for i in range(len(teffv)):
  for i in range(len(teffv)):
    print i, len(teffv)
    for j in range(len(loggv)):
      for k in range(len(fehv)):
        for l in range(len(vturv)):
          funcphy_mat[i,j,k,l] = func_eval(llv, epv, ionv, ewdata[i,l,k,l], ew_test_data)
#          fileout.write("%4d %6.2f %6.2f %6.2f %15.4f %15.4f\n" % (teffv[i],loggv[j],fehv[k],vturv[l],chi_mat[i,j,k,l],p_mat[i,j,k,l]))
#  fileout.close()
  return funcphy_mat



def compute_funcphy_mat_fast(teffv, loggv, fehv, vturv, llv, epv, loggfv, ionv, ewdata, ew_test_data):
  """
  Compute the evaluation function mat for all the grid
  using a fast? function evaluation
  """
  indexes_feh1 = np.where(ionv == 0)[0]
  indexes_feh2 = np.where(ionv == 1)[0]
  x1 = epv[indexes_feh1]
  funcphy_mat = np.empty([len(teffv), len(loggv), len(fehv), len(vturv)])

#  fileout = open('chi_file.temp','w')
#  for i in range(len(teffv)):
  for i in range(len(teffv)):
    print i, len(teffv)
    for j in range(len(loggv)):
      for k in range(len(fehv)):
        for l in range(len(vturv)):
          ew_model = ewdata[i,l,k,l,:]
          x2 = np.log(ew_model[indexes_feh1]/llv[indexes_feh1])
          yI = ew_test_data[indexes_feh1]-ew_model[indexes_feh1]
          yII = ew_test_data[indexes_feh2]-ew_model[indexes_feh2]
          funcphy_mat[i,j,k,l] = func_eval_fast(x1, x2, yI, yII)
#          fileout.write("%4d %6.2f %6.2f %6.2f %15.4f %15.4f\n" % (teffv[i],loggv[j],fehv[k],vturv[l],chi_mat[i,j,k,l],p_mat[i,j,k,l]))
#  fileout.close()
  return funcphy_mat

def compute_funcphy_mat2(teffv, loggv, fehv, vturv, llv, epv, loggfv, ionv, ewdata, ew_test_data):
  """
  Compute the evaluation function value mat for all the grid
  """
  funcphy_mat = np.empty([len(teffv), len(loggv), len(fehv), len(vturv)])
#  fileout = open('chi_file.temp','w')
#  for i in range(len(teffv)):
  for i in range(len(teffv))[23:24]:
    print i, len(teffv)
    for j in range(len(loggv))[0:1]:
      print j, len(loggv)
      for k in range(len(fehv))[0:1]:
        print k, len(fehv)
        for l in range(len(vturv)):
          print l, len(vturv)
          print teffv[i], loggv[j], fehv[k], vturv[l]
          funcphy_mat[i,j,k,l] = func_eval(llv, epv, ionv, ewdata[i,j,k,l], ew_test_data)
#          fileout.write("%4d %6.2f %6.2f %6.2f %15.4f %15.4f\n" % (teffv[i],loggv[j],fehv[k],vturv[l],chi_mat[i,j,k,l],p_mat[i,j,k,l]))
#  fileout.close()
  return funcphy_mat


def compute_funcphy_mat_local(teffv, loggv, fehv, vturv, llv, epv, loggfv, ionv, ewdata, ew_test_data, teff_guess, feh_guess):
  """
  Compute the evaluation function value mat
  for some local points of the grid around initial guess for
  teff and feh
  """
  dt = 300
  dl = 0.8
  df = 0.2
  dv = 1.
  tg = teff_guess
  fg = feh_guess
  lg = 4.0
  vg = 1.2

# Ramirez 2013
#  vtur = 1.163 + 7.808e-4 * (teff - 5800) - 0.494*(logg - 4.30) - 0.05*feh
# Tsantaki 2013
#  vg = 6.932e-4*tg - 0.348*lg - 1.437

  funcphy_mat = np.zeros([len(teffv), len(loggv), len(fehv), len(vturv)]) + 10000
#  fileout = open('chi_file.temp','w')
#  for i in range(len(teffv)):
  for i in range(len(teffv)):
    print i, len(teffv)
    for j in range(len(loggv)):
      for k in range(len(fehv)):
        for l in range(len(vturv)):
          if abs(teffv[i] - tg) < dt and abs(loggv[j] - lg) < dl and abs(fehv[k] - fg) < df and abs(vturv[l] - vg) < dv :
            #print teffv[i], loggv[j], fehv[k], vturv[l]
            funcphy_mat[i,j,k,l] = func_eval(llv, epv, ionv, ewdata[i,j,k,l], ew_test_data)
            print teffv[i], loggv[j], fehv[k], vturv[l], funcphy_mat[i,j,k,l]
#          fileout.write("%4d %6.2f %6.2f %6.2f %15.4f %15.4f\n" % (teffv[i],loggv[j],fehv[k],vturv[l],chi_mat[i,j,k,l],p_mat[i,j,k,l]))
#  fileout.close()
  return funcphy_mat



def get_n_min(nmin, teffv, loggv, fehv, vturv, llv, epv, loggfv, ewdata, chi_mat, p_mat):
  """
  Get the point in the grid with the least chisquare, and print the nmin minumum points
  """
  print chi_mat.min()
  print chi_mat.argmin()
  print np.unravel_index(chi_mat.argmin(), chi_mat.shape)
#  nmin = 15
  minv = np.argsort(chi_mat.ravel())[:nmin]
  (imin, jmin, kmin, lmin) =  np.unravel_index(minv, chi_mat.shape)
  for i in range(nmin):
    print teffv[imin[i]], loggv[jmin[i]], fehv[kmin[i]], vturv[lmin[i]], chi_mat[imin[i], jmin[i], kmin[i], lmin[i]], p_mat[imin[i], jmin[i], kmin[i], lmin[i]]
  return teffv[imin[0]], loggv[jmin[0]], fehv[kmin[0]], vturv[lmin[0]], chi_mat[imin[0], jmin[0], kmin[0], lmin[0]], p_mat[imin[0], jmin[0], kmin[0], lmin[0]]

def get_n_min_average(nmin, teffv, loggv, fehv, vturv, llv, epv, loggfv, ewdata, chi_mat, p_mat):
  """
  Get the weighted averaged point in the grid with the least chisquare,
  and print the nmin minumum points
  """
  print chi_mat.min()
  print chi_mat.argmin()
  print np.unravel_index(chi_mat.argmin(), chi_mat.shape)
#  nmin = 15
  minv = np.argsort(chi_mat.ravel())[:nmin]
  (imin, jmin, kmin, lmin) =  np.unravel_index(minv, chi_mat.shape)
  teff_mean =  np.average(teffv[imin[0:nmin]], weights=1./chi_mat[imin[0:nmin], jmin[0:nmin], kmin[0:nmin], lmin[0:nmin]]**2)
  logg_mean =  np.average(loggv[jmin[0:nmin]], weights=1./chi_mat[imin[0:nmin], jmin[0:nmin], kmin[0:nmin], lmin[0:nmin]]**2)
  feh_mean =  np.average(fehv[kmin[0:nmin]], weights=1./chi_mat[imin[0:nmin], jmin[0:nmin], kmin[0:nmin], lmin[0:nmin]]**2)
  vtur_mean =  np.average(vturv[lmin[0:nmin]], weights=1./chi_mat[imin[0:nmin], jmin[0:nmin], kmin[0:nmin], lmin[0:nmin]]**2)
  for i in range(nmin):
    print teffv[imin[i]], loggv[jmin[i]], fehv[kmin[i]], vturv[lmin[i]], chi_mat[imin[i], jmin[i], kmin[i], lmin[i]], p_mat[imin[i], jmin[i], kmin[i], lmin[i]]
  print teff_mean, logg_mean, feh_mean, vtur_mean
  return teff_mean, logg_mean, feh_mean, vtur_mean




def get_indexes_clean_outliers(ew_model, ew_obs, sigma):
  """
  get indexes without outliers. Using relative comparison of the ews
  """
  rel_diff = (ew_obs - ew_model)/ew_model
  std_dev = np.std(rel_diff)
  mean_val = np.mean(rel_diff)
  indexes = np.where(abs(rel_diff - mean_val) < sigma * std_dev)
  return indexes[0]


def plot_physical_dependences(llv, epv, ew_model, ew_obs):
  """
  Plot the delta EW vs.  EP and RW
  """

  fig = plt.figure(figsize=(9,12),frameon=False)

  # Make the plot FeI vs EP

  ax1 = plt.axes([.1, .7, .8, .2])
  ax1.scatter(epv,(ew_obs-ew_model)/ew_model,s=25, c='k', marker='o')
  ax1.set_xlabel('EP (ev)')
  ax1.set_ylabel('Delta EWs')
  slope, intercept, r_value, p_value, std_err = sst.linregress(epv,(ew_obs-ew_model)/ew_model)
#  a,b,sig_a,sig_b = lsq(epv,ew_obs - ew_model)
  xfit = [min(epv) - 0.1, max(epv) + 0.1]
  fit = [slope * i + intercept for i in xfit]
  ax1.plot(xfit,fit,c='k')
  ax1.set_ylim([-1., 1.])
  ax1.text(0.05, 0.85, 'slope = ' + str(round(slope,4)), transform = ax1.transAxes)
  ax1.text(0.55, 0.85, 'sigma = ' + str(np.std((ew_obs-ew_model)/ew_model)), transform = ax1.transAxes)

  # Make the plot FeI vs log (RW)

  ax2 = plt.axes([.1, .4, .8, .2])
  ax2.scatter(np.log(ew_model/llv),(ew_obs-ew_model)/ew_model,s=25, c='k', marker='o')
  ax2.set_xlabel('log RW')
  ax2.set_ylabel('Delta EWs')
  slope, intercept, r_value, p_value, std_err = sst.linregress(np.log(ew_model/llv),(ew_obs-ew_model)/ew_model)
#  a,b,sig_a,sig_b = lsq(epv,ew_obs - ew_model)
  xfit = [min(np.log(ew_model/llv)) - 0.1, max(np.log(ew_model/llv)) + 0.1]
  fit = [slope * i + intercept for i in xfit]
  ax2.set_ylim([-1., 1.])
  ax2.plot(xfit,fit,c='k')
  ax2.text(0.05, 0.85, 'slope = ' + str(round(slope,4)), transform = ax2.transAxes)

  # Make the plot FeII vs EP

#  ax = plt.axes([.1, .1, .3, .2])
#  plt.scatter(ep2,abund2,s=25, c='k', marker='o')
#  plt.xlabel('EP (ev)')
#  plt.ylabel('log FeII')

  plt.show(block=False)


def get_parameters_chi(file_ares, file_ews):
  """
  Getting the parameter estimation using chisquare
  """

  # Reading the binary grid
  (teffv, loggv, fehv, vturv, llv, epv, loggfv, ionv, ewdata) = read_grid(file_ews)

  # Selecting only the observed lines
  indexes, ew_test_data = get_indexes_lines(llv,file_ares)
  llv = llv[indexes]
  epv = epv[indexes]
  ionv = ionv[indexes]
  loggfv = loggfv [indexes]
  ewdata = ewdata [:,:,:,:,indexes]

  # Computing chisquares for all grid points
  chi_mat, p_mat = compute_quisquare_mat(teffv, loggv, fehv, vturv, llv, epv, loggfv, ewdata, ew_test_data)

  # Getting the weigthed average point in the grid (16 minimum points)
  teffm, loggm, fehm, vturm = get_n_min_average(16, teffv, loggv, fehv, vturv, llv, epv, loggfv, ewdata, chi_mat, p_mat)
  print "First guess for minimization: ", teffm, loggm, fehm, vturm

  # plotting physical dependencies
  ew_model = get_interpolated_ews(teffm, loggm, fehm, vturm, teffv, loggv, fehv, vturv, ewdata)
  plot_physical_dependences(llv,epv,ew_model,ew_test_data)

  # Removing 2 sigma outliers in observation
  indexes = get_indexes_clean_outliers(ew_model,ew_test_data, 2.)
  llv = llv[indexes]
  epv = epv[indexes]
  ionv = ionv[indexes]
  loggfv = loggfv[indexes]
  ewdata = ewdata[:,:,:,:,indexes]
  ew_test_data = ew_test_data[indexes]

  # recomputing chisquare for all grid points
  chi_mat, p_mat = compute_quisquare_mat(teffv, loggv, fehv, vturv, llv, epv, loggfv, ewdata, ew_test_data)

  # Getting the weigthed average point in the grid (16 minimum points)
  teffm, loggm, fehm, vturm = get_n_min_average(16, teffv, loggv, fehv, vturv, llv, epv, loggfv, ewdata, chi_mat, p_mat)
  print "First guess for minimization: ", teffm, loggm, fehm, vturm

  # plotting physical dependencies
  ew_model = get_interpolated_ews(teffm, loggm, fehm, vturm, teffv, loggv, fehv, vturv, ewdata)
  plot_physical_dependences(llv,epv,ew_model,ew_test_data)

  # Starting minimization to look for best chisquare with interpolation
  print "First guess for minimization: ", teffm, loggm, fehm, vturm
  x0 = np.array([teffm, loggm, fehm, vturm])

  t0 = time.time()
#  res = minimize(func_min, x0, method='TNC', bounds =[(5000,6000),(4.0,4.45),(-0.5,0.5),(0.5,1.5)],
  res = minimize(func_min, x0, method='L-BFGS-B', bounds =[(4200,6500),(2.0,4.49),(-2,0.39),(0.01,2.9)],
                args =(teffv, loggv, fehv, vturv, ewdata, ew_test_data),#,
                options={'ftol': 0.0001, 'disp': True})

#def func_min_physics(x, teffv, loggv, fehv, vturv, ewdata, llv, epv, ionv, ew_obs):
#  res = minimize(func_min_physics, x0, method='L-BFGS-B', bounds =[(4200,6500),(2.0,4.49),(-2,0.39),(0.01,2.9)],
#                args =(teffv, loggv, fehv, vturv, ewdata, llv, epv, ionv, ew_test_data),#,
#                options={'ftol': 0.0001, 'disp': True})

  print res
  t1 = time.time()
  print "Time to minimize: %d seconds" % (t1-t0)
  teffm, loggm, fehm, vturm = res.x[0], res.x[1], res.x[2], res.x[3]

  ew_model = get_interpolated_ews(teffm, loggm, fehm, vturm, teffv, loggv, fehv, vturv, ewdata)
  plot_physical_dependences(llv,epv,ew_model,ew_test_data)

  return teffm, loggm, fehm, vturm


#---------------------------------------------------
#---------------------------------------------------
#---------------------------------------------------

def read_grid(file_grid):
  """
  Read binary grid into memory
  """
  t0 = time.time()
  inputfile = open(file_grid, 'rb')
  (teffv2, loggv2, fehv2, vturv2, llv2, epv2, loggfv2, ionv2, ewdata2) = pickle.load(inputfile)
  inputfile.close()
  t1 = time.time()
  total3 = t1-t0
  return (teffv2, loggv2, fehv2, vturv2, llv2, epv2, loggfv2, ionv2, ewdata2)


def get_indexes_lines(llv, file_ares):
  """
  Get the indexes of the lines in the grid that correspond to the observed line list
  """
  data = read_ares_file_old('',file_ares)
#  data = read_ares_file('',file_ares)
  ll_ares = data['lambda_rest']
  ew_ares = data['ew']
  ew_test_data = []
  values = np.zeros(len(llv))
  for i, line in enumerate(llv):
    res = np.where( abs(line-ll_ares) < 0.1 )
#    print res, len(res[0])
    if len(res[0]) > 0:
#      print i, res, line, ll_ares[res[0][0]], ew_ares[res[0][0]]
      values[i] = 1
      ew_test_data.append(ew_ares[res[0][0]])
  indexes = np.where(values > 0)
#  print indexes
  ew_test_data = np.array(ew_test_data)
  return indexes[0], ew_test_data





def get_parameters_physics(file_ares, file_ews):
  """
  Getting the parameter estimation using excitation and ionization balance
  """

  # Reading the binary grid
  (teffv, loggv, fehv, vturv, llv, epv, loggfv, ionv, ewdata) = read_grid(file_ews)

  # Selecting only the observed lines
  indexes, ew_test_data = get_indexes_lines(llv,file_ares)
  llv = llv[indexes]
  epv = epv[indexes]
  loggfv = loggfv [indexes]
  ionv = ionv [indexes]
  ewdata = ewdata [:,:,:,:,indexes]


  teffm, fehm = get_tmcalc_teff_feh(file_ares)

  print teffm, fehm

 # Computing min func for all grid points
  t0 = time.time()
  funcphy_mat = compute_funcphy_mat_local(teffv, loggv, fehv, vturv, llv, epv, loggfv, ionv, ewdata, ew_test_data, teffm, fehm)
#  funcphy_mat = compute_funcphy_mat_fast(teffv, loggv, fehv, vturv, llv, epv, loggfv, ionv, ewdata, ew_test_data)
  t1 = time.time()
  print "Time to compute func eval on grid: %f seconds " % (t1 - t0)


#  print funcphy_mat
  print funcphy_mat[0,5,2,5]
  teffm, loggm, fehm, vturm = get_n_min_average(16, teffv, loggv, fehv, vturv, llv, epv, loggfv, ewdata, funcphy_mat, funcphy_mat)
  print "First guess for minimization: ", teffm, loggm, fehm, vturm

  # plotting physical dependencies
  ew_model = get_interpolated_ews(teffm, loggm, fehm, vturm, teffv, loggv, fehv, vturv, ewdata)
  plot_physical_dependences(llv,epv,ew_model,ew_test_data)

  # Removing 2 sigma outliers in observation
  indexes = get_indexes_clean_outliers(ew_model,ew_test_data, 2.)
  llv = llv[indexes]
  epv = epv[indexes]
  ionv = ionv[indexes]
  loggfv = loggfv[indexes]
  ewdata = ewdata[:,:,:,:,indexes]
  ew_test_data = ew_test_data[indexes]

  # recomputing chisquare for all grid points
  funcphy_mat = compute_funcphy_mat_local(teffv, loggv, fehv, vturv, llv, epv, loggfv, ionv, ewdata, ew_test_data, teffm, fehm)

  # Getting the weigthed average point in the grid (16 minimum points)
  teffm, loggm, fehm, vturm = get_n_min_average(16, teffv, loggv, fehv, vturv, llv, epv, loggfv, ewdata, funcphy_mat, funcphy_mat)
  print "First guess for minimization: ", teffm, loggm, fehm, vturm

  # plotting physical dependencies
  ew_model = get_interpolated_ews(teffm, loggm, fehm, vturm, teffv, loggv, fehv, vturv, ewdata)
  plot_physical_dependences(llv,epv,ew_model,ew_test_data)

  # Starting minimization to look for best chisquare with interpolation
  print "First guess for minimization: ", teffm, loggm, fehm, vturm
  x0 = np.array([teffm, loggm, fehm, vturm])

  t0 = time.time()
#  res = minimize(func_min, x0, method='TNC', bounds =[(5000,6000),(4.0,4.45),(-0.5,0.5),(0.5,1.5)],
#  res = minimize(func_min, x0, method='L-BFGS-B', bounds =[(4200,6500),(2.0,4.49),(-2,0.39),(0.01,2.9)],
#                args =(teffv, loggv, fehv, vturv, ewdata, ew_test_data),#,
#                options={'ftol': 0.0001, 'disp': False})

#def func_min_physics(x, teffv, loggv, fehv, vturv, ewdata, llv, epv, ionv, ew_obs):Nelder-Mead
  res = minimize(func_min_physics, x0, method='L-BFGS-B', bounds =[(4200,6500),(2.0,4.49),(-2,0.39),(0.01,2.9)],
                args =(teffv, loggv, fehv, vturv, ewdata, llv, epv, ionv, ew_test_data),#,
                options={'ftol': 0.0001, 'disp': True})
#  res = minimize(func_min_physics, x0, method='Nelder-Mead', bounds =[(4200,6500),(2.0,4.49),(-2,0.39),(0.01,2.9)],
#                args =(teffv, loggv, fehv, vturv, ewdata, llv, epv, ionv, ew_test_data),#,
#                options={'xtol': 0.1, 'maxfev' : 200})

#  res = fmin(func_min_physics, x0, args =(teffv, loggv, fehv, vturv, ewdata, llv, epv, ionv, ew_test_data)

  print res
  t1 = time.time()
  print "Time to minimize: %d seconds" % (t1-t0)
  teffm, loggm, fehm, vturm = res.x[0], res.x[1], res.x[2], res.x[3]

  ew_model = get_interpolated_ews(teffm, loggm, fehm, vturm, teffv, loggv, fehv, vturv, ewdata)
  plot_physical_dependences(llv,epv,ew_model,ew_test_data)

  return teffm, loggm, fehm, vturm




def func_eval2(llv, epv, ionv, ew_model, ew_obs):
  """
  Evaluation function based on the slopes of EP and RW
  and differences between FeI and FeII
  """
  indexes_feh1 = np.where(ionv == 0)[0]
  indexes_feh2 = np.where(ionv == 1)[0]
  x1 = epv[indexes_feh1]
  x2 = np.log(ew_model[indexes_feh1]/llv[indexes_feh1])
  y  = ew_obs[indexes_feh1]-ew_model[indexes_feh1]
  y2 = ew_obs[indexes_feh2]-ew_model[indexes_feh2]
  slope1, intercept, r_value, p_value, std_err = sst.linregress(x1, y/ew_model[indexes_feh1])
  slope2, intercept, r_value, p_value, std_err = sst.linregress(x2, y/ew_model[indexes_feh1])
  diff_feh = np.mean(y) - np.mean(y2)
  abundances =  [np.mean(y),np.mean(y2)]
#  func_val = 80. * slope1**2. + 40. * slope2**2. + diff_feh**2.
  func_val = slope1**2 + slope2**2 + np.diff(abundances)[0]**2
  return func_val, slope1, slope2, abundances


def func_moog_grid(x, grid_data):
  """
    Inputs
    ------
    x : tuple
      tuple/list with values (teff, logg, [Fe/H], vt) in that order
  """
  teffv, loggv, fehv, vturv, llv, epv, loggfv, ionv, ew_data, ew_obs = grid_data
#  print x
#  print teffv.shape, loggv.shape
#  print fehv.shape, vturv.shape
#  print ew_data.shape
#  print fehv
  ew_model = get_interpolated_ews(x[0],x[1],x[2],x[3], teffv, loggv, fehv, vturv, ew_data)
  res, slopeEP, slopeRW, abundances = func_eval2(llv, epv, ionv, ew_model, ew_obs)
  print res, slopeRW
  return res, slopeEP, slopeRW, abundances




def get_parameters_moogme_min(file_ares, file_ews):
  # Reading the binary grid
  (teffv, loggv, fehv, vturv, llv, epv, loggfv, ionv, ewdata) = read_grid(file_ews)

  # Selecting only the observed lines
  indexes, ew_obs_data = get_indexes_lines(llv,file_ares)
  llv = llv[indexes]
  epv = epv[indexes]
  loggfv = loggfv [indexes]
  ionv = ionv [indexes]
  ewdata = ewdata [:,:,:,:,indexes]

  teffm, fehm = get_tmcalc_teff_feh(file_ares)

  grid_data = (teffv, loggv, fehv, vturv, llv, epv, loggfv, ionv, ewdata, ew_obs_data)


  parameters=[teffm, 4.00, fehm, 1.00]

  options = {'fix_teff': False,
             'fix_logg': False,
             'fix_feh' : False,
             'fix_vt'  : False,
            }
  function = MoogMe_Minimize(parameters, func_moog_grid, grid_data, **options)
  parameters, converged = function.minimize()
  print parameters
  print converged





### Main program:
def main():

  file_ews = 'data2_kur.pkl'
#  file_ews = 'data2_mar.pkl'

  file_ares = 'test_stars/HD128620.ares'

  get_parameters_moogme_min(file_ares, file_ews)

  return


  teff, logg, feh, vtur = get_parameters_physics(file_ares, file_ews)
  print "Result : Teff: %d ; logg: %5.2f ; [Fe/H]: %5.2f ; vtur: %5.2f \n" % (teff, logg, feh, vtur)
  i = raw_input("Press any key to continue...  ").split



if __name__ == "__main__":
    main()

