#!/usr/bin/env python3

import disk_in_orbit as dio

import matplotlib as mpl
mpl.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

import emcee
import corner
from astropy.io.fits import getdata

import os
os.environ["OMP_NUM_THREADS"] = "1"

from multiprocessing import Pool
from sys import argv

import pickle

nproc = int(argv[1])


def inv_var_weight_avg(x, ex):
    # compute inverse variance weighted average and error
    w = 1./(ex**2) # weights
    x_wav = np.nansum(w * x, axis=0) / np.nansum(w, axis=0)
    σ_wav = 1./np.sqrt(np.nansum(w, axis=0))
    return x_wav, σ_wav

def make_flux(mags, mag_errors, baseline_mag):
    fluxes = 10**(0.4*(baseline_mag- mags))
    flux_errors = fluxes * (10**((2*mag_errors)/5) - 1)
    return fluxes, flux_errors







def priors(x):
    if any((x<0)|(x>1)) or (x[4]>x[3]):
        return -np.inf
    else:
        return 0.0


def map01(val, lower, upper):
    return (val * (upper-lower)) + lower

def mapXs(x):
    _x = x.copy()
    if len(_x.shape)==1:
        _x = _x[np.newaxis,:]
    _x[:,0] = t0_est + map01(_x[:,0], -10.0, 10.0) # t0
    _x[:,1] = map01(_x[:,1], 0.0, 1.0) # relative velocity
    _x[:,2] = map01(_x[:,2], 0.0, 10.0) # minimum impact parameter
    _x[:,3] = map01(_x[:,3], 0.0, 10.0) # semimajor axis of disk
    _x[:,4] = map01(_x[:,4], 0.0, 10.0) # semiminor axis of disk
    _x[:,5] = map01(_x[:,5], 0.0, np.pi) # angle of disk relative to direction of orbital motion
    _x[:,6] = map01(_x[:,6], 0.0, 0.04) # transmissivity of disk (0-4%)
    if len(x.shape)==1:
        return _x[0,:]
    return _x


def lnprob(x,I,xp,yp,tp,f,ef):
    """The log-likelihood function."""
    prior = priors(x) # all x between 0 and 1
    if prior<0:
        return prior
    _x = mapXs(x)

    # generate light curve
    _f = dio.get_model_lc_disk(I, xp, yp, tp, *_x)

    # calc likelihood
    residual = ((f-_f)/ef)**2
    _lnprob = -(np.sum(residual)/2) + prior
    #print(_lnprob)
    return _lnprob


# Load I data
I = np.genfromtxt('data/OGLE.I.dat', dtype=['f8','f4','f4'], names=['tobs','mag','emag'])
imags = I["mag"]
eimags = I["emag"]
imjdobs = I["tobs"]+50000.

# Load V data
V = np.genfromtxt('data/OGLE.V.dat', dtype=['f8','f4','f4'], names=['tobs','mag','emag'])
vmags = V["mag"]
evmags = V["emag"]
vmjdobs = V["tobs"]+50000.

# Load Ks data
Ks = getdata("data/virac2_data.fits", -1, view=np.recarray)
Ks = Ks[(Ks["tileloc"]!=0) & (Ks["filter"].astype("U2")=="Ks")]
kmags = Ks["hfad_mag"]
ekmags = Ks["hfad_emag"]
kmjdobs = Ks["mjdobs"]

# estimated baseline mag
t0_est = int(imjdobs[np.argmax(imags)])
m0_i_est, em0_i_est = inv_var_weight_avg(imags[np.abs(imjdobs-t0_est)>365],
                                     eimags[np.abs(imjdobs-t0_est)>365])
m0_v_est, em0_v_est = inv_var_weight_avg(vmags[np.abs(vmjdobs-t0_est)>365],
                                         evmags[np.abs(vmjdobs-t0_est)>365])
m0_k_est, em0_k_est = inv_var_weight_avg(kmags[np.abs(kmjdobs-t0_est)>365],
                                         ekmags[np.abs(kmjdobs-t0_est)>365])

# fluxes, errors
iflux, iflux_error = make_flux(imags, eimags, m0_i_est)
vflux, vflux_error = make_flux(vmags, evmags, m0_v_est)
kflux, kflux_error = make_flux(kmags, ekmags, m0_k_est)

# mjdobs, fluxes, flux errors to use for fitting
fit_mjdobs = np.concatenate((
    #vmjdobs[np.abs(vmjdobs-t0_est)<365],
    imjdobs[np.abs(imjdobs-t0_est)<200],
    #kmjdobs[np.abs(kmjdobs-t0_est)<365],
))
fit_fluxes = np.concatenate((
    #vflux[np.abs(vmjdobs-t0_est)<365],
    iflux[np.abs(imjdobs-t0_est)<200],
    #kflux[np.abs(kmjdobs-t0_est)<365],
))
fit_flux_errors = np.concatenate((
    #vflux_error[np.abs(vmjdobs-t0_est)<365],
    iflux_error[np.abs(imjdobs-t0_est)<200],
    #kflux_error[np.abs(kmjdobs-t0_est)<365],
))

# get giant intensity map
I,xp,yp = dio.get_I(step=0.02) # step=0.02 means max model error is ~0.35 * data error

ndim, nwalkers = 7, 4096 #int(2**15)
p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
nSteps = 1024 #int(2**12)

# run the sampler
with Pool(processes=nproc) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[I,xp,yp,fit_mjdobs,fit_fluxes,fit_flux_errors], pool=pool)
    pos, prob, state = sampler.run_mcmc(p0,nSteps)


samples = sampler.chain.reshape((-1, ndim))

# save output
pickle.dump([pos,prob,state,samples], open("dio_fit_op.p", "wb"))
