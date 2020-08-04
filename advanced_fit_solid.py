#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from astropy.io.fits import getdata
from versatile_disk_model import model_eclipse
import dynesty
from dynesty import plotting as dyplot
from utils import inv_var_weight_avg, make_flux, map01
import os
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
from sys import argv
import pickle
nproc = int(argv[1])
from data.coords import ra, dec


def prior_transform(utheta):
    ua0, ud0, uua, uud, upi, ur1, ur2, ut, uT, uCV, uCK = utheta
    # all distances are relative to giant radius
    # all times are units of Julian days
    a0 = map01(ua0, -100.0, 100.0) # offset in ra at t0
    d0 = map01(ud0, -100.0, 100.0) # offset in dec at t0
    ua = map01(uua, -100.0, 100.0) # velocity in ra
    ud = map01(uud, -100.0, 100.0) # velocity in dec
    pi = np.exp(map01(upi, -10.0, 10.0)) # parallax
    r1 = map01(ur1, 0.0, 100.0) # semimajor axis of disk
    r2 = r1*ur2 # semiminor axis is semimajor axis time axis ratio
    t = map01(ut, 0.0, np.pi) # angle of disk relative to direction of motion
    T = map01(uT, 0.00, 0.04) # transmissivity of disk (0-4%)
    CV = np.exp(map01(uCV, -0.1, 0.1)) # flux depth scale V = CV*I
    CK = np.exp(map01(uCK, -0.1, 0.1)) # flux depth scale K = CK*I
    return a0, d0, ua, ud, pi, r1, r2, t, T, CV, CK


def lnprob(theta, model_array, flux_array, eflux_array):
    """The log-likelihood function."""
    # generate lightcurve
    _fV = 1- ((1 - model_array[0].get_lc(theta[:-2])) * theta[-2])
    _fI = model_array[1].get_lc(theta[:-2])
    _fK = 1- ((1 - model_array[2].get_lc(theta[:-2])) * theta[-1])

    # calc likelihood
    r2_V = ((flux_array[0]-_fV)/eflux_array[0])**2
    r2_I = ((flux_array[1]-_fI)/eflux_array[1])**2
    r2_K = ((flux_array[2]-_fK)/eflux_array[2])**2
    r2 = r2_V.sum() + r2_I.sum() + r2_K.sum()
    _lnprob = -(r2/2)

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

# estimated time of minimum flux
t0_est = imjdobs[np.argmax(imags)]
# estimated baseline mag
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

# selections for fitting
vfit = np.abs(vmjdobs-t0_est)<365
ifit = np.abs(imjdobs-t0_est)<365
kfit = np.abs(kmjdobs-t0_est)<365

model_kwargs = {
    'step': 0.02,
    't0': t0_est,
    'ra': ra,
    'dec': dec,
    'motion_mode': 'parallactic',
    'occulter_mode': 'solid',
}
ecl_model_V = model_eclipse(mjd_points=vmjdobs[vfit], mu=1.20, **model_kwargs)
ecl_model_I = model_eclipse(mjd_points=imjdobs[ifit], mu=1.10, **model_kwargs)
ecl_model_K = model_eclipse(mjd_points=kmjdobs[kfit], mu=1.05, **model_kwargs)

with Pool(processes=nproc) as pool:
    dsampler = dynesty.NestedSampler(lnprob, prior_transform, ndim=11, periodic=[7],
                                     pool=pool, queue_size=nproc,
                                     logl_args=[
                                         [ecl_model_V, ecl_model_I, ecl_model_K],
                                         [vflux[vfit], iflux[ifit], kflux[kfit]],
                                         [vflux_error[vfit], iflux_error[ifit], kflux_error[kfit]]
                                     ])
    dsampler.run_nested()
dres = dsampler.results

pickle.dump(dres, open("dres_advanced_solid_corner.p", "wb"))
