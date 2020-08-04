#!/usr/bin/env python

import numpy as np
from astropy.io import fits
from astropy.coordinates import get_body_barycentric
from astropy.time import Time
import astropy.units as u

### position vector of earth ###
def get_R(mjd):
    # R is the position vector of Earth
    # requires an mjd (or array of)
    # returns R, a three component array describing the position vector of Earth
    # in au
    R = get_body_barycentric('earth', Time(mjd, format='mjd', scale='tcb'))
    return np.array([R.x.to(u.AU) / u.AU, R.y.to(u.AU) / u.AU, R.z.to(u.AU) / u.AU]).T

### local west unit vector ###
def get_W(a):
    # feed me ra in degrees
    a = np.radians(a)
    # local west unit vector
    return np.array([np.sin(a), -np.cos(a), 0.0])

### local north unit vector ###
def get_N(a,d):
    # feed me ra and dec in degrees
    a, d = np.radians((a,d))
    # local north unit vector
    return np.array([np.cos(a) * np.sin(d),
                     np.sin(a) * np.sin(d), # nb. Green p339 has the sign wrong here
                     -np.cos(d)])

### combined parallax parameters ###
def get_plx_params(mjd, ra, dec):
    R = get_R(mjd)
    W = get_W(ra)
    N = get_N(ra, dec)
    return R.dot(W), R.dot(N)




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


def map01(val, lower, upper):
    return (val * (upper-lower)) + lower
