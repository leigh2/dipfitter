#!/usr/bin/env python

import numpy as np


def lin_limb_dark(r_rs, mu):
    """
    Intensity as a function of radius for the linear limb darkening model:
        I(r) = ( 1 - mu*(1 - sqrt(1 - (r/rs)^2)) )  /  (1 - mu/3)  /  pi
    r_rs : radius as a fraction of stellar radius
    mu : linear limb darkening coefficient
    returns I(r)
    """
    Ir = 1 - mu * (1 - np.cos(np.arcsin(r_rs)))
    Ir[np.isnan(Ir)] = 0.0
    return Ir

def get_grid(step=0.01, full=False):
    # get grid points over which to perform occultation evaluation
    xpts, ypts = np.mgrid[-1:1:step, -1:1:step]
    if not full:
        xind = np.hypot(xpts, ypts)<1.0
        xpts, ypts = [_[xind] for _ in [xpts, ypts]]
    return xpts, ypts

def get_I(step=0.01, mu=1.05, full=False):
    # get grid positions
    xpts, ypts = get_grid(step=step, full=full)
    # get intensity of background source
    I = lin_limb_dark(np.hypot(xpts,ypts), mu)
    return I/I.sum(), xpts, ypts

def get_model_lc_disk(
    I, xpts, ypts, tpts,
    t0, vt, b, # time of conjunction, velocity, impact parameter
    r1, r2, tilt, T, # semimajor and semiminor axis, tilt angle, transmittance
    ):
    sin_t, cos_t = np.sin(tilt), np.cos(tilt)
    # current relative position
    xpos = b
    ypos = (tpts - t0) * vt
    # evaluate model
    curx, cury = xpos, ypos[:,None]# [_[:,None] for _ in [xpos,ypos]]
    x1 = -(xpts[None, :] - curx) * sin_t + (ypts[None, :] - cury) * cos_t
    x2 =  (xpts[None, :] - curx) * cos_t + (ypts[None, :] - cury) * sin_t
    _I = I[None, :] * np.where(((x1 / r1)**2 + (x2 / r2)**2) < 1, T, 1.0)
    return np.sum(_I, axis=1)
