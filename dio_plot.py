#!/usr/bin/env python3

import disk_in_orbit as dio

import matplotlib as mpl
mpl.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

import emcee
import corner
from astropy.io.fits import getdata

import pickle


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
    imjdobs[np.abs(imjdobs-t0_est)<365],
    #kmjdobs[np.abs(kmjdobs-t0_est)<365],
))
fit_fluxes = np.concatenate((
    #vflux[np.abs(vmjdobs-t0_est)<365],
    iflux[np.abs(imjdobs-t0_est)<365],
    #kflux[np.abs(kmjdobs-t0_est)<365],
))
fit_flux_errors = np.concatenate((
    #vflux_error[np.abs(vmjdobs-t0_est)<365],
    iflux_error[np.abs(imjdobs-t0_est)<365],
    #kflux_error[np.abs(kmjdobs-t0_est)<365],
))

pos,prob,state,samples = pickle.load(open("dio_fit_op.p", "rb"))

mapped = mapXs(samples)
bestX = pos[np.argsort(prob)[-1],:]
'''fig = corner.corner(mapped,
                    labels=[r"$t_{0}$",
                            r"$v_{t}$",
                            r"$b$",
                            r"$r_{maj}$",
                            r"$r_{min}$",
                            r"$\Theta$",
                            r"$T$",
                           ],
                    truths=mapXs(bestX),
                    label_kwargs=dict(fontsize=15),
                    show_titles=True,
                    )
plt.savefig('figs/I_corner_disk.pdf', bbox_inches='tight')
plt.close()'''

x_model = mapXs(bestX)
print(mapXs(bestX))

I,xp,yp = dio.get_I(step=0.01) # step=0.02 means max model error is ~0.35 * data error

t_model = np.linspace(imjdobs.min(), imjdobs.max(), 1000)
f_model = dio.get_model_lc_disk(I, xp, yp, t_model, *x_model)

f_plot_i = dio.get_model_lc_disk(I, xp, yp, imjdobs, *x_model)

f_plot_v = dio.get_model_lc_disk(I, xp, yp, vmjdobs, *x_model)

f_plot_k = dio.get_model_lc_disk(I, xp, yp, kmjdobs, *x_model)


fig = plt.figure(figsize=(10,5))

ax1 = plt.subplot(211)
plt.plot(t_model, f_model, label='Model', c='k')
plt.scatter(vmjdobs,vflux,s=3,label='V flux',zorder=102)
plt.scatter(imjdobs,iflux,s=3,label='I flux',zorder=100)
plt.scatter(kmjdobs,kflux,s=3,label='Ks flux',zorder=101)
plt.grid()
#plt.legend(bbox_to_anchor=(0.9, -0.5))
plt.ylabel("$\Delta$ flux")
plt.xlim(t0_est-365,t0_est+365)

ax2 = plt.subplot(212, sharex=ax1)
plt.errorbar(vmjdobs,vflux-f_plot_v,yerr=vflux_error,fmt=',', alpha=0.3, zorder=102)
plt.scatter(vmjdobs,vflux-f_plot_v,s=3,label='V flux', zorder=202)
plt.errorbar(imjdobs,iflux-f_plot_i,yerr=iflux_error,fmt=',', alpha=0.3, zorder=100)
plt.scatter(imjdobs,iflux-f_plot_i,s=3,label='I flux', zorder=200)
plt.errorbar(kmjdobs,kflux-f_plot_k,yerr=kflux_error,fmt=',', alpha=0.3, zorder=101)
plt.scatter(kmjdobs,kflux-f_plot_k,s=3,label='Ks flux', zorder=202)
plt.grid()
#plt.legend()
plt.xlabel("MJD")
plt.ylabel("residual flux")
plt.xlim(t0_est-365,t0_est+365)
plt.ylim(-0.05, 0.05)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.55))

plt.setp(ax1.get_xticklabels(), visible=False)
plt.tight_layout()
plt.savefig('figs/I_best_model_disk.pdf', bbox_inches='tight')
plt.close()
