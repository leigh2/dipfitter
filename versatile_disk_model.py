#!/usr/bin/env python

import numpy as np
from utils import get_plx_params




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




class model_eclipse:
    """
    Model light curve of a star occulted by an object.
    n.b. if a solid spherical occulter of an ordinary star is desired there
    exist transit modelling codes which are much more efficient, e.g. batman
    https://www.cfa.harvard.edu/~lkreidberg/batman/
    """

    def __init__(self,
                 mjd_points=None, step=0.01, mu=1.05,
                 t0=None, ra=None, dec=None,
                 motion_mode='linear', occulter_mode='solid'):

        if mjd_points is None:
            raise runtimeError("mjd_points must be provided")
        else:
            self.mjd_points = mjd_points

        # get evaluation grid
        self.I, self.xpts, self.ypts = get_I(step=step, mu=mu)

        if motion_mode=='parallactic':
            if t0 is None or ra is None or dec is None:
                raise runtimeError("t0, ra and dec must be provided if motion_mode=='parallactic'")
            else:
                self.RdotW, self.RdotN = get_plx_params(mjd_points, ra, dec)
                self.dt = mjd_points - t0
                self.linear_motion, self.parallactic_motion = False, True
                self.motion_params = 5
                # parameters are:
                # a0, d0: relative ra and dec offset at t0 in units of giant radii
                # ua, ud: relative proper motion in ra and dec in units of giant radii per time unit
                # pi: relative parallax in units of giant radii

        elif motion_mode=='linear':
            self.linear_motion, self.parallactic_motion = True, False
            self.motion_params = 3
            # parameters are:
            # t0: time of minimum separation
            # vt: relative tangential velocity in units of giant radii per time unit
            # b: distance between centers at minimum separation in units of giant radii

        else:
            raise NameError("motion mode {:s} not recognized, accepted values are 'linear' and 'parallactic'".format(motion_mode))



        if occulter_mode=='solid':
            self.solid_occulter, self.exponential_occulter = True, False
            # parameters are:
            # r1, r2: semimajor and semiminor axes (set these equal for circular) relative to giant radius
            # tilt: angle relative to motion vector (proper or tangential, not parallax) in radians
            # T: transmittance [0->1]

        elif occulter_mode=='exponential':
            self.solid_occulter, self.exponential_occulter = False, True
            # parameters are:
            # r1, r2: semimajor and semiminor axes (set these equal for circular) relative to giant radius
            # tilt: angle relative to motion vector (proper or tangential, not parallax) in radians
            # T0: central transmittance [0->1]
            # H: scale radius

        else:
            raise NameError("occulter_mode mode {:s} not recognized, accepted values are 'solid' and 'exponential'".format(occulter_mode))



    def get_lc(self, params):
        """
        generate the model light curve for the provided parameters
        """
        if self.linear_motion:
            t0, vt, b = params[0:self.motion_params]
            dt = self.mjd_points - t0
            curx = b
            cury = (dt * vt)[:,None]
        else:
            a0, d0, ua, ud, pi = params[0:self.motion_params]
            curx = (a0 + pi*self.RdotW + ua*self.dt)[:,None]
            cury = (d0 + pi*self.RdotN + ud*self.dt)[:,None]

        if self.solid_occulter:
            r1, r2, tilt, T = params[self.motion_params:]
        else:
            r1, r2, tilt, T0, H = params[self.motion_params:]

        # some trig, tiny performance increase doing it here instead of later
        sin_t, cos_t = np.sin(tilt), np.cos(tilt)

        # ellipse functions
        x1 = -(self.xpts - curx) * sin_t + (self.ypts - cury) * cos_t
        x2 =  (self.xpts - curx) * cos_t + (self.ypts - cury) * sin_t
        r = ((x1 / r1)**2 + (x2 / r2)**2)

        # convolve intensity map with occulter
        if self.solid_occulter:
            _I = self.I * np.where(r < 1, T, 1.0)
        else:
            _I = self.I * (1-(1-T0)*np.exp(-r/H))

        # return integrated flux
        return np.sum(_I, axis=1)






if __name__=="__main__":
    import matplotlib.pyplot as plt
    from astropy.io.fits import getdata

    def inv_var_weight_avg(x, ex):
        # compute inverse variance weighted average and error
        w = 1./(ex**2) # weights
        x_wav = np.nansum(w * x, axis=0) / np.nansum(w, axis=0)
        σ_wav = 1./np.sqrt(np.nansum(w, axis=0))
        return x_wav, σ_wav


    def impact_parameter(t0, t, dx, uy):
        dt = t-t0
        # return the impact parameter
        return np.hypot(dx,uy*dt)


    def make_flux(mags, mag_errors, baseline_mag):
        fluxes = 10**(0.4*(baseline_mag- mags))
        flux_errors = fluxes * (10**((2*mag_errors)/5) - 1)
        return fluxes, flux_errors


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

    x_model = [5.60214520e+04, 1.74569943e-02, 6.56927346e-01, 1.56950678e+00, 1.23366264e+00, 1.48362082e+00, 1.52279622e-02]

    I,xp,yp = get_I(step=0.01) # step=0.02 means max model error is ~0.35 * data error
    kwargs = {
        'motion_mode': 'linear',
        'occulter_mode': 'solid',
    }

    t_model = np.linspace(imjdobs.min(), imjdobs.max(), 1000)
    f_model = model_eclipse(mjd_points=t_model, **kwargs).get_lc(x_model)
    f_plot_i = model_eclipse(mjd_points=imjdobs, **kwargs).get_lc(x_model)
    f_plot_v = model_eclipse(mjd_points=vmjdobs, **kwargs).get_lc(x_model)
    f_plot_k = model_eclipse(mjd_points=kmjdobs, **kwargs).get_lc(x_model)

    fig = plt.figure(figsize=(10,5))

    ax1 = plt.subplot(211)
    plt.plot(t_model, f_model, label='Model', c='k')
    plt.plot(t_model, f_model+1.0, label='Model', c='k')
    plt.plot(t_model, f_model+2.0, label='Model', c='k')
    plt.scatter(vmjdobs,vflux,s=3,label='V flux',zorder=102)
    plt.scatter(imjdobs,iflux+1.0,s=3,label='I flux',zorder=100)
    plt.scatter(kmjdobs,kflux+2.0,s=3,label='Ks flux',zorder=101)
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
    plt.show()
