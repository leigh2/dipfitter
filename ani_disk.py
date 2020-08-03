#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from disk_in_orbit import lin_limb_dark, get_model_lc_disk
import matplotlib.animation as animation


def get_grid(step=0.01, full=False, scale=1.0):
    # get grid points over which to perform occultation evaluation
    xpts, ypts = np.meshgrid(np.arange(-scale*1.62,scale*1.62,step),
                             np.arange(-scale*1.0,scale*1.0,step))
    if not full:
        xind = np.hypot(xpts, ypts)<1.0
        xpts, ypts = [_[xind] for _ in [xpts, ypts]]
    return xpts, ypts

def get_I(step=0.01, mu=1.05, full=False, scale=1.0):
    # get grid positions
    xpts, ypts = get_grid(step=step, full=full, scale=scale)
    # get intensity of background source
    I = lin_limb_dark(np.hypot(xpts,ypts), mu)
    return I/I.sum(), xpts, ypts


def get_disk_illustration(
    xpts, ypts, tpt,
    t0, vt, b, # time of conjunction, velocity, impact parameter
    r1, r2, tilt, # semimajor and semiminor axis, tilt angle
    ):
    sin_t, cos_t = np.sin(tilt), np.cos(tilt)
    # current relative position
    cury = -b
    curx = (tpt - t0) * vt
    # evaluate model
    x2 = -(xpts - curx) * sin_t + (ypts - cury) * cos_t
    x1 =  (xpts - curx) * cos_t + (ypts - cury) * sin_t
    F = ((x1 / r1)**2 + (x2 / r2)**2) - 1
    return F


fig = plt.figure(figsize=(8,8/1.62))
ax = plt.subplot(111)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
xbest = [5.60214520e+04, 1.74569943e-02, 6.56927346e-01, 1.56950678e+00, 1.23366264e+00, 1.48362082e+00, 1.52279622e-02]
#xbest *= np.array([1.0,1.0,0.5,0.2,0.1,np.pi/4,1.0])
imscale = 3.0
I,xp,yp = get_I(full=True, scale=imscale)
ims = []
for t in np.linspace(-500,500,1001):
    F = get_disk_illustration(xp,yp,xbest[0]+t,*xbest[:-1])
    im1 = plt.imshow(I, extent=[-imscale*1.62,imscale*1.62, -imscale*1.0,imscale*1.0], interpolation='gaussian')
    im2 = plt.contourf(xp,yp,F,[-np.inf, 0], colors='grey', alpha=0.9, animated=True).collections
    txt = plt.text(-4.5,2.5, "t: {:+.1f} days".format(t), c='white')
    ims.append([im1]+im2+[txt])
ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True,
                                repeat_delay=0)
ani.save('ani_disk.mp4')
#plt.show()
