#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright INRIA
# Contributors: Wahiba Taouali (Wahiba.Taouali@inria.fr)
#               Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#
# This software is governed by the CeCILL license under French law and abiding
# by the rules of distribution of free software. You can use, modify and/ or
# redistribute the software under the terms of the CeCILL license as circulated
# by CEA, CNRS and INRIA at the following URL
# http://www.cecill.info/index.en.html.
#
# As a counterpart to the access to the source code and rights to copy, modify
# and redistribute granted by the license, users are provided only with a
# limited warranty and the software's author, the holder of the economic
# rights, and the successive licensors have only limited liability.
#
# In this respect, the user's attention is drawn to the risks associated with
# loading, using, modifying and/or developing or reproducing the software by
# the user in light of its specific status of free software, that may mean that
# it is complicated to manipulate, and that also therefore means that it is
# reserved for developers and experienced professionals having in-depth
# computer knowledge. Users are therefore encouraged to load and test the
# software's suitability as regards their requirements in conditions enabling
# the security of their systems and/or data to be ensured and, more generally,
# to use and operate it in the same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid


from model import *
from graphics import *
from stimulus import *
from parameters import *
from projections import *


def run(model, ax, y1=None):

    duration = 1000*millisecond
    dt = 4*millisecond

    SCV = model.run(duration=duration, dt=dt)
    n = int(duration/dt)

    x,_ = polar_to_logpolar(5/90.,0.)
    x = int(x*128)
    Z = np.zeros((n, colliculus_shape[1]))
    for i in range(n):
        Z[i] = SCV[i,:,x].ravel()

    X = np.linspace(-90, 90, colliculus_shape[1])
    Y = np.linspace(0, n, n)
    X, Y = np.meshgrid(X, Y)

    ax.plot_surface(X, Y, Z*500, rstride=8, cstride=8,linewidth=.4,  edgecolors='k',
                color="w", antialiased=True, shade=True)
    lw = 1.5
    
    if y1 is None:
        y1 = model.SC_V[:,x].argmax()
        y2 = colliculus_shape[1] - y1
        ax.plot(Y[:,y1], Z[:,y1]*500, zs=-90, zdir='x', zorder=-50, color='b', lw=lw)
        ax.plot(Y[:,y2], Z[:,y2]*500, zs=-90, zdir='x', zorder=-50, color='r', lw=lw)
        ax.plot(Y[:,y1], Z[:,y1]*500, zs=(2*(y1/128.)-1)*90, zdir='x', zorder=50, color='b', lw=lw)
        ax.plot(Y[:,y2], Z[:,y2]*500, zs=(2*(y2/128.)-1)*90, zdir='x', zorder=50, color='r', lw=lw)
    else:
        y2 = colliculus_shape[1] - y1
        y3 = colliculus_shape[1]//2
        #ax.plot(Y[:,y1], Z[:,y1]*500, zs=-90, zdir='x', zorder=-50, color='b')
        #ax.plot(Y[:,y2], Z[:,y2]*500, zs=-90, zdir='x', zorder=-50, color='r')
        #ax.plot(Y[:,y1], Z[:,y1]*500, zs=(2*(y1/128.)-1)*90, zdir='x', color='b')
        #ax.plot(Y[:,y2], Z[:,y2]*500, zs=(2*(y2/128.)-1)*90, zdir='x', zorder=50, color='r')
        ax.plot(Y[:,y3], Z[:,y3]*500, zs=-90, zdir='x', zorder=-50, color='k', lw=lw)
        ax.plot(Y[:,y3], Z[:,y3]*500, zs=(2*(y3/128.)-1)*90, zdir='x', zorder=50, color='k', lw=lw)
    fontsize=7


    ax.set_xlim(-90,90)
    ax.set_ylim(0,250)
    ax.set_zlim(0,300)
    ax.set_xlabel(u'Stimuli position [°]', fontsize=fontsize,labelpad=4)
    ax.set_ylabel('Time [ms]', fontsize=fontsize,labelpad=5)
    ax.set_zlabel('Discharge rate (spike/s)', fontsize=fontsize,labelpad=5)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-1)
    ax.set_zticks([0,100,200,300])


plt.figure(figsize=(2.5,6.5), facecolor='w',dpi=100)

model = Model()

model.reset()
model.R = stimulus((5.0, -30), size=1, intensity=1)


ax = plt.subplot(311, projection='3d')
run(model,ax)
ax.text2D(0.0, 1.0, "a", va='top', ha='right',
          transform=ax.transAxes, fontsize=10, fontweight='bold')


# Get location of -20/+20 on SC
model.reset()
model.R = stimulus((5.0, -20), size=1, intensity=1)
model.run(1.*second, 4*millisecond)
x,_ = polar_to_logpolar(5/90.,0.)
x = int(x*128)
y1 = model.SC_V[:,x].argmax()

model.reset()
model.R = np.maximum( stimulus((5.0, -20), size=1, intensity=1) ,
                      stimulus((5.0, +20), size=1, intensity=1) )


ax = plt.subplot(312, projection='3d')
run(model, ax, y1)
ax.text2D(0.0, 1.0, 'b', va='top', ha='right',
          transform=ax.transAxes, fontsize=10, fontweight='bold')


model.reset()
model.R = np.maximum( stimulus((5.0, -30), size=1, intensity=1) ,
                      stimulus((5.0, +30), size=1, intensity=1) )

ax =  plt.subplot(313, projection='3d')
run(model,ax)
ax.text2D(0.0, 1.0, 'c', va='top', ha='right',
          transform=ax.transAxes, fontsize=10, fontweight='bold')
ax.grid(color="k", alpha="1")
plt.tight_layout()

plt.savefig('figures/Fig-5.pdf')
plt.savefig('figures/Fig-5.eps')
plt.show()
