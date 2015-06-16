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
import os
import numpy as np

from helper import *
from graphics import *
from projections import *
from stimulus import *

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from mpl_toolkits.axes_grid1 import ImageGrid
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    P = retina_projection()

    # Checkerboard pattern for retina
    grid = 2*32
    even = grid / 2 * [0, 1]
    odd = grid / 2 * [1, 0]
    R = np.row_stack(grid / 2 * (even, odd))
    R = R.repeat(grid, axis=0).repeat(grid, axis=1)

    # Mask with a disc
    R = R* disc((retina_shape[0],retina_shape[0]),
                (retina_shape[0]//2,retina_shape[0]//2),
                retina_shape[0]//2)

    # Take half-retina
    R = R[:,retina_shape[1]:]

    # Project to colliculus
    SC = R[P[...,0], P[...,1]]


    fig = plt.figure(figsize=(10,15), facecolor='w')
    ######################
    
    ax1, ax2 = ImageGrid(fig, 211, nrows_ncols=(1,2), axes_pad=0.5)
    polar_frame(ax1, legend=True)
    polar_imshow(ax1, R, vmin=0, vmax=5)
    logpolar_frame(ax2, legend=True)
    logpolar_imshow(ax2, SC, vmin=0, vmax=5)
    ax1.text(1.1, 1.1, u"a",
          ha="left", va="bottom", fontsize=20, fontweight='bold')
        #ax1.text(0., -1.28, u"0°                                                     90°",
        #      ha="left", va="bottom", fontsize=10)
    ################################
    
    ax1, ax2 = ImageGrid(fig, 212, nrows_ncols=(1,2), axes_pad=0.5)
    polar_frame(ax1, legend=True,reduced=True)
    '''
    zax = zoomed_inset_axes(ax1, 6, loc=1)
    polar_frame(zax, zoom=True)
    zax.set_xlim(0.0, 0.15)
    zax.set_xticks([])
    zax.set_ylim(-.05, .05)
    zax.set_yticks([])
    zax.set_frame_on(True)
    mark_inset(ax1, zax, loc1=2, loc2=4, fc="none", ec="0.5")
    '''
    # Stimuli luminances are not additive
    R = np.maximum(stimulus(position=(1,0)), stimulus(position=(5,0)))
    R = np.maximum(R , stimulus(position=(10,0)))
    polar_imshow(ax1, R,reduced=True)
    #polar_imshow(zax, R)
    
    logpolar_frame(ax2, legend=True)
    P = retina_projection()
    SC = R[P[...,0], P[...,1]]
    logpolar_imshow(ax2, SC)
    ax1.text(1.1, 1.1, u"b",
                 ha="left", va="bottom", fontsize=20, fontweight='bold')


    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                                        wspace=0.15, hspace=0.2)
    plt.savefig("Fig-1.pdf")
    plt.savefig("Fig-1.eps")
    plt.show()
