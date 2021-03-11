import numpy as np
import sys
import os

import ChannelMaps_functions as f
import ChannelMaps_settings as s

if len(sys.argv) != 3:
    print("Error!")
    print("Usage: < $ python Plots.py ChannelMaps.param results_directory >")
    sys.exit(-1)

file = str(sys.argv[1])
dir = str(sys.argv[2])

s.init(file)
vK = f.create_Keplerian_velocity_field()

vr = np.loadtxt(dir + '/vr.txt')
f.make_contourplot(vr, bar_label='$v_r(r,\phi)$  [km/s]', saveas = dir + '/v_r.pdf')

vphi = np.loadtxt(dir + '/vphi.txt')
f.make_contourplot(vphi - (-s.cw * vK), bar_label='$v_{\phi}(r,\phi)$  [km/s]', saveas = dir + '/v_phi.pdf')

deltav = np.loadtxt(dir + '/deltav.txt')
f.make_contourplot(deltav, bar_label='$\delta v(r,\phi)$  [km/s]', saveas = dir + '/delta_v.pdf')

if os.path.exists(dir + '/density_pert.txt'):
    density_pert = np.loadtxt(dir + '/density_pert.txt')
    f.make_contourplot(density_pert, bar_label = '$(\Sigma - \Sigma _0)\Sigma _0^{-1}$', saveas = dir + '/density_pert.pdf')
