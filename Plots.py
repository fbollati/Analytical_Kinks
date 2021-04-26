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

vr = np.load(dir + '/vr.npy')
f.make_contourplot(vr, bar_label='$v_r(r,\phi)$  [km/s]', saveas = dir + '/v_r.pdf')

vphi = np.load(dir + '/vphi.npy')
f.make_contourplot(vphi, bar_label='$v_{\phi}(r,\phi)$  [km/s]', saveas = dir + '/v_phi.pdf')

#  - (-s.cw * vK)

deltav = np.load(dir + '/deltav.npy')
f.make_contourplot(deltav, bar_label='$\delta v(r,\phi)$  [km/s]', saveas = dir + '/delta_v.pdf')

if os.path.exists(dir + '/density_pert.npy'):
    density_pert = np.load(dir + '/density_pert.npy')
    f.make_contourplot(density_pert, bar_label = '$(\Sigma - \Sigma _0)\Sigma _0^{-1}$', saveas = dir + '/density_pert.pdf')

if np.ndim(s.vchs) > 0:

    print('~ Getting velocity Cartesian components ...')
    v_field0 = f.get_velocity_Cartesian_components(np.zeros((s.Nphi,s.Nr)),-s.cw * vK)
    v_field = f.get_velocity_Cartesian_components(vr,vphi)

    print('~ Rotating mesh and velocity field ...')
    f.rotate_meshgrid()
    v_field0 = f.rotate_velocity_field(v_field0)
    v_field = f.rotate_velocity_field(v_field)

    print('~ Making channel maps plot ...')
    f.make_contourplot(v_field[:,:,2]-v_field0[:,:,2], bar_label='$\\Delta v_n (r,\\varphi)$   [km/s]', WithChannels = True, vz_field = v_field[:,:,2], saveas = dir + 'contour.pdf')
