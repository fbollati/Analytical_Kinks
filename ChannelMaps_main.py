import numpy as np
import sys
import os
import shutil as sh

import ChannelMaps_functions as f
import ChannelMaps_settings as s

#from GridMapper import grid_mapper

import matplotlib.pyplot as plt


if len(sys.argv) != 2:
    print("Error!")
    print("Usage: < $ python ChannelMaps_main.py ChannelMaps.param >")
    sys.exit(-1)

file = str(sys.argv[1])

print('\n~ Reading parameter file, initializing global variables and meshgrid ...')
s.init(file)

path = s.name + '/'
os.makedirs(path, exist_ok=True)

sh.copy(file, path)

print('~ Creating Keplerian velocity field ...')
vK = f.create_Keplerian_velocity_field()
#vK = f.create_Keplerian_velocity_field_cartesian()

print('~ Uploading linear perturbations ...')
xl,yl,dl,ul,vl = f.upload_linear_perturbations()

print('~ Adding linear perturbations to Keplerian velocity field ...')
vr,vphi,deltav = f.vKepler_plus_linear_pert(xl,yl,ul,vl,vK)
#vr,vphi,deltav = f.vKepler_plus_linear_pert_cartesian(xl,yl,ul,vl,vK)

print('~ Nonlinear perturbations:')
dnl,unl,vnl = f.compute_nonlinear_pert()
#dnl,unl,vnl = f.compute_nonlinear_pert_cartesian()

print('~ Adding nonlinear perturbations ...')
vr,vphi,deltav = f.add_nonlinear_pert(unl,vnl,vr,vphi,deltav,vK)
#vr,vphi,deltav = f.add_nonlinear_pert_cartesian(unl,vnl,vr,vphi,deltav,vK)

if s.density:
    density_pert = f.merge_density_lin_nonlin(xl,yl,dl,dnl)
    #density_pert = f.merge_density_lin_nonlin_cartesian(xl,yl,dl,dnl)

print('~ Saving output fields to files')

if s.density:
    np.save(path + 'density_pert.npy', density_pert)
    rho = f.get_normalise_density_field(density_pert)
    #rho = f.get_normalise_density_field_cartesian(density_pert)
    np.save(path + 'density.npy', rho)

np.save(path + 'vr.npy', vr)
np.save(path + 'vphi.npy', vphi)
np.save(path + 'deltavphi.npy', vphi - (-s.cw * vK))
np.save(path + 'deltav.npy', deltav)

#print('~ Interpolating to new grid')

#grid_mapper(path)

"""
print('~ Making figures ...')
f.make_contourplot(deltav, bar_label='$\\delta v(r,\\varphi)$', saveas = path +'delta_v')
f.make_contourplot(vphi - (-s.cw * vK), bar_label='$v(r,\\varphi)$  [km/s]', saveas = path + 'azimuthal_vel_pert')
f.make_contourplot(vr, bar_label='$u(r,\\varphi)$  [km/s]', saveas = 'radial_vel_pert')
if s.density:
    f.make_contourplot(density_pert, bar_label = '$(\Sigma - \Sigma _0)\Sigma _0^{-1}$', saveas = path + 'density_pert')

if np.ndim(s.vchs) > 0:

    print('~ Getting velocity Cartesian components ...')
    v_field0 = f.get_velocity_Cartesian_components(np.zeros((s.Nphi,s.Nr)),-s.cw * vK)
    v_field = f.get_velocity_Cartesian_components(vr,vphi)

    print('~ Rotating mesh and velocity field ...')
    f.rotate_meshgrid()
    v_field0 = f.rotate_velocity_field(v_field0)
    v_field = f.rotate_velocity_field(v_field)

    print('~ Making channel maps plot ...')
    f.make_contourplot(v_field[:,:,2]-v_field0[:,:,2], bar_label='$\\Delta v_n (r,\\varphi)$   [km/s]', WithChannels = True, vz_field = v_field[:,:,2], saveas = path + 'contour.pdf')
"""

print('~ Done! \n')
