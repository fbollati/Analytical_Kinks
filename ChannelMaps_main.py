import numpy as np
import sys

import ChannelMaps_functions as f
import ChannelMaps_settings as s


if len(sys.argv) != 2:
    print("Error!")
    print("Usage: < $ python ChannelMaps_main.py ChannelMaps.param >")
    sys.exit(-1)

file = str(sys.argv[1])



print('\n~ Reading parameter file, initializing global variables and meshgrid ...')
s.init(file)

print('~ Creating Keplerian velocity field ...')
vK = f.create_Keplerian_velocity_field()

print('~ Uploading linear perturbations ...')
xl,yl,dl,ul,vl = f.upload_linear_perturbations()

print('~ Adding linear perturbations to Keplerian velocity field ...')
vr,vphi,deltav = f.vKepler_plus_linear_pert(xl,yl,ul,vl,vK)

print('~ Nonlinear perturbations:')
dnl,unl,vnl = f.compute_nonlinear_pert()

print('~ Adding nonlinear perturbations ...')
vr,vphi,deltav = f.add_nonlinear_pert(unl,vnl,vr,vphi,deltav,vK)

if s.density:
    density_pert = f.merge_density_lin_nonlin(xl,yl,dl,dnl)

print('~ Making figures ...')
f.make_contourplot(deltav, bar_label='$\\delta v(r,\\varphi)$', saveas = 'delta_v')
f.make_contourplot(vphi - (-s.cw * vK), bar_label='$v(r,\\varphi)$  [km/s]', saveas = 'azimuthal_vel_pert')
f.make_contourplot(vr, bar_label='$u(r,\\varphi)$  [km/s]', saveas = 'radial_vel_pert')
if s.density:
    f.make_contourplot(density_pert, bar_label = '$(\Sigma - \Sigma _0)\Sigma _0^{-1}$', saveas = 'density_pert')

if np.ndim(s.vchs)>0:

    print('~ Getting velocity Cartesian components ...')
    v_field0 = f.get_velocity_Cartesian_components(np.zeros((s.Nr,s.Nphi)),-s.cw * vK)
    v_field = f.get_velocity_Cartesian_components(vr,vphi)

    print('~ Rotating mesh and velocity field ...')
    f.rotate_meshgrid()
    v_field0 = f.rotate_velocity_field(v_field0)
    v_field = f.rotate_velocity_field(v_field)

    print('~ Making channel maps plot ...')
    f.make_contourplot(v_field[:,:,2]-v_field0[:,:,2], bar_label='$\\Delta v_n (r,\\varphi)$   [km/s]', WithChannels = True, vz_field = v_field[:,:,2])

print('~ Done! \n')
