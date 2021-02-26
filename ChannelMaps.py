import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.integrate as ode
import sys
import Functions as f
import matplotlib


plt.rcParams.update({'font.size': 16})
plt.tight_layout()
plt.rcParams.update({'figure.autolayout': True})

# CONSTANTS cgs ----------------------------------------------------------------

au = 1.49597870*10**13 #cm
G = 6.67*10**(-8)      #
Mj = 1.89819*10**30    #g
Msun = 1.989*10**33    #g


# SYSTEM PARAMETERS ------------------------------------------------------------

Rdisk = 300 # au (gas disc radius)
Rdust = 100 # au (dust disc radius)
rp = 100    # au (planet orbital radius)
phip = 0    # (initialize planet azimuth) DO NOT CHANGE THIS!
hr = 0.1    # hp/rp

M = 1       # Msun (stellar mass)
Mp = 1      # Mj (planet mass)

cw = -1     # disc rotation direction (-1 -> counterclockwise) DO NOT CHANGE THIS!

# Sound speed and surface density profile indices
nu = 1./4.    # c = c0(r/rp)^(-nu)
delta = 1     # Sigma = Sigma0(r/rp)^(-delta)

alfaSS = 5*10**(-3) # Shakura-Sunyaev parameter (for viscous damping of velocity perturbations)
m = 100             # Lindblad resonance index  (for viscous damping of velocity perturbations)

#mu = 2.3           # mean molecular weight
indad = 5/3         # E.O.S. adiabatic index

# angles for gas disc rotation
gammap = 0      # around z-axis (line-of-sight) (change planet azimuth)
teta = np.pi/6  # around y-axis (of fixed coordiante frame)
gamma = 0       # around z-axis


# BURGERS EQUATION INITIAL CONDITION -------------------------------------------

# (Numerical parameters related to the initial condition of
#  eq. (30) Rafikov 2002, i.e. Fig. 1-top. These parameters
#  appear in the asymptotic N-wave solution)

C0 = 0.4
eta_tilde = 3.497
t0 = 1.89


# DISC GRID --------------------------------------------------------------------

Ngrid = 500                           # suggested value = 1000

Delta = 2*Rdisk/Ngrid
x = np.arange(-Rdisk,Rdisk+Delta,Delta) # grid cartesian coordinates in au
y = np.arange(-Rdisk,Rdisk+Delta,Delta)
N = len(x)
X,Y = np.meshgrid(x,y)

# edges of gas and dust discs:
Ndust = 100
phidust = np.linspace(0,2*np.pi,Ndust)
disc = np.zeros((Ndust,2))
dust = np.zeros((Ndust,2))
for i in range(Ndust):
   disc[i,:] = f.cartesian(Rdisk,phidust[i])
   dust[i,:] = f.cartesian(Rdust,phidust[i])


# COMPUTATION OF SOME SYSTEM PARAMETERS ----------------------------------------

vkp = f.v(rp,M) # km/s - planet keplerian velocity
csp = vkp*hr    # km/s - cs/vK = h/r
hp = hr*rp      # au
l = (2/3)*hp    # au   - l = csp/|2A(rp)| unit of length Goodman & Rafikov 2001 (GR01)

M1p = (l*au)*(csp*10**5)**2/(G*Mj)  # Mj - reference mass of (GR01) eq.(19)
#Mp = M1p
betap = Mp/M1p                      # planet mass in unit of reference mass M1p


C = (indad + 1)*betap*C0/2**(3/4)   # parameter related to the shock amplitude

planet = np.array([rp,0,0])
planet = f.rotate_planet(0,0,phip,planet)

print("betap is: "+str(betap))
print("csp is: "+str(csp))


# BURGERS EQUATION NUMERICAL SOLUTION ------------------------------------------
# (Numerical solution of eq. (30) Rafikov 2002)

time,xc,ub,xIc,ubI = f.burgers(betap,indad,Mp) #ub[eta,t]


# ********************** V E L O C I T Y   F I E L D S *************************

# KEPLERIAN VELOCITY FIELD -----------------------------------------------------

vk0,vx0,vy0 = f.create_Kepler(x,y, N, cw, Rdisk,M)

# APPLY LINEAR PERTURBATIONS (1) -----------------------------------------------
# (in a square of side = 4*l)

un,vn,xn,yn = f.createUV(rp) # get dimensionless linear velocity perturbations
Xn,Yn = np.meshgrid(xn,yn)   # (in unit of csp) for a planet of mass 1 M1p

'''
plt.contourf(Xn,Yn,un)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.contourf(Xn,Yn,vn)
plt.show()
'''

un = un*betap*csp
vn = vn*betap*csp
xn = rp + l*xn
yn = l*yn
Xn,Yn = np.meshgrid(xn,yn)

'''
plt.contourf(Xn,Yn,un)
plt.xlabel("x [au]")
plt.ylabel("y [au]")
plt.colorbar()
plt.show()
plt.contourf(Xn,Yn,vn)
plt.colorbar()
plt.show()

plt.contourf(Xn,Yn,sn)
plt.xlabel("x [au]")
plt.ylabel("y [au]")
plt.colorbar()
plt.show()
'''

# Keplerian velocity field + linear velocity perturbation near the planet

vk, vx, vy = f.create_perturbed(rp,l,N,x,y,vx0,vy0,vk0,un,vn,xn,yn,False,alfaSS,m,hr,nu)

# Keplerian velocity field + linear velocity perturbation near the planet + viscous damping

vkS, vxS, vyS = f.create_perturbed(rp,l,N,x,y,vx0,vy0,vk0,un,vn,xn,yn,True,alfaSS,m,hr,nu)


# APPLY NONLINEAR PERTURBATIONS ------------------------------------------------
# (outside the small square of linear regime)

print("Computation of nonlinear perturbations on the grid:")

ri = rp*(1-4*hr/3) #not used in the function
ro = rp*(1+4*hr/3) #not used in the function
dnl,unl,vnl = f.nonlinear_perturbations(x,y,N,Rdisk,ri,ro,rp,phip,hr,nu,delta,cw,C,eta_tilde,t0,indad,csp,time,xc,ub,xIc,ubI)

# (the nonlinear perturbations are applied to the velocity field that already contains the linear perturbation)


# no viscosity
vx,vy,vk = f.apply_velocity_perturbations(N,x,y,unl,vnl,vx,vy,vk,Rdisk,False,alfaSS,m,hr,rp,nu)
dvofv = f.dvofv(N,vk0,vk) #Delta |velocity|/|velocity|

# viscosity
vxS,vyS,vkS = f.apply_velocity_perturbations(N,x,y,unl,vnl,vxS,vyS,vkS,Rdisk,True,alfaSS,m,hr,rp,nu)
dvofvS = f.dvofv(N,vk0,vkS)



# SOME PLOTS ...

'''
print("Creation of the wake")
phip = 0
Nwake = 3000 # points for wake linspace
wake, wakep = f.create_wake(Nwake,Rdisk,rp,phip,nu,hr,-1)
'''


# plot of Delta |velocity|/|velocity|
'''
maxdv = round(dvofv.max(),2)
d = 2*maxdv/1000

print(dvofv)
print(dvofv.max())
print(maxdv)
print(d)

fig, ax = plt.subplots()
conto = plt.contourf(X,Y,dvofv,levels =np.arange(-0.101,0.101,0.0001) ,cmap='seismic_r',zorder=0) #np.arange(-maxdv*(1+0.01),maxdv*(1+0.01),d)

cb = fig.colorbar(conto, ticks=[-0.1,-0.05, 0, 0.05, 0.1])
cb.set_label('$\delta v(r,\\varphi)$')
cb.ax.locator_params(nbins=5)
plt.plot(disc[:,0],disc[:,1],color='k')
#plt.plot(wake[0,:],wake[1,:], color = 'orange', label="$(r,\\varphi_{wake}(r))$", zorder=+1)
plt.scatter(planet[0],planet[1],label="planet",color='k',zorder = +2)
plt.scatter(0,0,label="star",color='gold',zorder = +2,s = 80, marker=(5, 1))
#plt.title("$\delta v(r,\\varphi)$")
plt.axis("equal")
plt.locator_params(nbins=3)
plt.xlabel("$x$ [au]")
plt.ylabel("$y$ [au]")
plt.show()
'''

# Plot of Total perturbations!!!!!! vtot, utot, stot
'''
#utot,vtot,utotS,vtotS = f.totalUV(l,N,x,y, unl,vnl, un, vn,xn,yn, alfaSS,m,hr,rp,nu)
stot,utot,vtot,stotS,utotS,vtotS = f.totalUVS(l,N,x,y,dnl, unl,vnl,sn, un, vn,xn,yn, alfaSS,m,hr,rp,nu)

maxdv = round(vkp*0.2,3)
d = 2*maxdv/1000
cmap_reversed = matplotlib.cm.get_cmap('RdBu')
fig, ax = plt.subplots()
conto = plt.contourf(X,Y,utot,levels = np.arange(-0.62,0.62,0.0001),cmap='RdBu',zorder=0)

cb = fig.colorbar(conto, ticks=[-maxdv,-round(maxdv/2,2), 0, round(maxdv/2,2),maxdv])
cb.set_label('$u$ [km/s]')
cb.ax.locator_params(nbins=5)
plt.plot(disc[:,0],disc[:,1],color='k')
#plt.plot(wake[0,:],wake[1,:], color = 'orange', label="$(r,\\varphi_{wake}(r))$", zorder=+1)
plt.scatter(planet[0],planet[1],label="planet",color='k',zorder = +2)
plt.scatter(0,0,label="star",color='gold',zorder = +2,s = 80, marker=(5, 1))
plt.title("Radial velocity perturbation \n $u(r,\\varphi)$")
plt.axis("equal")
plt.locator_params(nbins=3)
plt.xlim([40, 160])
plt.ylim([-50,60 ])
plt.xlabel("$x$ [au]")
plt.ylabel("$y$ [au]")
plt.show()

# azimuthal velocity perturbation

maxdv = round(vkp*0.2,2)
d = 2*maxdv/1000

fig, ax = plt.subplots()
conto = plt.contourf(X,Y,vtot,levels = np.arange(-0.62,0.62,0.0001),cmap='RdBu',zorder=0)

cb = fig.colorbar(conto, ticks=[-maxdv,-round(maxdv/2,2), 0, round(maxdv/2,2),maxdv])
cb.set_label('$v$ [km/s]')
cb.ax.locator_params(nbins=5)
plt.plot(disc[:,0],disc[:,1],color='k')
#plt.plot(wake[0,:],wake[1,:], color = 'orange', label="$(r,\\varphi_{wake}(r))$", zorder=+1)
plt.scatter(planet[0],planet[1],label="planet",color='k',zorder = +2)
plt.scatter(0,0,label="star",color='gold',zorder = +2,s = 80, marker=(5, 1))
plt.title("Azimuthal velocity perturbation \n $v(r,\\varphi)$")
plt.axis("equal")
plt.locator_params(nbins=3)
plt.xlim([40, 160])
plt.ylim([-50,60 ])
plt.xlabel("$x$ [au]")
plt.ylabel("$y$ [au]")
plt.show()

# total density perturbation:

maxs = 2.5
d = 1/1000

fig, ax = plt.subplots()
conto = plt.contourf(X,Y,stot,levels = np.arange(-2.51,2.51,d),cmap='RdBu',zorder=0)

cb = fig.colorbar(conto, ticks=[-2.5, 0, 2.5])
cb.set_label('sigma')
cb.ax.locator_params(nbins=3)
plt.plot(disc[:,0],disc[:,1],color='k')
#plt.plot(wake[0,:],wake[1,:], color = 'orange', label="$(r,\\varphi_{wake}(r))$", zorder=+1)
plt.scatter(planet[0],planet[1],label="planet",color='k',zorder = +2)
plt.scatter(0,0,label="star",color='gold',zorder = +2,s = 80, marker=(5, 1))
plt.title("Total density perturbation")
plt.axis("equal")
plt.locator_params(nbins=3)
plt.xlim([40, 160])
plt.ylim([-50,60 ])
plt.xlabel("$x$ [au]")
plt.ylabel("$y$ [au]")
plt.show()

sys.exit(-1)
'''
# ROTATIONS --------------------------------------------------------------------

print("Application of rotations to the fields")

Xnew,Ynew,vf0 = f.apply_arbitrary_rotation(N,gammap,teta,gamma,X,Y,x,y,Rdisk,vx0,vy0)
Xnew,Ynew,vf = f.apply_arbitrary_rotation(N,gammap,teta,gamma,X,Y,x,y,Rdisk,vx,vy)
Xnew,Ynew,vfS = f.apply_arbitrary_rotation(N,gammap,teta,gamma,X,Y,x,y,Rdisk,vxS,vyS)

for i in range(Ndust):
   disc[i,0],disc[i,1], hello = f.rotate_planet_arbitrary(0,teta,gamma,np.array([disc[i,0],disc[i,1],0]))
   #dust[i,0],dust[i,1], hello = f.rotate_planet_arbitrary(0,teta,-np.pi/4,np.array([dust[i,0],dust[i,1],0]))

planet = f.rotate_planet_arbitrary(gammap,teta,gamma,planet)


# chennels
vchs = np.array([-1.6,-0.9,-0.2,0.2,0.9,1.6])


#************************* C H A N N E L   M A P S *****************************

# SINGOLA CHANNEL, o multiple selezionando colore ------------------------------
'''
fig,ax = plt.subplots()

ppp=plt.contourf(Xnew, Ynew, vfS[2,:,:]-vf0[2,:,:], zorder=0, levels = np.arange(-0.3,0.3,0.001), cmap='seismic_r')
#ppp=plt.contourf(Xnew, Ynew, vf[2,:,:]-vf0[2,:,:], zorder=0, levels = np.arange(-0.15,0.15,0.001), cmap='seismic_r') ----> nel code finale ci vuole flag per decidere cosa stampare.
cb=fig.colorbar(ppp)
cb.set_label('$\delta v$')
cb.ax.locator_params(nbins=5)

plt.plot(disc[:,0],disc[:,1],zorder = 0, label ="disc",color='k')
#plt.plot(dust[:,0],dust[:,1],zorder = 0, label ="dust",color='k')

#colors = np.array(["green","blue","m","y","orange"])
colors = np.array(["orange" for i in range(len(vchs))])
Nch = len(vchs)
for i in range(Nch):
    vll = f.channels([vchs[i]],vfS)
    Levels = np.arange(vchs[i]-0.5,vchs[i]+1.5,1) # DA AGGIUSTAREEEEEEE!!!!!!
    cs = plt.contourf(Xnew, Ynew, vll,levels=Levels, colors = colors[i], zorder = +3)
    P = plt.contour(Xnew, Ynew, vll,levels=Levels, colors='black', linewidths = 1 ,zorder = +4)

items = [plt.Rectangle((0,0),1,1,fc=colors[i]) for i in range(Nch)] #fc = FaceColor
ax.legend(items, [str(vchs[i]) for i in range(Nch)], loc = "lower left",title="$v_{ch}$ [km/s]:",fontsize=13)
plt.scatter(0,0,color='gold',zorder = +4,s = 80, marker=(5, 1))
plt.scatter(planet[0],planet[1],color='k',zorder=+2)

plt.xlabel('$x$ [au]')
plt.ylabel('$y$ [au]')
plt.locator_params(nbins=3)

plt.axis("equal")
#plt.title('CHANNEL MAPS')
plt.show()

sys.exit(-1)
'''

# PLOT TANTE CHANNEL CON VISC --------------------------------------------------


# I LEVELS FUNZIONANO SE SONO EQUISPAZIATE!!!!!!

vl0 = f.channels(vchs, vf0)
vl = f.channels(vchs,vf)
vlS = f.channels(vchs,vfS)



cmaprev = matplotlib.cm.get_cmap('RdYlBu')
seisrev = matplotlib.cm.get_cmap('seismic_r')


#conto=plt.contourf(Xnew, Ynew, dvofv, zorder=0, levels = np.arange(-0.15,0.15,0.001) ,cmap='seismic') #|||||||||||||
#cb=fig.colorbar(conto)
#cb.set_label('$\delta v$')
#cb.ax.locator_params(nbins=5)

'''
phis = np.linspace(0,2*np.pi, 1000)
rs = np.zeros(phis.shape)
xs = np.zeros(phis.shape)
ys = np.zeros(phis.shape)
for i in range(len(rs)):
  rs[i] = f.rmezzi(M,phis[i],vchs[1],teta)
  xs[i],ys[i] = f.cartesian(rs[i],phis[i])
plt.plot(xs,ys,linestyle = '--',color = 'k')
for i in range(len(rs)):
  rs[i] = f.rmezzi(M,phis[i],-vchs[1],teta)
  xs[i],ys[i] = f.cartesian(rs[i],phis[i])
plt.plot(xs,ys,linestyle = '--',color = 'k')
'''



# PLOT CHANNEL MAPS UNPERTURBED DISC
fig,ax = plt.subplots()

#plt.plot(dust[:,0],dust[:,1],zorder = 0, label ="dust", color = 'k')
plt.plot(disc[:,0],disc[:,1],zorder = 0, label ="disc",color='k')
dvchs = np.abs(vchs[1]-vchs[0])
levels = np.array([vchs[i]-dvchs/2. for i in range(len(vchs))])
levels = np.append(levels, np.array([vchs[-1]+dvchs/2.]))

cs = plt.contourf(Xnew, Ynew, vl0, levels = levels, cmap='RdBu', zorder = 2) #||||||||||||||||||
#plt.scatter(planet[0],planet[1],color='k',zorder=+2)
del cs.collections[4:4]
P = plt.contour(Xnew, Ynew, vl0, levels = levels, colors='black', linewidths = 1 ,zorder = 3) #||||||||||||||||

plt.scatter(0,0,color='gold',zorder = +4,s = 80, marker=(5, 1))
proxy = [plt.Rectangle((0,0),1,1,fc=pc.get_facecolor()[0]) for pc in cs.collections]
leg2 = plt.legend(proxy, [str(vchs[i]) for i in range(len(vchs))],loc="upper left",title="$v_{ch}$ [km/s]:")
plt.xlabel('$x$ [au]')
plt.ylabel('$y$ [au]')
plt.locator_params(nbins=3)
plt.axis("equal")
#plt.title('CHANNEL MAPS')
#plt.show()
plt.savefig('CM_NoPlanet.png')
################################################################################

# PLOT CHANNEL MAPS PERTURBED DISC WITHOUT VISCOSITY

fig,ax = plt.subplots()
conto=plt.contourf(Xnew, Ynew, vf[2,:,:]-vf0[2,:,:], zorder=0, levels = np.arange(-0.3,0.3,0.001) ,cmap='seismic_r') #|||||||||||||
cb=fig.colorbar(conto)
cb.set_label('$\Delta v_n$   [km/s]')
cb.ax.locator_params(nbins=5)

#plt.plot(dust[:,0],dust[:,1],zorder = 0, label ="dust", color = 'k')
plt.plot(disc[:,0],disc[:,1],zorder = 0, label ="disc",color='k')
dvchs = np.abs(vchs[1]-vchs[0])
levels = np.array([vchs[i]-dvchs/2. for i in range(len(vchs))])
levels = np.append(levels, np.array([vchs[-1]+dvchs/2.]))

cs = plt.contourf(Xnew, Ynew, vl, levels = levels, cmap='RdBu', zorder = 2) #||||||||||||||||||
plt.scatter(planet[0],planet[1],color='k',zorder=+2)
del cs.collections[4:4]
P = plt.contour(Xnew, Ynew, vl, levels = levels, colors='black', linewidths = 1 ,zorder = 3) #||||||||||||||||

plt.scatter(0,0,color='gold',zorder = +4,s = 80, marker=(5, 1))
proxy = [plt.Rectangle((0,0),1,1,fc=pc.get_facecolor()[0]) for pc in cs.collections]
#leg2 = plt.legend(proxy, [str(vchs[i]) for i in range(len(vchs))],loc="upper left",title="$v_{ch}$ [km/s]:")
plt.xlabel('$x$ [au]')
plt.ylabel('$y$ [au]')
plt.locator_params(nbins=3)
plt.axis("equal")
#plt.title('CHANNEL MAPS')
#plt.show()

plt.savefig('CM_PlanetNoVisc.png')

################################################################################

# PLOT CHANNEL MAPS PERTURBED DISC WITH VISCOSITY

fig,ax = plt.subplots()
conto=plt.contourf(Xnew, Ynew, vfS[2,:,:]-vf0[2,:,:], zorder=0, levels = np.arange(-0.15,0.15,0.001) ,cmap='seismic_r') #|||||||||||||
cb=fig.colorbar(conto)
cb.set_label('$\Delta v_n$   [km/s]')
cb.ax.locator_params(nbins=5)

#plt.plot(dust[:,0],dust[:,1],zorder = 0, label ="dust", color = 'k')
plt.plot(disc[:,0],disc[:,1],zorder = 0, label ="disc",color='k')
dvchs = np.abs(vchs[1]-vchs[0])
levels = np.array([vchs[i]-dvchs/2. for i in range(len(vchs))])
levels = np.append(levels, np.array([vchs[-1]+dvchs/2.]))

cs = plt.contourf(Xnew, Ynew, vlS, levels = levels, cmap='RdBu', zorder = 2) #||||||||||||||||||
plt.scatter(planet[0],planet[1],color='k',zorder=+2)
del cs.collections[4:4]
P = plt.contour(Xnew, Ynew, vlS, levels = levels, colors='black', linewidths = 1 ,zorder = 3) #||||||||||||||||

plt.scatter(0,0,color='gold',zorder = +4,s = 80, marker=(5, 1))
proxy = [plt.Rectangle((0,0),1,1,fc=pc.get_facecolor()[0]) for pc in cs.collections]
#leg2 = plt.legend(proxy, [str(vchs[i]) for i in range(len(vchs))],loc="upper left",title="$v_{ch}$ [km/s]:")
plt.xlabel('$x$ [au]')
plt.ylabel('$y$ [au]')
plt.locator_params(nbins=3)
plt.axis("equal")
#plt.title('CHANNEL MAPS')
#plt.show()

plt.savefig('CM_PlanetVisc.png')
