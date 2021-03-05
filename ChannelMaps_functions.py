import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import quad
import sys
import ChannelMaps_settings as s



def vKepler(R,M):

   M_cgs = M*s.Msun
   R_cgs = R*s.au
   v_cgs = np.sqrt(s.G*M_cgs/R_cgs)
   return v_cgs*10**(-5) # km/s

def create_Keplerian_velocity_field():

    vK = np.zeros((s.Nr,s.Nphi))
    for i in range(s.Nr):
        vK[:,i] = vKepler(s.R[0,i],s.M) * np.ones(s.Nphi)
    return vK


def upload_linear_perturbations():

    # dimensionless perturbations from linear perturbations computation
    s.Vlin = np.loadtxt('vtot.dat')
    s.Ulin = np.loadtxt('utot.dat')
    s.Dlin = np.loadtxt('stot.dat')

    s.Xlin = np.loadtxt('X.dat') # initialize linear mesh
    s.Ylin = np.loadtxt('Y.dat')

    x = s.Xlin[0,:]
    y = s.Ylin[:,0]

    # extract linear perturbations in the linear box side

    xl = x[(x>-s.x_match)&(x<s.x_match)]
    Nx = len(xl)
    indx_xl_start = np.argmin(np.abs(x-xl[0]))
    indx_xl_end = np.argmin(np.abs(x-xl[-1]))

    y_match = s.x_match**2/2
    yl = y[(y>-y_match)&(y<y_match)]
    Ny = len(yl)
    indx_yl_start = np.argmin(np.abs(y-yl[0]))
    indx_yl_end = np.argmin(np.abs(y-yl[-1]))

    vl = s.Vlin[indx_yl_start:indx_yl_end+1, indx_xl_start:indx_xl_end+1]
    ul = s.Ulin[indx_yl_start:indx_yl_end+1, indx_xl_start:indx_xl_end+1]
    dl = s.Dlin[indx_yl_start:indx_yl_end+1, indx_xl_start:indx_xl_end+1]

    if s.cw == 1:

        ul_temp = np.zeros(ul.shape)
        vl_temp = np.zeros(vl.shape)
        dl_temp = np.zeros(dl.shape)

        for i in range(Nx):
            for j in range(Ny):

                ul_temp[j,i] = ul[Ny-1-j,i]
                vl_temp[j,i] = - vl[Ny-1-j,i]
                dl_temp[j,i] = dl[Ny-1-j,i]

        ul = ul_temp.copy()
        vl = vl_temp.copy()
        dl = dl_temp.copy()


    # rescale to physical units
    ul = ul*s.csp*s.betap
    vl = vl*s.csp*s.betap
    dl = dl*s.betap

    xl = s.Rp + s.l*xl
    yl = s.l*yl

    return xl,yl,dl,ul,vl


def vKepler_plus_linear_pert(xl,yl,ul,vl,vK):

    vr = np.zeros((s.Nr,s.Nphi))
    vphi = np.zeros((s.Nr,s.Nphi))
    deltav = np.zeros((s.Nr,s.Nphi))

    vphi = - s.cw * vK.copy()

    for i in range(s.Nr):
       if s.X[0,i] >= (s.Rp - s.x_match*s.l) and s.X[0,i]<= (s.Rp + s.x_match*s.l):
           for j in range(s.Nphi):
               if s.Y[j,i] >= - s.x_match*s.l and s.Y[j,i] <= s.x_match*s.l and (s.PHI[j,i]<np.pi/2 or s.PHI[j,i]>3*np.pi/2):

                   indx = np.argmin(np.abs(s.X[0,i]-xl)) # index of linear pert grid that best approximate the point (x[i],y[j]) of the disc grid .....
                   indy = np.argmin(np.abs(s.Y[j,i]-yl))

                   if s.malpha == 0:
                       damp = 1
                   else:
                       damp = viscous_damping_factor(s.X[0,i])

                   vr[j,i] += ul[indy,indx]*damp
                   vphi[j,i] += vl[indy,indx]*damp
                   deltav[j,i] = (np.sqrt(vr[j,i]**2+vphi[j,i]**2) - vK[j,i])/vK[j,i]

    return vr,vphi,deltav

## functions for viscous damping factor

def visc_integrand(a):
    return np.abs(a**(-1.5)-1)*a**s.q

def visc_integral(up):
    return quad(visc_integrand,1,up)[0]

def viscous_damping_factor(r):  # Eq. (44) Bollati et al. 2021
    coeff = s.malpha*7/(6*s.hr)
    module_integral = np.abs(visc_integral(r/s.Rp))
    return np.exp(-coeff*module_integral)



# NONLINEAR PERTURBATIONS

def extract_burgers_IC(): # take the initial condition for burgers equation from linear density perturbation ...


   eta_max = 25 # Semi-width of the support of the azimuthal profile of the linear density perturbation (Fig. 2-right panel Bollati et al. 2021 -> eta_max does not depend on x_match)
                # (This value does not depend on betap)
   x = s.Xlin[0,:]
   y = s.Ylin[:,0]

   index = np.argmin(np.abs(x-s.x_match))
   profile = s.Dlin[:,index]/np.sqrt(np.abs(s.x_match))
   y_match = -np.sign(s.x_match)*0.5*s.x_match**2
   y_cut = y[(y-y_match>-eta_max) & (y-y_match<eta_max)]
   eta = y_cut - y_match*np.ones(len(y_cut))
   profile = profile[(y-y_match>-eta_max) & (y-y_match<eta_max)]

   # set t0 (t correponding to the x_match)
   s.t0 = t(s.Rp + s.l*s.x_match)

   # set eta_tilde:
   for i in range(len(eta)):
      if profile[i] == 0 and eta[i]>-10 and eta[i]<0 :
         zero = eta[i]
      elif i!=(len(eta)-1) and profile[i]*profile[i+1]<0 and eta[i]>-10 and eta[i]<0:
         zero = 0.5*(eta[i]+eta[i+1])
   s.eta_tilde = -zero

   # set C:
   deta = eta[1]-eta[0]
   profile0 = profile[eta<-s.eta_tilde]
   C0 = -np.trapz(profile0, dx =  deta)
   s.C = (s.indad + 1)*s.betap*C0/2**(3/4)

   print('     eta_tilde = ',s.eta_tilde)
   print('     C0 = ',C0)
   print('     t0 = ',s.t0)

   return eta,profile



def solve_burgers(eta,profile): # Solve eq. (10) Bollati et al. 2021

   profile = profile * (s.indad+1) * s.betap / 2**(3/4) # Eq. (15) Bollati et al. 2021

   deta = eta[1]-eta[0]

   tf_th = 300 # time required to develop N-wave for betap = 1
   tf = tf_th/s.betap # time required to display N-wave for generic betap, Eq. (39) Rafikov 2002
   dt = 0.02
   Nt = int(tf/dt) + 1

   eta_min = -s.eta_tilde-np.sqrt(2*s.C*tf) - 3  # Eq. (18) Bollati et al. 2021
   eta_max = -s.eta_tilde+np.sqrt(2*s.C*tf) + 3

   # extend the profile domain due to profile broadening during evolution ....
   extr = eta[-1]
   while extr<eta_max:
      eta = np.append(eta,eta[-1]+deta)
      extr = eta[-1]
      profile = np.append(profile,0)
   extr = eta[0]
   while extr>eta_min:
      eta = np.insert(eta,0,eta[0]-deta)
      extr = eta[0]
      profile = np.insert(profile,0,0)

   Neta = len(eta) # number of centers
   a = eta[0] - deta/2
   b = eta[-1] + deta/2

   L  = b-a   # spatial grid length
   T  = tf    # final time

   # set time array
   time = np.array([dt*i for i in range(Nt)])
   x = np.zeros(Neta+1) # cells edges
   for i in range(0,Neta+1):
       x[i] = a+i*deta

   solution = np.zeros((Neta,Nt), dtype=float)
   # linear solution as initial condition
   for i in range(0,Neta):
      solution[i,0] = profile[i]

   # define flux vector
   F = np.zeros(Neta+1) #there is a flux at each cell edge!!!!!

   # define the flux function of Burgers equation
   def flux(u):
       return 0.5*u**2;

   # define the Central difference numerical flux---> ritorna la media aritmetica dei flux sx e dx passati
   def CentralDifferenceFlux(uL,uR):
       # compute physical fluxes at left and right state
       FL = flux(uL)
       FR = flux(uR)
       return 0.5*(FL+FR)

   def GodunovNumericalFlux(uL,uR):
     # compute physical fluxes at left and right state
     FL = flux(uL)
     FR = flux(uR)
     # compute the shock speed
     s = 0.5*(uL + uR)
     # from Toro's book
     if (uL >= uR):
         if (s > 0.0):
             return FL
         else:
             return FR
     else:
         if (uL > 0.0):
             return FL
         elif (uR < 0.0):
             return FR
         else:
             return 0.0

   def NumericalFlux(uL,uR):
       # return CentralDifferenceFlux(uL,uR)
       return GodunovNumericalFlux(uL,uR);

   # time integrate
   for n in range(0,Nt-1):

      # estimate the CFL
      CFL = max(abs(solution[:,n])) * dt / deta
      if CFL > 0.5:
         print("Warning: CFL > 0.5")

      # compute the interior fluxes
      for i in range(1,Neta):
          uL = solution[i-1,n]
          uR = solution[i,n]
          F[i] = NumericalFlux(uL,uR)

      # compute the left boundary flux
      if solution[0,n] < 0.0:
          uL = 2.0*solution[0,n] - solution[1,n]
      else:
          uL = solution[0,0]
      uR = solution[0,n]
      F[0] = NumericalFlux(uL,uR)

      # compute the right boundary flux
      if solution[Neta-1,n] > 0.0:
          uR = 2.0 * solution[Neta-1,n] - solution[Neta-2,n]
      else:
          uR = solution[Neta-1,0]
      uL = solution[Neta-1,n]
      F[Neta] = NumericalFlux(uL,uR)

      # update the state
      for i in range(0,Neta):
          solution[i,n+1] = solution[i,n] - dt / deta * (F[i+1] - F[i])

   # plot Fig. 3 Bollati et al. 2021
   '''
   plt.plot(eta, solution[:,0], label = "$t=t_0+$ "+str(0*dt))
   plt.plot(eta, solution[:,int(Nt/20)], label = "$t=t_0+$ "+str(round(dt*Nt/10,2)))
   plt.plot(eta, solution[:,int(Nt/10)], label = "$t=t_0+$ "+str(round(dt*Nt/6,2)))
   plt.plot(eta, solution[:,int(Nt/5)], label = "$t=t_0+$ "+str(round(dt*Nt/3,2)))
   plt.plot(eta, solution[:,int(Nt-1)], label = "$t=t_0+$ "+str(round(T,2)))
   plt.legend()
   plt.xlabel("$\eta$")
   plt.ylabel("$\chi(t,\eta)$")
   plt.grid(True)
   plt.title('$\chi$ "evolution" $r > r_p$')
   plt.show()
   '''

   solution_inner = np.zeros(solution.shape) # solution for r < Rp (and disc rotating counterclockwise)
   for i in range(Neta):
      for j in range(Nt):
         solution_inner[i,j] = solution[int(Neta-1-i),j]

   eta_inner = - eta[::-1]

   return time, eta, solution, eta_inner, solution_inner


def phi_wake(r): # Eq. (4) Bollati et al. 2021

   rr = r / s.Rp
   return -s.cw * np.sign(r-s.Rp)*(1/s.hr)*( rr**(s.q-0.5)/(s.q-0.5) - rr**(s.q + 1)/(s.q+1)-3/((2*s.q-1)*(s.q+1)) )


def Eta(r,phi): # Eq. (14) Bollati et al. 2021

   coeff = 1.5/s.hr
   phi_w = mod2pi(phi_wake(r))
   deltaphi = phi - phi_w

   if deltaphi > np.pi:
       deltaphi = deltaphi -2*np.pi
   elif deltaphi < -np.pi:
       deltaphi = deltaphi + 2*np.pi

   return coeff*deltaphi


def mod2pi(phi):
   if phi>=0:
      return phi%(2*np.pi)
   else:
      if np.abs(phi)<np.pi*2:
         return phi+2*np.pi
      else:
         phi = -phi
         resto = phi%(2*np.pi)
      return -resto+2*np.pi


# Equation (43) Rafikov 2002    (Eq. 13 Bollati et al. 2021)

def t_integrand(x):
   rho = 5*s.q + s.p
   w = rho/2 - 11/4
   return np.abs( 1 - x**(1.5) )**(1.5) * x**w

def t_integral(up):
   return  quad(t_integrand,1,up)[0]

def t(r):
   module_integral = np.abs( t_integral(r/s.Rp) )
   coeff = 3*s.hr**(-5/2)/(2**(5/4))
   return coeff*module_integral


# Equation (12) Bollati et al. 2021

def g(r):

    coeff = 2**0.25*s.hr**0.5
    term1 = (r/s.Rp)**(0.5*(1-s.p-3*s.q))
    term2 = np.abs( (r/s.Rp)**(-1.5)-1 )**(-0.5)
    return coeff*term1*term2




def Lambda_fu(r):       # Eq. (28) Bollati et al. 2021

    coeff = 2**0.75*s.csp*s.hr**(-0.5)/(s.indad+1)
    term1 = np.abs( (r/s.Rp)**(-1.5)-1 )**0.5
    term2 = (r/s.Rp)**(0.5*(s.p+s.q-1))
    return coeff*term1*term2


def Lambda_fv(r):      # Eq. (29) Bollati et al. 2021

    coeff = 2**0.75*s.csp*s.hr**0.5/(s.indad+1)
    term1 = np.abs( (r/s.Rp)**(-1.5)-1 )**(-0.5)
    term2 = (r/s.Rp)**(0.5*(s.p-s.q-3))
    return coeff*term1*term2


def compute_nonlinear_pert():

   print('  * Extracting Burgers initial condition from linear density perturbation ...')
   eta, profile = extract_burgers_IC()

   print('  * Solving Burgers equation ...')
   time, eta, solution, eta_inner, solution_inner  = solve_burgers(eta,profile)

   print('  * Computing nonlinear perturbations ...')

   tf = time[-1]

   dnl = np.zeros((s.Nr,s.Nphi))
   unl = np.zeros((s.Nr,s.Nphi))
   vnl = np.zeros((s.Nr,s.Nphi))

   r = s.R[0,:]
   phi = s.PHI[:,0]

   for i in range(s.Nr):
       rr = r[i]
       if (rr < (s.Rp - s.x_match*s.l)) or (rr >(s.Rp + s.x_match*s.l)): # non sono nell'annulus del linear regime .....
           for j in range(s.Nphi):
               pphi = phi[j]


               # COMPUTATION OF Chi

               # change coordinates of the grid point (rr,pphi) to (t1,eta1)
               t1 = t(rr)
               eta1 = Eta(rr,pphi)

               if t1 < (tf + s.t0):   # use numerical solution before the profile develops N-wave

                   index_t = np.argmin( np.abs(s.t0 + time - t1) )

                   # the density (Chi) azimuthal profile is flipped along the azimuthal direction
                   # both passing from r > Rp to r < Rp and from cw = -1 to cw = +1:

                   if s.cw*(rr-s.Rp) < 0:
                       if eta1 > eta[-1] or eta1 < eta[0]:
                           Chi = 0
                       else:
                           index_eta = np.argmin( np.abs(eta - eta1) )
                           Chi = solution[index_eta,index_t]
                   else:
                        if eta1 > eta_inner[-1] or eta1 < eta_inner[0]:
                            Chi = 0
                        else:
                            index_eta = np.argmin(np.abs(eta_inner - eta1))
                            Chi = solution_inner[index_eta,index_t]


               else: # for large t use the analytical N-wave shape (Eq. 17 Bollati et al. 2021)

                   extr_left = +s.cw * np.sign(rr-s.Rp) * s.eta_tilde - np.sqrt(2*s.C*(t1-s.t0))
                   extr_right = +s.cw * np.sign(rr-s.Rp) * s.eta_tilde + np.sqrt(2*s.C*(t1-s.t0))

                   if eta1 > extr_left and eta1 < extr_right:
                       Chi = (-s.cw * np.sign(rr-s.Rp) * eta1 + s.eta_tilde) / (t1-s.t0)  #eq.(29) nonlinear.pdf
                   else:
                       Chi = 0

               # COMPUTE DENSITY AND VELOCITY PERTURBATIONS
               g1 = g(rr)
               dnl[j,i] = Chi * 2 / (g1*(s.indad + 1))     # Eq. (11) Bollati et al. 2021

               Lfu = Lambda_fu(rr)
               Lfv = Lambda_fv(rr)
               unl[j,i] = np.sign(rr-s.Rp) * Lfu * Chi           # Eq. (23) Bollati et al. 2021
               vnl[j,i] = np.sign(rr-s.Rp) * Lfv * Chi * (-s.cw) # Eq. (24) Bollati et al. 2021 (the sign of v is reversed if we change cw)

   return dnl, unl, vnl



def add_nonlinear_pert(unl,vnl,vr,vphi,deltav,vK):

       for i in range(s.Nr):
          rr = s.R[0,i]
          if rr < (s.Rp - s.x_match*s.l) or rr > (s.Rp + s.x_match*s.l):  # outisde annulus of linear regime !!!
              for j in range(s.Nphi):
                  if s.malpha == 0:
                      damp = 1
                  else:
                      damp = viscous_damping_factor(rr)

                  vr[j,i] += unl[j,i]*damp
                  vphi[j,i] += vnl[j,i]*damp
                  deltav[j,i] = (np.sqrt(vr[j,i]**2 + vphi[j,i]**2) - vK[j,i])/vK[j,i]

       return vr,vphi,deltav


def merge_density_lin_nonlin(xl,yl,dl,dnl):

    for i in range(s.Nr):
       if s.X[0,i] >= (s.Rp - s.x_match*s.l) and s.X[0,i]<= (s.Rp + s.x_match*s.l):
           for j in range(s.Nphi):
               if s.Y[j,i] >= - s.x_match*s.l and s.Y[j,i] <= s.x_match*s.l and (s.PHI[j,i]<np.pi/2 or s.PHI[j,i]>3*np.pi/2):

                   indx = np.argmin(np.abs(s.X[0,i]-xl)) # index of linear pert grid the best approximate the point (x[i],y[j]) of the disc grid .....
                   indy = np.argmin(np.abs(s.Y[j,i]-yl))
                   if s.malpha == 0:
                       damp = 1
                   else:
                       damp = viscous_damping_factor(s.X[0,i])
                   dnl[j,i] += dl[indy,indx]*damp
    return dnl


## FUNCTIONS FOR PLOTS ....

def make_contourplot(field, bar_label = None, title = None, saveas = None, WithChannels = None, vz_field = None):

    fig, ax = plt.subplots()

    #fig.set_size_inches(10,10)

    # set contour levels
    max_field = round(field.max(),2)
    d = 2*max_field/1000
    levels = np.arange(-max_field*(1+0.1),max_field*(1+0.1),d)

    contour = plt.contourf(s.X,s.Y,field, levels = levels ,cmap='RdBu',zorder=0)

    # colorbar
    cb = fig.colorbar(contour, ticks=[-max_field,-max_field/2,0,max_field/2,max_field])
    if bar_label != None:
        cb.set_label(bar_label)
    cb.ax.locator_params(nbins=5)

    # plot disc edge and star
    plt.plot(s.disc_edge[:,0],s.disc_edge[:,1],color='k')
    plt.scatter(0,0,label="star",color='gold',zorder = +2,s = 80, marker=(5, 1))

    # plot channel maps
    if WithChannels == True:

        channel_maps = get_channel_maps(vz_field)

        # set channel maps levels
        levelsch = [s.vchs[0] - s.delta_vch]
        Nvch = len(s.vchs)
        levelsch += [0.5*(s.vchs[i]+s.vchs[i+1]) for i in range(0,Nvch-1)]
        levelsch += [s.vchs[-1] + s.delta_vch]
        levelsch = np.array(levelsch)

        # plot channel maps
        if Nvch > 1:
            chmps = plt.contourf(s.X, s.Y, channel_maps, levels = levelsch, cmap = 'RdBu', zorder = 2)
        else:
            chmps =  plt.contourf(s.X, s.Y, channel_maps, levels = levelsch, colors = 'orange', zorder = 2)
        del chmps.collections[4:4]
        cm = plt.contour(s.X, s.Y, channel_maps, levels = levelsch, colors='black', linewidths = 1 ,zorder = 3)

        # legend
        proxy = [plt.Rectangle((0,0),1,1,fc=pc.get_facecolor()[0]) for pc in chmps.collections]
        leg2 = plt.legend(proxy, [str(s.vchs[i]) for i in range(len(s.vchs))],loc="upper left",title="$v_{ch}$ [km/s]:")

        # disable limited range when plotting channel maps
        plt.axis('equal')
        s.xrange = 0
        s.yrange = 0

    if title != None:
        plt.title(title)


    # set range
    if np.ndim(s.xrange) == 0 and np.ndim(s.yrange) == 0:
        plt.axis('equal')
    else:
        if np.ndim(s.xrange)>0:
            plt.xlim(s.xrange)

        if np.ndim(s.yrange)>0:
            plt.ylim(s.yrange)


    plt.locator_params(nbins=7) # number of ticks per axis
    plt.xlabel("$x$   [au]")
    plt.ylabel("$y$   [au]")

    if s.show:
       plt.show()
    else:
       plt.savefig(saveas)



# FUNCTIONS FOR ROTATIONS

def get_velocity_Cartesian_components(vr,vphi):

    v_field = np.zeros((s.Nr,s.Nphi,3))
    v_field[:,:,0] = - vphi * np.sin(s.PHI) + vr * np.cos(s.PHI)  # funziona sia che la rotazione sia cw = \pm 1.
    v_field[:,:,1] = vphi * np.cos(s.PHI) + vr * np.sin(s.PHI)

    return v_field


def Rx(a):
   R = [[1,0,0],[0,np.cos(a),-np.sin(a)],[0, np.sin(a), np.cos(a)]]
   return R
def Ry(b):
   R = [[np.cos(b),0,np.sin(b)],[0,1,0],[-np.sin(b),0,np.cos(b)]]
   return R
def Rz(g):
   R = [[np.cos(g),-np.sin(g),0],[np.sin(g),np.cos(g),0],[0,0,1]]
   return R

def rotate_meshgrid():

        for i in range(s.Nr):
            for j in range(s.Nphi):
                # rotation around disc normal axis, needed to match the required PAp
                s.X[j,i], s.Y[j,i], temp = np.dot( Rz(s.xi), np.array([s.X[j,i],s.Y[j,i],0]) )

                # rotation araound (sky plane) x-axis to get the required inclination!
                s.X[j,i], s.Y[j,i], temp = np.dot( Rx(s.i), np.array([s.X[j,i],s.Y[j,i],0]) )

                # rotation around disc normal axis to get the given PA!
                s.X[j,i], s.Y[j,i], temp = np.dot( Rz(s.PA), np.array([s.X[j,i],s.Y[j,i], 0]) )

        for i in range(200):
            s.disc_edge[i,0],s.disc_edge[i,1], temp = np.dot( Rx(s.i), np.array([s.disc_edge[i,0],s.disc_edge[i,1],0]) )
            s.disc_edge[i,0],s.disc_edge[i,1], temp = np.dot( Rz(s.PA), np.array([s.disc_edge[i,0],s.disc_edge[i,1],0]) )


def rotate_velocity_field(v_field):

    for i in range(s.Nr):
        for j in range(s.Nphi):

            # rotation around disc normal axis, needed to match the required PAp
            v_field[j,i,:] = np.dot( Rz(s.xi), v_field[j,i,:] )

            # rotation araound (sky plane) x-axis to get the required inclination!
            v_field[j,i,:] = np.dot( Rx(s.i), v_field[j,i,:] )

            # rotation around disc normal axis to get the given PA!
            v_field[j,i,:] = np.dot( Rz(s.PA), v_field[j,i,:] )


    return v_field


def get_channel_maps(vz_field):

    Nvch = len(s.vchs)
    channel_maps = 100 * np.ones(vz_field.shape) # Arbitrary large value
    for i in range(len(channel_maps[:,0])):
        for j in range(len(channel_maps[0,:])):
            for ch in range(Nvch):
                if (vz_field[i,j] > s.vchs[ch] - s.delta_vch and vz_field[i,j] < s.vchs[ch] + s.delta_vch):
                    channel_maps[i,j] = vz_field[i,j]

    return channel_maps
