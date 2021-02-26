import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.integrate as ode
from scipy.integrate import quad
import sys


# KEPLERIAN VELOCITY

# ([r]=au, [M] = Msun)-->[v]=km/s

def v(r,M):
   G = 6.67*10**(-8)
   Msun = 1.989*10**33 #g
   M = M*Msun
   au = 1.49597870*10**13 #cm
   r = r*au
   return np.sqrt(G*M/r)*10**(-5)


# COORDINATE TRANSFORMATIONS

# cartesian
def cartesian(r,phi):
   x = r*np.cos(phi)
   y = r*np.sin(phi)
   return x,y


# polar
# returns azimuth in [0,2pi)
def polar(x,y):
   if x==0 and y==0:
      r=0
      phi = 0
   else:
      r = np.sqrt(x**2+y**2)
      if x>0:
         phi = np.arctan(y/x)
         if phi<0:
            phi = 2*np.pi + phi
      elif x<0:
         phi = np.pi + np.arctan(y/x)
      else:
         phi = 0.5*np.pi*np.sign(y)
   return r,phi


# KEPLERIAN VELOCITY FIELD

def create_Kepler(x,y, N, clockwise, Rdisk,M): # N = grid linear dimension; x.shape = y.shape = (N,)
   vKepler = np.zeros((N,N))
   vx = np.zeros((N,N))
   vy = np.zeros((N,N))
   for i in range(N):    #cycle on rows
      for j in range(N): #cycle on columns
         r,phi = polar(x[j],y[i])
         if r==0:
            continue
         if r<= Rdisk:
            vKepler[i,j] = v(r,M)
            if clockwise == -1: #(counterclockwise)
               vx[i,j] = -vKepler[i,j]*np.sin(phi)
               vy[i,j] = vKepler[i,j]*np.cos(phi)
            if clockwise == 1:
               vx[i,j] = +vKepler[i,j]*np.sin(phi)
               vy[i,j] = -vKepler[i,j]*np.cos(phi)
         else:
            vKepler[i,j] = 100 #very big velocity outside disc
   return vKepler,vx,vy   # []=km/s


# LINEAR PERTURBATIONS

def createUV(rp): #'vtot.dat','utot.dat','X.dat','Y.dat'

   V = np.loadtxt('vtot_final.dat') #velocità adimensionali
   U = np.loadtxt('utot_final.dat')
   X = np.loadtxt('X_final.dat')    #griglia adimensionale
   Y = np.loadtxt('Y_final.dat')

   x = X[0,:]
   y = Y[:,0]

   #griglia ristretta adimensionale xn
   xmax =  2
   xn = x[(x>-xmax)&(x<xmax)]
   xstart = np.argmin(np.abs(x-xn[0]))
   xend = np.argmin(np.abs(x-xn[-1]))

   #griglia ristretta adimensionale yn
   ymax = xmax**2/2
   yn = y[(y>-ymax)&(y<ymax)]
   ystart = np.argmin(np.abs(y-yn[0]))
   yend = np.argmin(np.abs(y-yn[-1]))
    #griglia adimensionale ristretta

   #velocità ristretta adimensionale un,vn
   vn = V[ystart:yend+1,xstart:xend+1]
   un = U[ystart:yend+1,xstart:xend+1]
   return un,vn,xn,yn


def createUVS(rp): #'vtot.dat','utot.dat','X.dat','Y.dat'

   V = np.loadtxt('vtot.dat') #velocità adimensionali
   U = np.loadtxt('utot.dat')
   S = np.loadtxt('stot.dat')

   X = np.loadtxt('X.dat')    #griglia adimensionale
   Y = np.loadtxt('Y.dat')

   x = X[0,:]
   y = Y[:,0]

   #griglia ristretta adimensionale xn
   xmax =  2
   xn = x[(x>-xmax)&(x<xmax)]
   xstart = np.argmin(np.abs(x-xn[0]))
   xend = np.argmin(np.abs(x-xn[-1]))

   #griglia ristretta adimensionale yn
   ymax = xmax**2/2
   yn = y[(y>-ymax)&(y<ymax)]
   ystart = np.argmin(np.abs(y-yn[0]))
   yend = np.argmin(np.abs(y-yn[-1]))
    #griglia adimensionale ristretta

   #velocità ristretta adimensionale un,vn
   vn = V[ystart:yend+1,xstart:xend+1]
   un = U[ystart:yend+1,xstart:xend+1]
   sn = S[ystart:yend+1,xstart:xend+1]
   return sn,un,vn,xn,yn



def totalUV(l,N,x,y, unl,vnl, un, vn,xn,yn, alfa,m,hr,rp,nu): # concatenate linear and nonlinear perturbation in single array
   utot = np.zeros(unl.shape)
   vtot = np.zeros(vnl.shape)
   utotv = np.zeros(unl.shape)
   vtotv = np.zeros(vnl.shape)
   for i in range(N):
       for j in range(N):
          r,phi = polar(x[i],y[j])
          damp = viscous_damping(alfa,m,hr,r,rp,nu)
          if x[i]<= rp-2*l or x[i]>= rp+2*l or y[j]<=-2*l or y[j]>=2*l:
             utot[j,i] = unl[j,i]
             vtot[j,i] = vnl[j,i]
             utotv[j,i] = unl[j,i]*damp
             vtotv[j,i] = vnl[j,i]*damp
          else:
             indx = np.argmin(np.abs(x[i]-xn))
             indy = np.argmin(np.abs(y[j]-yn))
             utot[j,i] = un[indy,indx]
             vtot[j,i] = vn[indy,indx]
             utotv[j,i] = un[indy,indx]*damp
             vtotv[j,i] = vn[indy,indx]*damp
   return utot,vtot, utotv, vtotv

def totalUVS(l,N,x,y,dnl,unl,vnl, sn, un, vn,xn,yn, alfa,m,hr,rp,nu): # concatenate linear and nonlinear perturbation in single array
   utot = np.zeros(unl.shape)
   vtot = np.zeros(vnl.shape)
   stot = np.zeros(dnl.shape) #
   utotv = np.zeros(unl.shape)
   vtotv = np.zeros(vnl.shape)
   stotv = np.zeros(dnl.shape)#

   for i in range(N):
       for j in range(N):
          r,phi = polar(x[i],y[j])
          damp = viscous_damping(alfa,m,hr,r,rp,nu)
          if x[i] <= rp-2*l or x[i]>= rp+2*l or y[j]<=-2*l or y[j]>=2*l:
             utot[j,i] = unl[j,i]
             vtot[j,i] = vnl[j,i]
             stot[j,i] = dnl[j,i] #
             utotv[j,i] = unl[j,i]*damp
             vtotv[j,i] = vnl[j,i]*damp
             stotv[j,i] = dnl[j,i]*damp #
          else:
             indx = np.argmin(np.abs(x[i]-xn))
             indy = np.argmin(np.abs(y[j]-yn))

             utot[j,i] = un[indy,indx]
             vtot[j,i] = vn[indy,indx]
             stot[j,i] = sn[indy,indx] #

             utotv[j,i] = un[indy,indx]*damp
             vtotv[j,i] = vn[indy,indx]*damp
             stotv[j,i] = sn[indy,indx]*damp #

   return stot, utot,vtot, stotv, utotv, vtotv #



#Npoints=points of the linear dimension of the square around the planet in linear regime
#U,V velocity field on the square

def create_perturbed(rp,l,N,x,y,vx0,vy0,vk0,un,vn,xn,yn,viscb,alfa,m,hr,nu):
   Nx = len(xn)
   Ny = len(yn)
   vKepler = np.zeros((N,N))
   vx = np.zeros((N,N))
   vy = np.zeros((N,N))
   for i in range(N):
      for j in range(N):
        vKepler[i,j] = vk0[i,j]
        vx[i,j] = vx0[i,j]
        vy[i,j] = vy0[i,j]
   for i in range(N):
       for j in range(N):
          r,phi = polar(x[i],y[j])
          if x[i]<= rp-2*l or x[i]>= rp+2*l or y[j]<=-2*l or y[j]>=2*l:
             continue
          else:
             indx = np.argmin(np.abs(x[i]-xn))
             indy = np.argmin(np.abs(y[j]-yn))
             if viscb == True: # A viscous damping is applied
                  damp = viscous_damping(alfa,m,hr,r,rp,nu)
                  #print(damp)
                  vn[indy,indx] = vn[indy,indx]*damp
                  un[indy,indx] = un[indy,indx]*damp
             vx[j,i] += un[indy,indx]
             vy[j,i] += vn[indy,indx]
             vKepler[j,i] = np.sqrt(vx[j,i]**2+vy[j,i]**2)
   return vKepler,vx,vy



'''
   #
   x_begin = rp-2*l #2 is the adimensional end of linear regime Rafikov
   y_begin = -2*l
   tempx = np.abs(x-x_begin)
   indexx = tempx.argmin() #second index, column
   tempy =np.abs(y-y_begin)
   indexy = tempy.argmin() #first index, row
   #così ho trovato gli indici da cui inizia il patche in basso a sinistra rispetto agli array x e y che def grid disc
   for i in range(Npoints):  #cycle on rows
      for j in range(Npoints): #cycle on columns
         r,phi = polar(x[j+indexx],y[i+indexy])
         if r==0:
            continue
         if r<= Rdisk: #aggiorno le componenti dei vettori
            vx[indexy+i,indexx+j]+=U[i,j]#
            vy[indexy+i,indexx+j]+=V[i,j]#
            vKepler[indexy+i,indexx+j] = np.sqrt(vx[indexy+i,indexx+j]**2+vy[indexy+i,indexx+j]**2)
   return vKepler,vx,vy
'''




# WAKE SHAPE (equation (44) Rafikov2 with a sign '+' instead of '-' after phi0)

def wake_shape(nu,rp,phip,r,eps,cw):
   rad = r/rp
   if rad >0:
      return phip-cw*np.sign(r-rp)*(1/eps)*((rad)**(nu-0.5)/(nu-0.5)-(rad)**(nu+1)/(nu+1)-3/((2*nu-1)*(nu+1)))
   else:
      return 0

def create_wake(Nwake,Rdisk,rp,phip,nu,eps,cw): #eps = h/r
   radii = np.linspace(Rdisk/20,Rdisk,Nwake)
   rout = radii[radii>=rp]
   rin = radii[radii<rp]
   wakeout = np.zeros((2,len(rout))) #wake cartesian coordinates for r>rp
   wakein = np.zeros((2,len(rin)))   #wake cartesian coordinates for r<rp
   for k in range(len(rout)):
      wakeout[:,k] = cartesian(rout[k],wake_shape(nu,rp,phip,rout[k],eps,cw))
   for k in range(len(rin)):
      wakein[:,k] = cartesian(rin[k],wake_shape(nu,rp,phip,rin[k],eps,cw))
   wake = np.append(wakein,wakeout,axis=1)   #wake cartesian coordinates for r \in (Rdisc/20,Rdisc)
   wake_polar = np.zeros(wake.shape)         #wake polar coordinates for r \in (Rdisc/20,Rdisc)
   for i in range(len(wake[0,:])):
      wake_polar[0,i],wake_polar[1,i] = polar(wake[0,i],wake[1,i])
   return wake, wake_polar
   #return --> ((2,Nwake)) cartesian , ((2,Nwake)) polar


# FUNCTION FOR SHOCK COORDINATES (asymptotic nonlinear perturbations will be applied after shock formation)

# From equations (40) and (43) of Rafikov2 we have
# (t0 + 0.79M1/Mp)(2^{5/4}/3)(hp/rp)^{5/2} = | int_1^{rsh/rp} |s^{3/2}-1|^{3/2}s^{(5\nu+\delta)/2-11/4}ds |. (*)
# The function find_rhs computes the right-hand-side of this equation.

def find_rhs(nu,delta,rsurp_max,N): #rsup_max = Rdisk/rp, N=Nr
   upp = np.linspace(0,rsurp_max,N)
   rsurp_dx = upp[upp>=1] #r/rp for r>rp
   rsurp_sx = upp[upp<=1] #r/rp for r<rp
   #
   integrals = np.vectorize(integral)
   rhs_dx = np.abs(integrals(rsurp_dx, nu, delta)) #rhs for r>rp
   rhs_sx = np.abs(integrals(rsurp_sx, nu, delta)) #rhs for r<rp
   return rsurp_sx,rsurp_dx,rhs_sx,rhs_dx


# integrand of rhs in (*)

def integrand(s,nu,delta):
   rho = 5*nu + delta
   w = rho/2 -11/4
   return np.abs(s**(3/2)-1)**(3/2)*s**w


# compute integral in (*)

def integral(up, nu, delta):
   return  quad(integrand,1,up, args=(nu,delta))[0]


# coordinate t eq. (43) Rafikov2

def t(r,rp,eps,nu,delta):
   module_integral = np.abs(integral(r/rp,nu,delta))
   coeff = 3*eps**(-5/2)/(2**(5/4))
   return coeff*module_integral


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


def shock_coordinates(nu,delta,Rdisk,rp,phip,Nr,t0,betap,hr,cw):
   rsx,rdx,rhssx,rhsdx = find_rhs(nu,delta,Rdisk/rp,Nr) #rsx, rdx are in unit of rp
   #
   lhs = (t0+0.79/betap)*(2**(5/4)*hr**(5/2)/3)
   indx_dx = np.argmin(np.abs(lhs-rhsdx))
   indx_sx = np.argmin(np.abs(lhs-rhssx))
   rout = rdx[indx_dx]*rp
   rin = rsx[indx_sx]*rp
   #
   phiout = mod2pi( wake_shape(nu,rp,phip,rout,hr,cw) )
   phiin = mod2pi( wake_shape(nu,rp,phip,rin,hr, cw) )
   xin,yin = cartesian(rin,phiin)
   xout,yout = cartesian(rout,phiout)
   return xin,yin,rin,phiin,xout,yout,rout,phiout



# SHU VISCOUS DAMPING (to be applied to velocity perturbations)

#****** must be improved *******

def v_integrand(s,nu):
   return np.abs(s**(-3/2)-1)*s**nu

def v_integral(up, nu):
   return quad(v_integrand,1,up, args=(nu))[0]

def viscous_damping(alfa,m,hr,r,rp,nu): #nu exp di c
   coeff = alfa*7*m/(6*hr)
   module_integral = np.abs(v_integral(r/rp,nu))
   return np.exp(-coeff*module_integral)


# NONLINEAR PERTURBATIONS

# coordinate eta eq. (33) Rafikov2

def eta(r,phi,eps, nu, rp, phip, cw): #eps = h/r
   coeff = 3/(2*eps)
   phi_wake = mod2pi(wake_shape(nu, rp, phip, r, eps, cw))
   deltaphi = phi - phi_wake #phi is in mod2pi
   if np.abs(deltaphi)>np.pi:
      if phi<=np.pi:
         deltaphi = deltaphi + 2*np.pi
      else:
         deltaphi = deltaphi - 2*np.pi
   return coeff*deltaphi


# equation (34) Rafikov2 using power-law discs Sigma = Sigmap(r/rp)^-delta and c = cp(r/rp)^-nu

def g(r,eps,rp,delta,nu):
   if r>0:
      coeff = 2**(1/4)*eps**(1/2)
      term1 = (r/rp)**((1-delta-3*nu)/2)
      term2 = np.abs((r/rp)**(-3/2)-1)**(-1/2)
      return coeff*term1*term2
   else:
      return 0


# Lambda_p*f_u(r) and Lambda_p*f_u(r), equations (44) and (45) of nonlinear.pdf

def Lfur(r,rp,cp,eps,delta,nu,indad):
   if r>0:
      coeff = 2**(3/4)*cp*eps**(-1/2)/(indad+1)
      term1 = np.abs((r/rp)**(-3/2)-1)**(1/2)
      term2 = (r/rp)**((delta+nu-1)/2)
      return coeff*term1*term2
   else:
      return 0

def Lfvr(r,rp,cp,eps,delta,nu,indad):
   if r>0:
      coeff = 2**(3/4)*cp*eps**(1/2)/(indad+1)
      term1 = np.abs((r/rp)**(-3/2)-1)**(-1/2)
      term2 = (r/rp)**((delta-nu-3)/2)
      return coeff*term1*term2
   else:
      return 0

def linear_perturbations(x,y,N,Rdisk,rp,phip, hr,nu,cw,xc,u,v,xIc,uI,vI): #qui u,uI,v,vI sono i profiles linear di u e v

   ul = np.zeros((N,N))
   vl = np.zeros((N,N))
   for i in range(N):    #cycle on columns
      for j in range(N): #cycle on rows
         r,phi = polar(x[i],y[j])
         # Perturbations are applied outside the linear regime square around the planet
         if r<Rdisk and (r<rp*(1.+4.*hr/3.) and r>rp*(1.-4.*hr/3.)):
            #compute (t1,eta1) corresponding to coordinates (r,phi) of the grid point considered
            eta1 = eta(r,phi,hr, nu, rp, phip, cw) #dep. on cw because phi_wake dep. on cw
            if r>rp:
               inde = np.argmin(np.abs(eta1-xc))
               ul[j,i] = u[inde]
               vl[j,i] = v[inde]
            else:
               inde = np.argmin(np.abs(eta1-xIc))
               ul[j,i] = uI[inde]
               vl[j,i] = vI[inde]
   return ul,vl


def linear_perturbations3(x,y,N,Rdisk,rp,csp, phip, hr,indad, delta,nu,cw,xc,chi,xIc,chiI): #non gli passo le funz 2d ma già solo per t=0....
   ul = np.zeros((N,N))
   vl = np.zeros((N,N))

   ro = rp*(1.+4.*hr/3.)
   ri = rp*(1.-4.*hr/3.)
   Lfuo = Lfur(ro,rp,csp,hr,delta,nu,indad)
   Lfvo = Lfvr(ro,rp,csp,hr,delta,nu,indad)
   Lfui = Lfur(ri,rp,csp,hr,delta,nu,indad)
   Lfvi = Lfvr(ri,rp,csp,hr,delta,nu,indad)

   for i in range(N):    #cycle on columns
      for j in range(N): #cycle on rows
         r,phi = polar(x[i],y[j])
         # Perturbations are applied outside the linear regime square around the planet
         if r<Rdisk and (r<=ro and r>=ri):
            #compute (t1,eta1) corresponding to coordinates (r,phi) of the grid point considered
            eta1 = eta(r,phi,hr, nu, rp, phip, cw) #dep. on cw because phi_wake dep. on cw
            if r>rp:
               inde = np.argmin(np.abs(eta1-xc))
               Chi = chi[inde]

               ul[j,i] = Lfuo*Chi
               vl[j,i] = Lfvo*Chi
            else:
               inde = np.argmin(np.abs(eta1-xIc))
               Chi = chiI[inde]

               ul[j,i] = -Lfui*Chi
               vl[j,i] = -Lfvi*Chi
   return ul,vl






################################################

# Compute nonlinear perturbations on the grid

def nonlinear_perturbations(x,y,N,Rdisk,rin,rout,rp,phip, hr,nu,delta,cw,C,eta_tilde,t0,indad,csp,time,xc,u,xIc,uI): #qui u e uI sono la chi!!

   time = np.reshape(time,(len(time[0,:])))
   tf = time[-1]
   dnl = np.zeros((N,N)) #relative to Sigma0
   unl = np.zeros((N,N))
   vnl = np.zeros((N,N))
   for i in range(N):    #cycle on columns
      for j in range(N): #cycle on rows
         r,phi = polar(x[i],y[j])
         # Perturbations are applied outside the linear regime square around the planet
         if r<Rdisk and (r>rp*(1.+4.*hr/3.) or r<rp*(1.-4.*hr/3.)):
            #compute (t1,eta1) corresponding to coordinates (r,phi) of the grid point considered
            t1 = t(r,rp,hr,nu,delta)
            eta1 = eta(r,phi,hr, nu, rp, phip, cw) #dep. on cw because phi_wake dep. on cw
            '''
            if np.abs(tf-t1)<0.5:
               plt.scatter(x[i],y[j],zorder=+2)
               print(x[i])
               print(y[j])
            '''
            if t1<(tf+1.89):
               indt = np.argmin(np.abs(time-t1+1.89)) #second index of u and uI
               if r>rp:
                  inde = np.argmin(np.abs(eta1-xc))
                  chi = u[inde,indt]
               else:
                  inde = np.argmin(np.abs(eta1-xIc))
                  chi = uI[inde,indt]

               g1 = g(r,hr,rp,delta,nu)
               dnl[j,i] = chi*2/(g1*(indad+1))
               #
               #Coefficients Lambda_fu e Lambda_fv:
               Lfu = Lfur(r,rp,csp,hr,delta,nu,indad)
               Lfv = Lfvr(r,rp,csp,hr,delta,nu,indad)
               unl[j,i] = Lfu*chi*np.sign(r-rp) #eq. (38) nonlinear.pdf
               vnl[j,i] = Lfv*chi*np.sign(r-rp) #eq. (39) nonlinear.pdf

            else: #I apply the analytical asymptotic formula
               #condition for non vanishing \chi (equations (29),(30) nonlinear.pdf)
               extr_sx = -np.sign(r-rp)*eta_tilde - np.sqrt(2*C*(t1-t0))
               extr_dx = -np.sign(r-rp)*eta_tilde + np.sqrt(2*C*(t1-t0))
               if eta1 > extr_sx and eta1 < extr_dx:
                  #compute \chi and density correction dnl
                  chi = (np.sign(r-rp)*eta1 + eta_tilde)/(t1-t0)  #eq.(29) nonlinear.pdf
                  g1 = g(r,hr,rp,delta,nu)
                  dnl[j,i] = chi*2/(g1*(indad+1))
                  #
                  #Coefficients Lambda_fu e Lambda_fv:
                  Lfu = Lfur(r,rp,csp,hr,delta,nu,indad)
                  Lfv = Lfvr(r,rp,csp,hr,delta,nu,indad)
                  unl[j,i] = Lfu*chi*np.sign(r-rp) #eq. (38) nonlinear.pdf
                  vnl[j,i] = Lfv*chi*np.sign(r-rp) #eq. (39) nonlinear.pdf
   return dnl,unl,vnl


'''
def nonlinear_perturbations(x,y,N,Rdisk,rin,rout,rp,phip, hr,nu,delta,cw,C,eta_tilde,t0,indad,csp,tf,xc,u,xIc,uI):
   dnl = np.zeros((N,N)) #relative to Sigma0
   unl = np.zeros((N,N))
   vnl = np.zeros((N,N))
   for i in range(N):    #cycle on columns
      for j in range(N): #cycle on rows
         r,phi = polar(x[i],y[j])
         # Perturbations are applied outside the linear regime square around the planet
         if r<Rdisk and (r>=rp*(1.+4.*hr/3.) or r<=rp*(1.-4.*hr/3.)):
            #compute (t1,eta1) corresponding to coordinates (r,phi) of the grid point considered
            t1 = t(r,rp,hr,nu,delta)
            eta1 = eta(r,phi,hr, nu, rp, phip, cw) #dep. on cw because phi_wake dep. on cw
            if t1<tf+1.89: #I apply the numerical solution using chin and chinI ###########!##############!#############!############!
               if eta1<-20 or eta1>20: #sono fuori dal range la pert è zero
                  unl[j,i]=0
                  vnl[j,i]=0
                  dnl[j,i]=0
               else:
                  indt = np.argmin(np.abs(time-t1+1.89)) #row index of chi
                  #if indt<10:
                  #   print(indt)
                  if r>rp:
                     inde = np.argmin(np.abs(eta1-xc))
                     chi = u[indt,inde]
                  else:
                     inde = np.argmin(np.abs(eta1-xIc))
                     chi = uI[indt,inde]
                  g1 = g(r,hr,rp,delta,nu)
                  dnl[j,i] = chi*2/(g1*(indad+1))
                  #
                  #Coefficients Lambda_fu e Lambda_fv:
                  Lfu = Lfur(r,rp,csp,hr,delta,nu,indad)
                  Lfv = Lfvr(r,rp,csp,hr,delta,nu,indad)
                  unl[j,i] = Lfu*chi*np.sign(r-rp) #eq. (38) nonlinear.pdf
                  vnl[j,i] = Lfv*chi*np.sign(r-rp) #eq. (39) nonlinear.pdf

            else: #I apply the analytical asymptotic formula
               #condition for non vanishing \chi (equations (29),(30) nonlinear.pdf)
               extr_sx = -np.sign(r-rp)*eta_tilde - np.sqrt(2*C*(t1-t0))
               extr_dx = -np.sign(r-rp)*eta_tilde + np.sqrt(2*C*(t1-t0))
               if eta1 > extr_sx and eta1 < extr_dx:
                  #compute \chi and density correction dnl
                  chi = (np.sign(r-rp)*eta1 + eta_tilde)/(t1-t0)  #eq.(29) nonlinear.pdf
                  g1 = g(r,hr,rp,delta,nu)
                  dnl[j,i] = chi*2/(g1*(indad+1))
                  #
                  #Coefficients Lambda_fu e Lambda_fv:
                  Lfu = Lfur(r,rp,csp,hr,delta,nu,indad)
                  Lfv = Lfvr(r,rp,csp,hr,delta,nu,indad)
                  unl[j,i] = Lfu*chi*np.sign(r-rp) #eq. (38) nonlinear.pdf
                  vnl[j,i] = Lfv*chi*np.sign(r-rp) #eq. (39) nonlinear.pdf
   return dnl,unl,vnl

'''

def apply_lin_perturbations(N,x,y,ul,vl,vx0,vy0,vk0,Rdisk,viscb,alfa,m,hr,rp,nu):
   vk = vk0.copy()
   vx = vx0.copy()
   vy = vy0.copy()
   Np = 0
   for i in range(N):     #cycle on columns
      for j in range(N):  #cycle on rows
         r,phi = polar(x[i],y[j])
         if r<Rdisk and (r<=rp*(1.+4.*hr/3.) and r>=rp*(1.-4.*hr/3.)):
             #if ul[j,i]!=0 and vl[j,i]!=0: #[j,i] is a grid point with nonvanishing perturbation
                  if viscb == True: # A viscous damping is applied
                     damp = viscous_damping(alfa,m,hr,r,rp,nu)
                     #print(damp)
                     V = vl[j,i]*damp
                     U = ul[j,i]*damp
                  else:
                     V = vl[j,i]
                     U = ul[j,i]
                  vphi = vk0[j,i]+V
                  vr = U
                  vk[j,i] = np.sqrt(vphi**2+vr**2)
                  #dvofv[j,i] = (vk[j,i]-vk0[j,i])/vk0[j,i]
                  vx[j,i] = vr*np.cos(phi)-vphi*np.sin(phi)
                  vy[j,i] = vr*np.sin(phi)+vphi*np.cos(phi)
   return vx,vy,vk






def apply_velocity_perturbations(N,x,y,unl,vnl,vx0,vy0,vk0,Rdisk,viscb,alfa,m,hr,rp,nu): # N = grid linear dimension
   vk = vk0.copy()
   vx = vx0.copy()
   vy = vy0.copy()
   #dvofv = np.zeros(vk0.shape) #percentage of variation of total velocity module
   Np = 0
   for i in range(N):     #cycle on columns
      for j in range(N):  #cycle on rows
         r,phi = polar(x[i],y[j])
         if r<Rdisk:
            if x[i]<=rp*(1.+4.*hr/3.) and x[i]>=rp*(1.-4.*hr/3.) and y[j]<=rp*4*hr/3 and  y[j]>=-rp*4*hr/3:
               continue
            else:
               if unl[j,i]!=0 and vnl[j,i]!=0: #[j,i] is a grid point with nonvanishing perturbation
                  if viscb == True: # A viscous damping is applied
                     damp = viscous_damping(alfa,m,hr,r,rp,nu)
                     #print(damp)
                     V = vnl[j,i]*damp
                     U = unl[j,i]*damp
                  else:
                     V = vnl[j,i]
                     U = unl[j,i]
                  vphi = vk0[j,i]+V
                  vr = U
                  vk[j,i] = np.sqrt(vphi**2+vr**2)
                  #dvofv[j,i] = (vk[j,i]-vk0[j,i])/vk0[j,i]
                  vx[j,i] = vr*np.cos(phi)-vphi*np.sin(phi)
                  vy[j,i] = vr*np.sin(phi)+vphi*np.cos(phi)
   return vx,vy,vk


def dvofv(N,vk0,vk):
   dvofv = np.zeros(vk0.shape)
   for i in range(N):
      for j in range(N):
         dvofv[i,j] = (vk[i,j]-vk0[i,j])/vk0[i,j]
   return dvofv






# ROTATIONS

def Rx(a):
   R = [[1,0,0],[0,np.cos(a),-np.sin(a)],[0, np.sin(a), np.cos(a)]]
   return R
def Ry(b):
   R = [[np.cos(b),0,np.sin(b)],[0,1,0],[-np.sin(b),0,np.cos(b)]]
   return R
def Rz(g):
   R = [[np.cos(g),-np.sin(g),0],[np.sin(g),np.cos(g),0],[0,0,1]]
   return R

def R_arbitrary(i,g):
    R = [[np.cos(i)+np.cos(g)**2*(1-np.cos(i)), np.cos(g)*np.sin(g)*(1-np.cos(i)),np.sin(g)*np.sin(i)],
         [np.cos(g)*np.sin(g)*(1-np.cos(i)), np.cos(i)+np.sin(g)**2*(1-np.cos(i)),-np.cos(g)*np.sin(i)],
         [-np.sin(g)*np.sin(i), np.sin(i)*np.cos(g), np.cos(i)]]
    return R


def rotate_planet_arbitrary(gammap,i,gamma,planet):
   planet = np.dot(Rz(gammap),planet)
   planet = np.dot(Rz(gamma), np.dot(Ry(i),planet))
   return planet



def apply_arbitrary_rotation(N,gammap,teta,gamma,X,Y,x,y,Rdisk,vx,vy):
   #
   Xnew = np.zeros(X.shape)
   Ynew = np.zeros(Y.shape)
   Znew = np.zeros(Y.shape)
   #
   vf = np.zeros((3,N,N)) #velocity field (v,i,j) with v=vx,vy,vz
   #
   #fill vf
   for i in range(N):
      for j in range(N):
         vf[0,i,j] = vx[i,j]
         vf[1,i,j] = vy[i,j]
   #apply rotations
   for i in range(N):
      for j in range(N):
         r,phi = polar(x[j],y[i])
         #R planet prima di inclinazione disco
         Xnew[i,j],Ynew[i,j],Znew[i,j] = np.dot(Rz(gammap),np.array([X[i,j],Y[i,j],0]))
         if r<= Rdisk:
            vf[:,i,j] = np.dot(Rz(gammap),vf[:,i,j])
         #
         #Rdisco
         Xnew[i,j],Ynew[i,j],Znew[i,j] = np.dot( Rz(gamma),np.dot(Ry(teta),np.array([Xnew[i,j],Ynew[i,j],Znew[i,j]])))
         if r<= Rdisk:
            vf[:,i,j] = np.dot(Rz(gamma),np.dot(Ry(teta),vf[:,i,j]))
            #
            #high max limit to vz to avoid divergence near disc center
            if vf[2,i,j]>30:
               vf[2,i,j] = 30
         else:
            vf[2,i,j] = 100
   #
   return Xnew,Ynew,vf


def apply_rotations_new(N,alpha,beta,gamma,X,Y,Z,x,y,Rdisk,vx,vy,vz):
   #
   Xnew = np.zeros(X.shape)
   Ynew = np.zeros(Y.shape)
   Znew = np.zeros(Y.shape)
   #
   vf = np.zeros((3,N,N)) #velocity field (v,i,j) with v=vx,vy,vz
   #
   #fill vf
   for i in range(N):
      for j in range(N):
         vf[0,i,j] = vx[i,j]
         vf[1,i,j] = vy[i,j]
         vf[2,i,j] = vz[i,j]
   #apply rotations
   for i in range(N):
      for j in range(N):
         r,phi = polar(x[j],y[i])
         #Rz
         Xnew[i,j],Ynew[i,j],Znew[i,j] = np.dot(Rz(gamma),np.array([X[i,j],Y[i,j],Z[i,j]]))
         if r<= Rdisk:
            vf[:,i,j] = np.dot(Rz(gamma),vf[:,i,j])
         #
         #Rx, Ry
         Xnew[i,j],Ynew[i,j],Znew[i,j] = np.dot( Rx(alpha),np.dot(Ry(beta),np.array([Xnew[i,j],Ynew[i,j],Znew[i,j]])))
         if r<= Rdisk:
            vf[:,i,j] = np.dot(Rx(alpha),np.dot(Ry(beta),vf[:,i,j]))
            #
            #high max limit to vz to avoid divergence near disc center
            if vf[2,i,j]>30:
               vf[2,i,j] = 30
         else:
            vf[2,i,j] = 100
   #
   return Xnew,Ynew,Znew,vf







def apply_rotations(N,alpha,beta,gamma,X,Y,x,y,Rdisk,vx,vy):
   #
   Xnew = np.zeros(X.shape)
   Ynew = np.zeros(Y.shape)
   Znew = np.zeros(Y.shape)
   #
   vf = np.zeros((3,N,N)) #velocity field (v,i,j) with v=vx,vy,vz
   #
   #fill vf
   for i in range(N):
      for j in range(N):
         vf[0,i,j] = vx[i,j]
         vf[1,i,j] = vy[i,j]
   #apply rotations
   for i in range(N):
      for j in range(N):
         r,phi = polar(x[j],y[i])
         #Rz
         Xnew[i,j],Ynew[i,j],Znew[i,j] = np.dot(Rz(gamma),np.array([X[i,j],Y[i,j],0]))
         if r<= Rdisk:
            vf[:,i,j] = np.dot(Rz(gamma),vf[:,i,j])
         #
         #Rx, Ry
         Xnew[i,j],Ynew[i,j],Znew[i,j] = np.dot( Rx(alpha),np.dot(Ry(beta),np.array([Xnew[i,j],Ynew[i,j],Znew[i,j]])))
         if r<= Rdisk:
            vf[:,i,j] = np.dot(Rx(alpha),np.dot(Ry(beta),vf[:,i,j]))
            #
            #high max limit to vz to avoid divergence near disc center
            if vf[2,i,j]>30:
               vf[2,i,j] = 30
         else:
            vf[2,i,j] = 100
   #
   return Xnew,Ynew,vf


def rotate_planet(alpha,beta,gamma,planet):
   planet = np.dot(Rz(gamma),planet)
   planet = np.dot(Rx(alpha),np.dot(Ry(beta),planet))
   return planet



# DENSITY NONLINEAR PERTURBATIONS

def create_density(Sigmap,rp,delta,N,x,y,Rdisk):
   Sigma0 = np.zeros((N,N))
   for i in range(N):   #cycle on columns
      for j in range(N):#cycle on rows
         r, phi = polar(x[i],y[j])
         if r<Rdisk and r>0:
            Sigma0[j,i] = min(Sigmap*(r/rp)**(-delta),1100)
         else:
            Sigma0[j,i] = 1100
   return Sigma0

def apply_density_perturbation(Sigma0, dnl, N, x, y,Rdisk):
   Sigma = np.zeros((N,N))
   for i in range(N):   #cycle on columns
      for j in range(N):#cycle on rows
         r, phi = polar(x[i],y[j])
         if r<Rdisk:
            Sigma[j,i] = Sigma0[j,i]*(1+dnl[j,i])
         else:
            Sigma[j,i] = 1100
   return Sigma



# CHANNEL MAPS

def channels(vchs, v_field): #vchs contains vch modules
   dch = 0.05  # channel maps semi-width
   Nvch = len(vchs)
   vl = 50*np.ones(v_field[2,:,:].shape) #vz field
   for i in range(len(vl[:,0])):     #cycle on rows
      for j in range(len(vl[0,:])):  #cycle on coloumns
         vf = v_field[2,i,j]
         for iv in range(Nvch):
            if (vf > vchs[iv]-dch and vf < vchs[iv] + dch): # or (vf > -vchs[iv]-dch and vf < -vchs[iv] + dch):
               vl[i,j] = vf
   return vl


'''
def dzofv(N,vf,vf0):
   dzofv = np.zeros(vf[2,:,:].shape)
   for i in range(N):
      for j in range(N):
         dzofv[i,j] = (vf[2,i,j]-vf0[2,i,j])/vf0[2,i,j]
   return dzofv
'''





# VELOCITY PROFILES ALONG THE WAKE


def dvz(N,vf,vf0):
   dz = np.zeros(vf[2,:,:].shape)
   for i in range(N):
      for j in range(N):
         dz[i,j] = vf[2,i,j]-vf0[2,i,j]
   return dz


def maxmin_v_profiles(rp,phip,phish,Nphi,X,Y,N,pert): #works for counterclockwise velocity field
   #
   # Create new grid in polar coordinates
   R = np.zeros(X.shape)
   PHI = np.zeros(X.shape)
   for i in range(N):
      for j in range(N):
         R[i,j],PHI[i,j] = polar(X[i,j],Y[i,j])
         #
         # Now I shift the zero of phis of + phip
         PHI[i,j] -= phip
         if PHI[i,j] <0:
            PHI[i,j]+=2*np.pi
         PHI[i,j] = 2*np.pi - PHI[i,j] #phi clockwise, following the outer wake
   #
   # Define dphi
   offset = 0
   dphi = ((3/2)*np.pi-offset)/Nphi
   phis = np.linspace(offset,(3/2)*np.pi,Nphi)
   #
   # Define vmax and vmin profiles along the wake
   vM = np.zeros(Nphi)
   vm = np.zeros(Nphi)
   #
   for k in range(Nphi):
         vmax=-1
         vmin=+1
         imax=0
         jmax=0
         imin=0
         jmin=0
         #define azimuth segment extrema
         phiinf = offset+dphi*k
         phisup = offset+dphi*(k+1)
         for i in range(N):    #check every point of the grid
            for j in range(N):
                  if PHI[i,j]>phiinf and PHI[i,j]<phisup and R[i,j]>rp:
                     #I'm in the clove in the outer wake
                     if pert[i,j]>vmax:
                        vmax = pert[i,j]
                        imax = i
                        jmax = j
                     if pert[i,j]<vmin:
                        vmin = pert[i,j]
                        imin = i
                        jmin = j
         vM[k] = vmax
         vm[k] = vmin
         if k%3 ==0:
            plt.scatter(X[imax,jmax],Y[imax,jmax],zorder = +1, color = 'r', marker = '^')
            plt.scatter(X[imin,jmin],Y[imin,jmin],zorder = +1, color = 'r', marker = '^')
   return phis, vM,vm


##################################################################################################################################

# QUELLO NUOVO
def sqrt_rb(phi,teta,vch,M):
   Msun = 1.989*10**33
   G = 6.67*10**(-8)
   alfa = np.sqrt(G*M*Msun)
   if phi != np.pi/2 and phi != 3*np.pi/2:
      if phi > np.pi/2 and phi < 3*np.pi/2 :
         eps = -1
      else:
         eps = +1
      numeratore = alfa*np.sin(teta)*np.sin(np.arctan(np.tan(phi)*np.cos(teta)))*eps
      denominatore = vch*( (np.cos(phi)*np.tan(teta))**2 + 1 )**(1/4)
      rb2 = numeratore/denominatore
      if rb2>0:
         return rb2
      else:
         return 0
   else:
      rb2 = alfa*np.sin(teta)/vch
      if rb2>0:
         return rb2
      else:
         return 0

# QUELLO NUOVO
def butterfly_pattern(vch,M,teta,Nb,gamma):
   au = 1.49597870*10**13 #cm
   phis = np.linspace(0,2*np.pi,Nb)
   rb2 = np.zeros(phis.shape) #sqrt
   for i in range(Nb):
      rb2[i] = sqrt_rb(phis[i],teta,vch,M)
   x = np.zeros(Nb)
   y = np.zeros(Nb)
   for i in range(Nb):
      x[i],y[i] = cartesian(rb2[i]**2,phis[i])
   x = x/au
   y = y/au
   for i in range(len(phis)):
      x[i],y[i],z = np.dot(Rz(gamma),np.array([x[i],y[i],0]))
   i=0
   while i < Nb:
      if x[i] == 0 and y[i] == 0:
         x = np.delete(x, i)
         y = np.delete(y, i)
         Nb = Nb-1
      else:
         i=i+1
   N = len(x)
   rb = np.zeros(N)
   phib = np.zeros(N)
   for i in range(N):
      rb[i],phib[i] = polar(x[i],y[i])
   return x,y,rb,phib



def def_seg(xcut,ycut,Nc):
   seg0 = np.zeros((Nc,2))
   for i in range(Nc):
      seg0[i,0] = xcut[i]
      seg0[i,1] = ycut[i]
   return seg0



def cut_iso(xiso, yiso, xe,ye):
   Ne = int(len(xe))
   er = np.zeros(Ne)
   ephi = np.zeros(Ne)
   for i in range(Ne):
      er[i],ephi[i] = polar(xe[i],ye[i])
   Niso = int(len(xiso))
   iso0r = np.zeros(Niso)
   iso0phi = np.zeros(Niso)
   for j in range(Niso):
      iso0r[j],iso0phi[j] = polar(xiso[j],yiso[j])
   i = 0
   end = Niso-1 #il valore dell'ultimo indice
   while i <= end: #cycle on iso0
      indd = np.argmin( np.abs(iso0phi[i]-ephi) ) #index dell'orbit edge che corr a quel phiiso
      re = er[indd] #raggio edge
      if (re-10) <= iso0r[i]: #considero il raggio dell'orbita ridotto di un'au, taglio un po' prima
         end = end-1
         iso0r = np.delete(iso0r,i)
         iso0phi = np.delete(iso0phi,i)
      else:
         i+=1
   Ncut = end+1
   xcut = np.zeros(Ncut)
   ycut = np.zeros(Ncut)
   for i in range(Ncut):
      xcut[i],ycut[i] = cartesian(iso0r[i],iso0phi[i])
   return xcut, ycut, Ncut

def rb(phi, a,b,vc, mu):
   au = 1.49597870*10**13
   coeff = ((mu/vc)**2)/au #il coeff è una lunghezza. Viene calcolato da grandezze con cm e s. Converto in au.
   zeta = np.cos(phi)**2*(np.cos(b)**(-2) + (np.cos(a)**(-2))*(np.tan(phi)-np.sin(a)*np.tan(b))**2)
   kappa = np.cos(phi)*np.sin(a)*np.cos(b) + np.sin(phi)*np.sin(b)
   return coeff*(zeta**(-3/2))*kappa**2


def rotate_disc_edge(Rdisk,bo,rp,l,n,a,b):
   #
   Ne = 500
   xe = np.zeros(Ne)
   ye = np.zeros(Ne)
   phie = np.linspace(0,2*np.pi-0.001,Ne)
   if bo:
      re = Rdisk
   else:
      re = (rp+n*l)
   for i in range(Ne):
      xe[i],ye[i] = cartesian(re,phie[i])
      xe[i],ye[i],tmp = np.dot(Rx(a),np.dot(Ry(b),np.array([xe[i],ye[i],0])))
   return xe,ye

def compute_dist(seg,seg0,Rdisk):
   dist = np.zeros((len(seg[:,0]),2)) #primo indice condiene la distanza minima dalla iso0, il secondo l'indice del punto di iso0 alla
   indx = 0
   for i in range(len(seg[:,0])): #ciclo sulla isov1 con pochi punti
      d = Rdisk
      for j in range(len(seg0[:,0])): #ciclo sulla isov0 analitica
         e = euclid(seg[i,0],seg[i,1],seg0[j,0],seg0[j,1])
         if e < d:
            d = e
            indx = j
      #finito il ciclo in dist metto la distanza più piccola che ho trovato tra le due curve.
      dist[i,0] = d
      dist[i,1] = indx
      #plt.plot([seg0[indx,0],seg[i,0]],[seg0[indx,1],seg[i,1]],color = 'g')
   indu = np.argmax(dist[:,0]) #indice su seg del punto in cui si ha distanza massima tra le iso
   indu0 = int(dist[indu,1])   #indice su seg0 del punto in cui si ha distanza massima tra le iso
   metric = max(dist[:,0])
   return metric, indu, indu0


def contourplot(Xnew,Ynew,vf,levels):
   PLOT = plt.contour(Xnew, Ynew, vf[2,:,:], levels = levels, colors='black', linewidths = 0.5, linestyles = '-')
   plt.contourf(Xnew, Ynew, vf[2,:,:], levels = levels, cmap='seismic')
   plt.colorbar().set_label('$v \quad (km/s)$')
   plt.xlabel('$ x \quad (au)$')
   plt.ylabel('$ y \quad (au)$')
   plt.title('Isovelocity curves')
   plt.legend()
   return PLOT

def euclid(x1,y1,x2,y2):
   return np.sqrt((x1-x2)**2+(y1-y2)**2)




########################## butterfly new

def rmezzi(M,phi,vch,theta): #M [Msun], vch [Km/s]
  au = 1.49597870*10**13 #cm
  G = 6.67*10**(-8)      #cgs
  Msun = Msun = 1.989*10**33    #g
  den = (1+(np.tan(phi)*np.cos(theta))**2)**(3/4)
  num = np.sin(theta)*np.cos(theta)*np.tan(phi)
  coeff = np.sign(np.cos(phi))*np.sqrt(G*M*Msun*np.abs(np.cos(theta)/np.cos(phi)))/vch
  if np.sin(phi)*vch*np.sin(theta) >0:
     return ((coeff*num/den)*10**(-5)/np.sqrt(au))**2
  else:
     return 0


def fuu(d,n,r):
   return r**((d+n-1)/2)*np.abs(r**(-3/2)-1)**(1/2)

def fvv(d,n,r):
   return r**((d-n-3)/2)*np.abs(r**(-3/2)-1)**(-1/2)


def uvlin(betap,csp):
   x, u = np.loadtxt("u_ky11.0x2.dat", usecols=(0,1), delimiter=' ', unpack='true')
   x, v = np.loadtxt("v_ky11.0x2.dat", usecols=(0,1), delimiter=' ', unpack='true')
   u = u*betap*csp
   v = v*betap*csp

   plt.plot(x,u)
   plt.plot(x,v)
   plt.show()

   uI = np.zeros(u.shape)
   vI = np.zeros(v.shape)
   for i in range(len(x)):
         uI[i] = -u[int(len(x)-1-i)]
         vI[i] = -v[int(len(x)-1-i)]
   xI = - x[::-1]
   return x, u, v, xI, uI, vI



def burgers(betap,indad,Mp):

   xc, f = np.loadtxt("ky12.0x2.dat", usecols=(0,1), delimiter=' ', unpack='true') #upload dimensionless profile
   f = f*(indad+1)*betap/2**(3/4)                                                 #rescale to obtain Chi_r+


   dx = xc[1]-xc[0]
   #
   t0 = 0.0   # initial time
   #tf = 150.0   # final time

   # NUOVA IMPLEMENTAZIONE
   #tfmth = 200 #questo tempo per mth è suff per vedere la formaz di N wave
   tf = 150 #  IN OGNI CASO  ########## tfmth/betap # tempo ridotto perchè lo shock avviene prima (maggiore nonlinearità)
   Nt = 10000 # fisso!!!! (ma cambia dt a seconda del tempo finale!!)
   dt = tf/Nt
   #print(dt)
   # le eta +/- cui mi serve arrivare sono gli stessi ovvero sqrt(2C_th * betap * tf) = sqrt(2 Cth 150)
   Cth = 0.4*(indad+1)/2**(3/4) # dove 0.4 è C0 (vedi la mia tesi)
   C = Cth*betap
   eta_tilde = 3.497

   xmin = -eta_tilde-np.sqrt(2*C*150) - 3
   xmax = -eta_tilde+np.sqrt(2*C*150) + 3
   ext = xc[-1]
   #estendo il range preso da xc all'intervallo che mi interessa
   while ext<xmax:
      xc = np.append(xc,xc[-1]+dx)
      ext = xc[-1]
      f = np.append(f,0)
   ext = xc[0]
   while ext>xmin:
      xc = np.insert(xc,0,xc[0]-dx)
      ext = xc[0]
      f = np.insert(f,0,0)


   '''
   Ctf = 150.0 #C*tf (perché vogliamo che eta+/- = sqrt(2C*tf) sia fissato per tutti)
   #
   C=0.4*(indad+1)*betap/2**(3/4) # the area under the lobe in the case of generic Mp
   tf = Ctf/C  #si intende t-t0
                    #print("Ecco: "+str(tf))


   dt = 0.015
   Nt = int(tf/dt)
   print('Nt = ',Nt)


   eta_tilde = 3.497
   xmin = -eta_tilde-np.sqrt(2*C*tf) - 3
   xmax = -eta_tilde+np.sqrt(2*C*tf) + 3
   ext = xc[-1]
   #estendo il range preso da xc all'intervallo che mi interessa
   while ext<xmax:
      xc = np.append(xc,xc[-1]+dx)
      ext = xc[-1]
      f = np.append(f,0)
   ext = xc[0]
   while ext>xmin:
      xc = np.insert(xc,0,xc[0]-dx)
      ext = xc[0]
      f = np.insert(f,0,0)

   #plt.plot(xc,f)
   #plt.show()
   '''

   #
   Nx = len(xc) #Nx da' il numero di centri
   #print('Nx =',Nx)
   a = xc[0] - dx/2
   b = xc[-1] + dx/2
   #
   # derived constants
   L  = b-a   # spatial grid length
   T  = tf-t0 # final time
   #
   # set time array
   #dt = T / (Nt - 1)
   time = np.array([range(0,Nt)]) * dt
   #
   # set x arrays
   #dx = L / Nx
   x = np.zeros(Nx+1) #Nx centers, as the number of cells
   for i in range(0,Nx+1):
       x[i] = a+i*dx

   # initialize the state !!!!! Metto linear solution
   u = np.zeros((Nx,Nt), dtype=float)
   #
   for i in range(0,Nx):
      u[i,0] = f[i]
   #
   # define flux vector
   F = np.zeros(Nx+1) #there is a flux at each cell edge

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
   print('Starting integration ...', end=' ')
   for n in range(0,Nt-1):

      # estimate the CFL
      CFL = max(abs(u[:,n])) * dt / dx
      if CFL > 0.5:
         print("Warning: CFL > 0.5")

      # compute the interior fluxes
      for i in range(1,Nx):
          uL = u[i-1,n]
          uR = u[i,n]
          F[i] = NumericalFlux(uL,uR)

      # compute the left boundary flux
      if u[0,n] < 0.0:
          uL = 2.0*u[0,n] - u[1,n]
      else:
          uL = u[0,0]
      uR = u[0,n]
      F[0] = NumericalFlux(uL,uR)

      # compute the right boundary flux
      if u[Nx-1,n] > 0.0:
          uR = 2.0 * u[Nx-1,n] - u[Nx-2,n]
      else:
          uR = u[Nx-1,0]
      uL = u[Nx-1,n]
      F[Nx] = NumericalFlux(uL,uR)

      # update the state
      for i in range(0,Nx):
          u[i,n+1] = u[i,n] - dt / dx * (F[i+1] - F[i])

   '''
   plt.plot(xc, u[:,0], label = "$t=t_0+$ "+str(0*dt))
   plt.plot(xc, u[:,int(Nt/20)], label = "$t=t_0+$ "+str(round(dt*Nt/10,2)))
   plt.plot(xc, u[:,int(Nt/10)], label = "$t=t_0+$ "+str(round(dt*Nt/6,2)))
   plt.plot(xc, u[:,int(Nt/5)], label = "$t=t_0+$ "+str(round(dt*Nt/3,2)))
   plt.plot(xc, u[:,int(Nt-1)], label = "$t=t_0+$ "+str(round(T,2)))
   plt.legend()
   plt.xlabel("$\eta$")
   plt.ylabel("$\chi(t,\eta)$")
   plt.grid(True)
   plt.title('$\chi$ "evolution" $r > r_p$')
   plt.show()
   '''

   uI = np.zeros(u.shape)
   for i in range(len(xc)):
      for j in range(Nt):
         uI[i,j] = u[int(len(xc)-1-i),j]

   xIc = - xc[::-1]

   '''
   plt.plot(xIc, uI[:,0], label = "$t=t_0+$ "+str(0*dt))
   plt.plot(xIc, uI[:,int(Nt/20)], label = "$t=t_0+$ "+str(round(dt*Nt/10,2)))
   plt.plot(xIc, uI[:,int(Nt/10)], label = "$t=t_0+$ "+str(round(dt*Nt/6,2)))
   plt.plot(xIc, uI[:,int(Nt/5)], label = "$t=t_0+$ "+str(round(dt*Nt/3,2)))
   plt.plot(xIc, uI[:,int(Nt-1)], label = "$t=t_0+$ "+str(round(T,2)))
   plt.legend()
   plt.xlabel("$\eta$")
   plt.ylabel("$\chi(t,\eta)$")
   plt.grid(True)
   plt.title('$\chi$ "evolution" $r < r_p$')
   plt.show()
   '''

   return time, xc,u, xIc, uI



'''
def butterfly_pattern(a,b, vc, cw, M_star,Niso):
   iso0phi = np.linspace(0, 2*np.pi , Niso)
   iso0r_sqrt = rb_sqrt(iso0phi,a,b,vc , cw, M_star)
   #questo probabilmente potevo farlo con un .all() o .any() o cose simili
   iso0r = np.zeros(Niso)
   for i in range(Niso):
      iso0r[i] = (max(0,iso0r_sqrt[i]))**2
   x = np.zeros(Niso)
   y = np.zeros(Niso)
   for i in range(Niso):
      x[i],y[i] = cartesian(iso0r[i],iso0phi[i])
   return x,y
'''

'''
def rb_sqrt(phi, a,b,vc, cw, M_star): #la massa in unità di masse solari
   M_sun = 2*10**33 #g
   M_star = M_star*M_sun
   G = 6.67*10**(-8)
   mu = -cw*np.sqrt(M_star*G)
   au = 1.49597870*10**13
   coeff = (mu/vc)/(au)**0.5 #il coeff è una lunghezza. Viene calcolato da grandezze con cm e s. Converto in au.
   zeta = np.cos(phi)**2*(np.cos(b)**(-2) + (np.cos(a)**(-2))*(np.tan(phi)-np.sin(a)*np.tan(b))**2)
   kappa = np.cos(phi)*np.sin(a)*np.cos(b) + np.sin(phi)*np.sin(b)
   return coeff*(zeta**(-3/4))*kappa
'''
