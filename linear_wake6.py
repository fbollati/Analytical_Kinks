import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.integrate as ode

plt.rcParams.update({'font.size': 18})
plt.tight_layout()
plt.rcParams.update({'figure.autolayout': True})



au = 1.49597870*10**1 #cm
G = 6.67*10**(-8)      #cgs
Mj = 1.89819*10**30    #g
Msun = 1.989*10**33    #g

#ygrid
Ny = 2**10
Dy = np.pi/8.
Ly = Ny*Dy
fyc = 1./(2.*Dy)
kyc = 2.*np.pi*fyc

#xgrid
Lx = 2.*np.sqrt(Ly)
Nx = 2**9
Dx = Lx/Nx
fxc = 1./(2.*Dx)
kxc = 2.*np.pi*fxc

fx = np.array([-fxc + i/(Nx*Dx) for i in range(Nx)]) #at i = Nx/2 fx = 0.0
fy = np.array([-fyc + i/(Ny*Dy) for i in range(Ny)]) #at i = Ny/2 fy = 0.0

kx = 2.*np.pi*fx
ky = 2.*np.pi*fy

Fx,Fy =np.meshgrid(fx,fy)
Kx,Ky =np.meshgrid(kx,ky)


def func(a,b,kx,ky):
   return (np.pi**2)*np.exp(-kx**2/(4*a)-b*np.abs(ky))/a


tau_max = Lx/2.
print(tau_max)



# INTEGRATION FUNCTIONS

def rafikov(ky,t,y):
   dydt = np.zeros(len(y),dtype = 'complex')
   dydt[0] = -((t**2 + 1.)*(ky**2) + 4./9.)*y[1]-np.sign(ky)*(2.*1j*np.pi/3.)*(t*(t**2+4.))/((t**2+1.)**(3/2))
   dydt[1] = y[0]
   return dydt

def sigma(kx,ky,v,z):
   return 1j*(ky*z + kx*v/3. - 1j*2.*np.pi*(ky**2)/((kx**2+ky**2)**0.5))/(ky**2 + 1./9.)

def u_radial(kx,ky,v,z):
   k = (kx**2+ky**2)**0.5
   return -(z/3. - kx*ky*v - 2*1j*np.pi*ky/(3*k))/(ky**2+1./9.)



# FUNCTION FOR CUTOFF

def cutoff(v,t,kyj,tau_max,kyc): #passo il t,kyj e v correnti. spero che le variabili fuori dal CYCLE già le conosca.
      a = v
      if t >= tau_max/2. and t<tau_max:
         w = 2.*(1-t/tau_max)
         a *= w
      elif t >= tau_max:
         a = 0.
      kyy = np.abs(kyj)
      if kyy >= kyc/2. and kyy<kyc:
         w = 2.*(1-kyy/kyc)
         a *= w
      elif kyy>=kyc:
         a = 0.
      return a


# FUNCTION FOR FIND PROPER INTEGRATION INTERVAL

def newtau(tau_max, taus): #lo passo giò riverso
      dt = taus[1]-taus[0]
      a = int((tau_max-np.abs(taus[0]))/dt)
      print("a= "+str((tau_max-np.abs(taus[0]))/dt))
      Nnew = len(taus)+a
      t2 = np.zeros(Nnew)
      for u in range(Nnew):
         if u<a:
            t2[u] = taus[0]-(a-u)*dt
         else:
            t2[u] = taus[u-a]
      return a,t2

# ARRAYS

y0 = np.array([0.0+0.0j,0.1+0.0j])
v = np.zeros((Ny,Nx), dtype = 'complex')
v_NoCut = np.zeros((Ny,Nx), dtype = 'complex') #le funzioni senza cutoff servono per essere passate alla funzione sigma
z = np.zeros((Ny,Nx), dtype = 'complex')
z_NoCut = np.zeros((Ny,Nx), dtype = 'complex')
s = np.zeros((Ny,Nx), dtype = 'complex')
u = np.zeros((Ny,Nx), dtype = 'complex')
v_ext = np.zeros((int(Ny/2),Nx+1),dtype = 'complex') #array per contenere la parte sopra ky=0 e con kx esteso per intervallo sym
z_ext = np.zeros((int(Ny/2),Nx+1),dtype = 'complex')
v_ext_nocut = np.zeros((int(Ny/2),Nx+1),dtype = 'complex') #array per contenere la parte sopra ky=0 e con kx esteso per intervallo sym
z_ext_nocut = np.zeros((int(Ny/2),Nx+1),dtype = 'complex')

# CYCLE OF INTEGRATIONS

for j in range(0,int(Ny/2)): #j=0,..,Ny/2-1. in this range all ky<=0 and ky=0 at j=Ny/2 excluded

   kyj = ky[j]
   print("ky="+str(kyj))

   #define derivative function for Runge-Kutta method
   def f(t,y):
      return rafikov(kyj,t,y)

   #definisco l'intervallo di integrazione
   taus = kx/kyj                       #(tM, tM-dt, ..., +dt, 0, -dt, ..., -tM + dt)
   dt = taus[0]-taus[1]
   taus = np.append(taus, taus[-1]-dt) #(tM, tM-dt, ..., +dt, 0, -dt, ..., -tM + dt, -tM)
   if ky[j]<0:
     taus = taus[::-1]                 #(-tM, -tM+dt,..., -dt, 0, +dt,..., +tM - dt, tM)--> sym and properly ordered, ready for integratation


   soluz = np.zeros((2,Nx+1), dtype = 'complex' ) #the dimension is Nx+1 because we have extended the interval of taus




   '''
   #provo a integrare nel range dato sui taus della griglia e a vedere come viene la v(kx,ky)
   sol2 = ode.solve_ivp(f, [taus[0],taus[-1]],y0, t_eval = taus)
   if j%8 == 0:
      plt.subplot(121)
      plt.plot(sol2.t, sol2.y[1,:].real, label='real')
      plt.plot(sol2.t, sol2.y[1,:].imag, label='imag')
      plt.legend()
      plt.title('complex ode v')
      #
      plt.subplot(122)
      plt.plot(sol2.t, sol2.y[0,:].real, label='real')
      plt.plot(sol2.t, sol2.y[0,:].imag, label='imag')
      plt.legend()
      plt.title('complex ode z')
      plt.show()

   for i in range(Nx+1):
      soluz[0,i] = sol2.y[0,i]
      soluz[1,i] = sol2.y[1,i]

   '''
   if np.abs(taus[0])<=tau_max:
      a,t2 = newtau(tau_max,taus)
      print("dim t2= "+str(len(t2)))
      #print("taus2= "+str(t2))
      print("a= "+str(a))
      sol2 = ode.solve_ivp(f, [t2[0],t2[-1]],y0, t_eval = t2)
      '''
      if j%8 == 0:
         plt.subplot(121)
         plt.plot(sol2.t, sol2.y[1,:].real, label='real')
         plt.plot(sol2.t, sol2.y[1,:].imag, label='imag')
         plt.legend()
         plt.title('complex ode v')
         #
         plt.subplot(122)
         plt.plot(sol2.t, sol2.y[0,:].real, label='real')
         plt.plot(sol2.t, sol2.y[0,:].imag, label='imag')
         plt.legend()
         plt.title('complex ode z')
         plt.show()
      '''
      for i in range(Nx+1):
         soluz[0,i] = sol2.y[0,i+a]
         soluz[1,i] = sol2.y[1,i+a]

   else:
      g = 0
      while np.abs(taus[g])>tau_max:
         g+=1
      start = g
      end = Nx-2-start
      print("start= "+str(start)+"   end = "+str(end))
      #print(taus[start:end+1])
      print(str(taus[start])+" "+str(taus[end]))
      if start==end:
         continue
      sol = ode.solve_ivp(f, [taus[start],taus[end]], y0, t_eval = taus[start:end+1]) #I solve with taus in the correct order (-,+)
      '''
      if j%8 == 0:
         plt.subplot(121)
         plt.plot(sol.t, sol.y[1,:].real, label='real')
         plt.plot(sol.t, sol.y[1,:].imag, label='imag')
         plt.legend()
         plt.title('complex ode v SHORT RANGE ')
         #
         plt.subplot(122)
         plt.plot(sol2.t, sol2.y[0,:].real, label='real')
         plt.plot(sol2.t, sol2.y[0,:].imag, label='imag')
         plt.legend()
         plt.title('complex ode z SHORT RANGE ')
         plt.show()
      '''
      for l in range(start,end+1):
         soluz[0,l] = sol.y[0,l-start]
         soluz[1,l] = sol.y[1,l-start]

   #'''

   # fill vectors v and z
   for i in range(Nx+1):
      v_ext_nocut[j,i] =soluz[1,Nx-i] #v_ext è orientato già in modo giusto come v. tuttavia alla fine comp in più che corr a -tauM.
      z_ext_nocut[j,i] = soluz[0,Nx-i] #lo assegno così come viene trovato senza applicare cutoff

      #applico il cutoff e trovo v_ext e z_ext
      tauij = taus[Nx-i]
      v_ext[j,i] = cutoff(v_ext_nocut[j,i],tauij,kyj,tau_max,kyc)
      z_ext[j,i] = cutoff(z_ext_nocut[j,i],tauij,kyj,tau_max,kyc)

      if i==Nx:
         continue
      #parte superiore con cutoff
      v[j,i] = v_ext[j,i]
      z[j,i] = z_ext[j,i]
      #parte superiore senza cutoff
      v_NoCut[j,i] = v_ext_nocut[j,i]
      z_NoCut[j,i] = z_ext_nocut[j,i]




# END CYCLE
#riempio la seconda metà del vettore v
for j in range(int(Ny/2+1),Ny): #j=Ny/2 + 1, ..., Ny - 1 #rifletti, forse qua devo modificare qualcosa, la parte complessa è diversa...
   for i in range(Nx):
      #parte inferiore con cutoff
      v[j,i] = -v_ext[Ny-j,Nx-i]
      z[j,i] = -z_ext[Ny-j,Nx-i]
      #parte inferiore seza cutoff
      v_NoCut[j,i] = -v_ext_nocut[Ny-j,Nx-i]
      z_NoCut[j,i] = -z_ext_nocut[Ny-j,Nx-i]

# PRINT FOURIER TRANSOFR OF V AND Z
#plot v. sopra con cutoff, sotto senza
'''
fig = plt.figure()
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot_surface(Fx, Fy, v.real,  color='green',cmap='winter') #Prova a cambiare con la mesh kx,ky
ax.set_xlabel('fx')
ax.set_ylabel('fy')
plt.title('real')

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.plot_surface(Fx, Fy, v.imag,  color='green',cmap='winter') #Prova a cambiare con la mesh kx,ky
ax.set_xlabel('fx')
ax.set_ylabel('fy')
plt.title('imag')

ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.plot_surface(Fx, Fy, v_NoCut.real,  color='green',cmap='winter') #Prova a cambiare con la mesh kx,ky
ax.set_xlabel('fx')
ax.set_ylabel('fy')
plt.title('real')

ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.plot_surface(Fx, Fy, v_NoCut.imag,  color='green',cmap='winter') #Prova a cambiare con la mesh kx,ky
ax.set_xlabel('fx')
ax.set_ylabel('fy')
plt.title('imag')
plt.suptitle('v(fx,fy)')
plt.show()

newv = np.zeros((v.shape), dtype = "complex")
for i in range(Nx):
   for j in range(Ny):
      newv[j,i] = v[Ny-1-j,i]


fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(Fx, Fy, newv.real,  color='green',cmap='winter') #Prova a cambiare con la mesh kx,ky
ax.set_xlabel('fx')
ax.set_ylabel('fy')
plt.title('real new')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(Fx, Fy, newv.imag,  color='green',cmap='winter') #Prova a cambiare con la mesh kx,ky
ax.set_xlabel('fx')
ax.set_ylabel('fy')
plt.title('imag new')
plt.show()

'''




#### DIRECT SPACE

x = np.array([i*Dx - (Nx-1)*Dx/2. for i in range(Nx)])
y = np.array([i*Dy - (Ny-1)*Dy/2. for i in range(Ny)])

X,Y = np.meshgrid(x,y)

# I print the grid in X.dat, Y.dat
xfile = open("X.dat", "w")
np.savetxt(xfile, X, fmt='%.3f')
xfile.close()
yfile = open("Y.dat", "w")
np.savetxt(yfile, Y, fmt='%.3f')
yfile.close()

# V
v = np.fft.ifftshift(v)
vd = np.fft.ifft2(v/(Dx*Dy)) #questi due passaggi per ottenere la gg nello spazio x,y
vd = np.fft.fftshift(vd)

#save vd for maxv.py
vfile = open("vtot.dat", "w")
np.savetxt(vfile, vd.real, fmt='%.3f')
vfile.close()


'''
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X, Y, vd.real,  color='green',cmap='winter') #Prova a cambiare con la mesh kx,ky
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.title('real')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(X, Y, vd.imag,  color='green',cmap='winter') #Prova a cambiare con la mesh kx,ky
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.title('imag')
plt.suptitle('v(x,y)')
plt.show()
'''












def wake(x):
   return -np.sign(x)*(x**2)/2
yy = wake(x)

'''
plt.contourf(X, Y, vd.real, label='numerical simulation')
plt.colorbar();
plt.plot(x,yy, c='r', label = "$ y = -0.5 \cdot sign(x) \cdot x^2 $", linewidth = 0.5)
plt.legend()
plt.xlabel('$ x $')
plt.ylabel('$ y $')
plt.title('$ v(x,y) $')
plt.show()
'''

xmax = max(x)/2
xn = x[(x>-xmax)&(x<xmax)]
print(xn.shape)
xstart = np.argmin(np.abs(x-xn[0]))
xend = np.argmin(np.abs(x-xn[-1]))
print(xstart)
print(xend)

ymax = 0.5*xmax**2
yn = y[(y>-ymax)&(y<ymax)]
print(yn.shape)
ystart = np.argmin(np.abs(y-yn[0]))
yend = np.argmin(np.abs(y-yn[-1]))

vdn = vd.real[ystart:yend+1,xstart:xend+1]
print(vdn.shape)
XN,YN = np.meshgrid(xn,yn)
print(XN.shape)
yyn = wake(xn)

'''
plt.contourf(XN, YN, vdn)
plt.colorbar();
plt.arrow(-xmax/2.,-2*ymax/3.,0,ymax/3., length_includes_head = True, head_width=xmax/20)
plt.arrow(xmax/2.,2*ymax/3.,0,-ymax/3., length_includes_head = True, head_width=xmax/20)
plt.plot(xn,yyn, c='r', label = "$ y = -0.5 \cdot sign(x) \cdot x^2 $", linewidth = 0.5)
plt.legend()
plt.xlabel('$ x $')
plt.ylabel('$ y $')
plt.title('$ v(x,y) $')
plt.show()
'''


###cose da stampare nel file per il kink lineare in butterfly.py
M =1     #Msun
rp = 100 #au
vkp =  np.sqrt(G*M*Msun/rp)*10**(-5) #km/s
hr = 0.1
csp = vkp*hr
hp = hr*rp
l = (2/3)*hp
M1p = (l*au)*(csp*10**5)**2/(G*Mj) #planet mass in MJ

xlinmax = 8
xkink = x[(x>-xlinmax)&(x<xlinmax)]
xstartk = np.argmin(np.abs(x-xkink[0]))
xendk = np.argmin(np.abs(x-xkink[-1]))
ylinmax = 0.5*xlinmax**2
ykink = y[(y>-ylinmax)&(y<ylinmax)]
ystartk = np.argmin(np.abs(y-ykink[0]))
yendk = np.argmin(np.abs(y-ykink[-1]))
vdk = vd.real[ystartk:yendk+1,xstartk:xendk+1]

yw = wake(xkink)


Xk,Yk = np.meshgrid(xkink,ykink)

Xkd = Xk*l
Ykd = Yk*l
vdkd = vdk*csp


fig, ax = plt.subplots()
conto = ax.contourf(Xkd, Ykd, vdkd/vkp, levels = np.arange(-0.07,0.07,0.0005),cmap='RdBu')

cb = fig.colorbar(conto, ticks=[-0.05, 0, 0.05])
cb.ax.locator_params(nbins=3)

#plt.plot([-2*l,+2*l,+2*l,-2*l,-2*l],[-2*l,-2*l,+2*l,+2*l,-2*l],color = 'k', linewidth=2)
plt.arrow(-xlinmax*l/2.,-2*ylinmax*l/3.,0,ylinmax*l/3., length_includes_head = True, head_width=xlinmax*l/20)
plt.arrow(l*xlinmax/2.,2*ylinmax*l/3.,0,-ylinmax*l/3., length_includes_head = True, head_width=xlinmax*l/20)

#plt.tight_layout()
plt.rcParams.update({'figure.autolayout': True})

plt.locator_params(nbins=3)
plt.xlabel('$ x \quad [au]$')
plt.ylabel('$ y \quad [au]$')
#ax.axis('equal')
plt.title('$ v(x,y)/v_p $')
plt.show()




'''
xlinmax =2
xkink = x[(x>-xlinmax)&(x<xlinmax)]
xstartk = np.argmin(np.abs(x-xkink[0]))
xendk = np.argmin(np.abs(x-xkink[-1]))
ylinmax = 0.5*xlinmax**2
ykink = y[(y>-ylinmax)&(y<ylinmax)]
ystartk = np.argmin(np.abs(y-ykink[0]))
yendk = np.argmin(np.abs(y-ykink[-1]))
vdk = vd.real[ystartk:yendk+1,xstartk:xendk+1]

Xk,Yk = np.meshgrid(xkink,ykink)
plt.contourf(Xk, Yk, vdk)
plt.colorbar()
plt.xlabel('$ x $')
plt.ylabel('$ y $')
plt.title('$ v(x,y) linear Rafikov $')
plt.show()
'''

# ora stampo su file i valori di vx
'''
file = open("LinearV.dat", "w")
file.write(str(Dx)+", "+str(Dy)+", "+str(Dy/Dx)+", "+str(len(ykink))+"\n") #nella prima riga stampo i Dx e Dy e il numero di righe che seguono relative a v
file.write("v\n")
np.savetxt(file, vdk, fmt='%.3f')
file.close()
'''





#Z

z = np.fft.ifftshift(z)
zd = np.fft.ifft2(z/(Dx*Dy)) #questi due passaggi per ottenere la gg nello spazio x,y
zd = np.fft.fftshift(zd)

'''
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X, Y, zd.real,  color='green',cmap='winter') #Prova a cambiare con la mesh kx,ky
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.title('real')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(X, Y, zd.imag,  color='green',cmap='winter') #Prova a cambiare con la mesh kx,ky
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.title('imag')
plt.suptitle('z(x,y)')
plt.show()
'''

#ORA PROVO A RISTAMPARE LA VF E ZF NEL DOPO AVER USATO SHIFT. NB QUESTA È LA FORMA CHE POI PASSERÒ ALLA FUNZ SIGMA

#li risistemo come se dovessi passarli alla funz sigma. Ma mi sa che non è così che sono i nocut che devo passare.
v = np.fft.ifftshift(v)
z = np.fft.ifftshift(z)

# ORA CALCOLO SIGMA E U

#Now sigma
for j in range(Ny):
   kyj = ky[j]
   if ky[j]==0:
      continue
   taus = kx/kyj #a seconda di j andrà in una direzione o nell'altra
   for i in range(Nx):
      tauij = np.abs(taus[i])
      #
      u[j,i] = u_radial(kx[i],ky[j],v_NoCut[j,i],z_NoCut[j,i])
      u[j,i] = cutoff(u[j,i],tauij,kyj,tau_max,kyc)
      #
      s[j,i] = sigma(kx[i],ky[j],v_NoCut[j,i],z_NoCut[j,i])
      s[j,i] = cutoff(s[j,i],tauij,kyj,tau_max,kyc)


# GRFICI DI SIGMA #################################################################################################
'''
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(Fx, Fy, s.real,  color='green',cmap='winter') #Prova a cambiare con la mesh kx,ky
ax.set_xlabel('fx')
ax.set_ylabel('fy')
plt.title('real')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(Fx, Fy, s.imag,  color='green',cmap='winter') #Prova a cambiare con la mesh kx,ky
ax.set_xlabel('fx')
ax.set_ylabel('fy')
plt.title('imag')
plt.suptitle('s(fx,fy)')
plt.show()
'''


#### FUNZIONE NELLO SPAZIO DIRETTO



s = np.fft.ifftshift(s)
sd = np.fft.ifft2(s/(Dx*Dy)) #questi due passaggi per ottenere la gg nello spazio x,y
sd = np.fft.fftshift(sd)

u = np.fft.ifftshift(u)
ud = np.fft.ifft2(u/(Dx*Dy)) #questi due passaggi per ottenere la gg nello spazio x,y
ud = np.fft.fftshift(ud)


#save ud for maxv.py
ufile = open("utot.dat", "w")
np.savetxt(ufile, ud.real, fmt='%.3f')
ufile.close()




for i in range(Nx):
   for j in range(Ny):
      if sd[j,i]>2.5:
         sd[j,i] = 2.5


sfile = open("stot.dat", "w")
np.savetxt(sfile, sd.real, fmt='%.3f')
sfile.close()

'''
#sigma
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X, Y, sd.real,  color='green',cmap='winter') #Prova a cambiare con la mesh kx,ky
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.title('real')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(X, Y, sd.imag,  color='green',cmap='winter') #Prova a cambiare con la mesh kx,ky
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.title('imag')
plt.suptitle('s(x,y)')
plt.show()
'''
'''
#u
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X, Y, ud.real,  color='green',cmap='winter') #Prova a cambiare con la mesh kx,ky
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.title('real')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(X, Y, ud.imag,  color='green',cmap='winter') #Prova a cambiare con la mesh kx,ky
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.title('imag')
plt.suptitle('u(x,y)')
plt.show()
'''


#CONTOUR PLOT SIGMA E U

sdn = sd.real[ystart:yend+1,xstart:xend+1]
plt.contourf(XN, YN, sdn, label='numerical simulation')
plt.colorbar();
plt.arrow(-xmax/2.,-2*ymax/3.,0,ymax/3., length_includes_head = True, head_width=xmax/20)
plt.arrow(xmax/2.,2*ymax/3.,0,-ymax/3., length_includes_head = True, head_width=xmax/20)
plt.plot(xn,yyn, c='r', label = "$ y = -0.5 \cdot sign(x) \cdot x^2 $", linewidth = 0.5)
plt.legend()
plt.xlabel('$ x $')
plt.ylabel('$ y $')
plt.title('$ \sigma (x,y) $')
plt.show()



udn = ud.real[ystart:yend+1,xstart:xend+1]
plt.contourf(XN, YN, udn, label='numerical simulation')
plt.colorbar();
plt.arrow(-xmax/2.,-2*ymax/3.,0,ymax/3., length_includes_head = True, head_width=xmax/20)
plt.arrow(xmax/2.,2*ymax/3.,0,-ymax/3., length_includes_head = True, head_width=xmax/20)
plt.plot(xn,yyn, c='r', label =" $ y = -0.5 \cdot sign(x) \cdot x^2 $", linewidth = 0.5)
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$ u(x,y) $')
plt.show()


# profilo u da stampare su file



fig, ax = plt.subplots()
udk = ud.real[ystartk:yendk+1,xstartk:xendk+1]
udkd = udk*csp
conto = ax.contourf(Xkd, Ykd, udkd/vkp, np.arange(-0.2,0.2,0.001),cmap='RdBu')
cb = fig.colorbar(conto, ticks=[-0.15, 0, 0.15])
cb.ax.locator_params(nbins=3)
#plt.contourf(Xkd, Ykd, udkd/vkp, np.arange(-0.2,0.2,0.001))
#plt.colorbar()
plt.arrow(-xlinmax*l/2.,-2*ylinmax*l/3.,0,ylinmax*l/3., length_includes_head = True, head_width=xlinmax*l/20)
plt.arrow(l*xlinmax/2.,2*ylinmax*l/3.,0,-ylinmax*l/3., length_includes_head = True, head_width=xlinmax*l/20)
plt.locator_params(nbins=3)
plt.xlabel('$ x \quad [au]$')
plt.ylabel('$ y \quad [au]$')
plt.title('$ u(x,y)/v_p$')
plt.show()


fig, ax = plt.subplots()
sdk = sd.real[ystartk:yendk+1,xstartk:xendk+1]
sdkd = sdk
#plt.contourf(Xkd, Ykd, sdkd, np.arange(-3,3,0.01))
conto = ax.contourf(Xkd, Ykd, sdkd, np.arange(-3,3,0.01),cmap='RdBu')
cb = fig.colorbar(conto, ticks=[-2.5, 0, 2.5])
cb.ax.locator_params(nbins=3)
#plt.colorbar()
#plt.arrow(-xlinmax*l/2.,-2*ylinmax*l/3.,0,ylinmax*l/3., length_includes_head = True, head_width=xlinmax*l/20)
#plt.arrow(l*xlinmax/2.,2*ylinmax*l/3.,0,-ylinmax*l/3., length_includes_head = True, head_width=xlinmax*l/20)
plt.locator_params(nbins=3)
plt.plot(-2*l*np.ones(2),[2*l-22*l,2*l+22*l],color='orange')
plt.plot(-4*l*np.ones(2),[8*l-22*l,8*l+22*l],color='blue')
plt.plot(-6*l*np.ones(2),[18*l-22*l,(ylinmax-0.5)*l],color='green')
plt.xlabel('$ x\quad [au]$')
plt.ylabel('$ y\quad[au]$')
plt.title('$ \sigma(x,y)=\Sigma _1/\Sigma _0$')
plt.show()

'''
udk = ud.real[ystartk:yendk+1,xstartk:xendk+1]
plt.contourf(Xk, Yk, udk)
plt.colorbar()
plt.xlabel('$ x $')
plt.ylabel('$ y $')
plt.title('$ u(x,y) linear Rafikov $')
plt.show()
'''
# ora stampo su file i valori di vx

file = open("LinearU.dat", "w")
file.write("u\n")
np.savetxt(file, udk, fmt='%.3f')
file.close()


# PROFILI FINALI RAFIKOV


yc = 22

def profile(x_disp, x_arr, vxy, y_arr,yc):

   newx = np.abs(x_arr-x_disp)
   index = newx.argmin()
   profile = vxy[:,index]/np.sqrt(np.abs(x_disp))
   rhs = np.sign(x_disp)*0.5*x_disp**2 #right hand side
   y1 = y_arr[(y_arr+rhs>-yc) & (y_arr+rhs<yc)]
   y1 = y1 +rhs*np.ones(len(y1))
   p1 = profile[(y_arr+rhs>-yc) & (y_arr+rhs<yc)]
   return y1,p1



for u in [-2,-4,-6,-8]:
   y1,p = profile(u,x,sd.real,y,yc)
   plt.plot(y1,p, label = 'x = '+str(u) )

plt.legend()
plt.grid(True)
plt.xlabel('$\eta = y + sgn(x)\cdot x^2/2$')
plt.ylabel('$ \sigma /|x|^{1/2} $')
plt.title("Surface density profile")
plt.show()

y1,p = profile(-2,x,sd.real,y,yc)
plt.plot(y1,p, label = "x' = "+str(-2),color = 'orange' )

y1,p = profile(-4,x,sd.real,y,yc)
plt.plot(y1,p, label = "x' = "+str(-4), color = 'blue' )

y1,p = profile(-6,x,sd.real,y,yc)
plt.plot(y1,p, label = "x' = "+str(-6), color = 'green' )

plt.legend()
plt.grid(True)
plt.xlabel("$\eta = y' + sgn(x')\cdot x'^2/2$")
plt.ylabel("$ \sigma /|x'|^{1/2} $")
#plt.title("Surface density profile")
plt.show()

#################################################### F I N A L   I N F O ####################################################################
# calcolo la derivata del profilo
x0= +2
y0,p0 = profile(x0,x,sd.real,y,yc) #ottengo il profilo con x0 = +2
np.savetxt('ky'+str(np.log2(Ny))+'x'+str(x0)+'.dat', np.transpose([y0, p0]), delimiter=' ')

x0= -2
y0,p0 = profile(x0,x,sd.real,y,yc) #ottengo il profilo con x0 = -2
np.savetxt('ky'+str(np.log2(Ny))+'x'+str(x0)+'.dat', np.transpose([y0, p0]), delimiter=' ')


dy = y[1]-y[0]
dpdy = np.gradient(p0,dy)
plt.plot(y0,p0, label = "$\sigma (x_0,y-x_0^2/2)/|x_0|^{1/2}$")
plt.plot(y0,dpdy, label = "derivative")
plt.legend()
plt.show()
print("Il massimo della derivata è : "+str(max(dpdy)))


dpdysign = dpdy*np.sign(p0)
plt.plot(y0,p0, label = "$\sigma (x_0,y-x_0^2/2)/|x_0|^{1/2}$")
plt.plot(y0,dpdysign, label = "$\chi ' * sign(\chi)$")
plt.legend()
plt.show()
print("Il massimo della derivata*sign è : "+str(max(dpdysign)))

#calcolo gli zeri della funzione

zeri = []
for i in range(len(y0)):
   if p0[i] == 0:
      zeri = np.append(zeri,[y0[i]])
   elif i!=(len(y0)-1) and p0[i]*p0[i+1]<0:
      zeri = np.append(zeri, [0.5*(y0[i]+y0[i+1])])

#calcolo gli integrali, quello totale e delle singole parti. Per un np.array uso trapz. Uso le funzioni in functions.py se ho
#l'epressione analitica dell'integranda.

  #integrale totale
int_tot = np.trapz(p0, dx = dy ) # mi aspetto sia zero

  #intevallo 1
p0_range1 = p0[y0<zeri[0]]
int1 = np.trapz(p0_range1, dx = dy )

print(zeri)

  #intervallo 2
p0_range2 = p0[(y0>zeri[0]) & (y0<zeri[1])]
int2 = np.trapz(p0_range2, dx = dy )

  #intervallo 3
p0_range3 = p0[y0>zeri[1]]
int3 = np.trapz(p0_range3, dx = dy )


print("Gli integrali sono:")
print("   int_tot = "+str(int_tot))
print("      int1 = "+str(int1))
print("      int2 = "+str(int2))
print("      int3 = "+str(int3))
