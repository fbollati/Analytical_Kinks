from astropy import constants as const
import numpy as np


def mysplit(word):
    return [char for char in word]


def read_parameters(file):

    parobj = open(file)
    params = {} # this is a dictionary
    for line in parobj:
        line = line.strip() # removes additional spaces after a line and also the newline character
        if line and line[0] != '%': # the first 'if line' allows to remove blank lines.
            line = line.split()
            if len(line) == 1:
                params[line[0]] = None

            elif len(line) == 2:
                params[line[0]] = line[1]

            elif len(line) > 2:
                params[line[0]] = [ line[1] ]
                for i in range(2,len(line)):
                    params[line[0]] += [ line[i] ]

    return params



def init(file):

    # PHYSICAL CONSTANTS
    global G,au,Msun,Mj

    G = const.G.cgs.value
    au = const.au.cgs.value
    Msun = const.M_sun.cgs.value
    Mj = const.M_jup.cgs.value


    # PARAMETERS
    global M, Mp, Rdisc, Rp, cw, q, p, hr, malpha, indad
    global PA, PAp, i, xi
    global vchs, delta_vch
    global Nr, Nphi, Rmin, x_match
    global show,xrange,yrange, density

    params = read_parameters(file)

    M = float(params['Mstar'])
    Mp = float(params['Mplanet'])
    Rdisc = float(params['Rdisc'])
    Rp = float(params['Rplanet'])
    cw = int(params['cw'])
    q = float(params['q'])
    p = float(params['p'])
    hr = float(params['hrp'])
    if 'malpha' in params.keys():
        malpha = float(params['malpha'])
    else:
        malpha = 0
    indad = float(params['indad'])
    PA = float(params['PA']) * np.pi/180
    PAp = float(params['PAp']) * np.pi/180
    i = float(params['inclination']) * np.pi/180

    # auxiliary angle needed for PAp
    xi = np.arctan( np.tan(PAp)/np.cos(i) )

    if PAp == np.pi/2 or PAp == 3*np.pi/2:
        xi = PAp
    elif PAp > np.pi/2 and PAp < 3*np.pi/2:
        xi += np.pi
    elif PAp > 3*np.pi/2:
        xi += 2*np.pi

    if 'channel_velocities' in params.keys():
        vchs = np.array([ float(vch) for vch in params['channel_velocities']])
        vchs.sort()
    else:
        vchs = 0

    if 'channel_resolution' in params.keys():
        delta_vch = float(params['channel_resolution'])
    else:
        delta_vch = 0.05

    if 'show' in params.keys():
        show = True
    else:
        show = False

    if 'density' in params.keys():
        density = True
    else:
        density = False

    if 'xrange' in params.keys():
        xrange = np.array([ float(params['xrange'][0]), float(params['xrange'][1]) ])
    else:
        xrange = 0
    if 'yrange' in params.keys():
        yrange = np.array([ float(params['yrange'][0]), float(params['yrange'][1]) ])
    else:
        yrange = 0


    Nr = int(params['Nr'])
    Nphi = int(params['Nphi'])
    x_match = float(params['x_MatchLinearAndNonlinear'])

    if 'Rmin' in params.keys():
        Rmin = float(params['Rmin'])
    else:
        Rmin = Rdisc/50


    global R,PHI,X,Y
    global disc_edge

    r = np.linspace(Rmin,Rdisc,Nr)
    phi = np.linspace(0,2*np.pi,Nphi)
    R,PHI = np.meshgrid(r,phi)
    X = R*np.cos(PHI)
    Y = R*np.sin(PHI)

    disc_edge = np.array( [ [Rdisc*np.cos(i),Rdisc*np.sin(i)] for i in np.linspace(0,2*np.pi,200) ] )

    global Xlin, Ylin, Ulin,Vlin,Dlin  # cartesian mesh of the linear box domain

    # OTHER PARAMETERS (section 2.3 Bollati et al. 2021)

    global eta_tilde
    global t0
    global C # Eq. (16) Bollati et al. 2021

    t0 = 1.89

    global vKp  # planet keplerian velocity
    global csp  # local sound speed (at Rp)
    global l    # unit lenght l = csp/|2A(rp)| unit of length Goodman & Rafikov 2001 ( = (2/3)h_p )
    global mth  # thermal mass
    global betap

    vKp = np.sqrt(G*M*Msun/(Rp*au))*10**(-5) #[km/s]
    csp = vKp*hr    # [km/s]  (cs/vK = h/r)
    l = (2/3)*Rp*hr # [au]
    mth = (l*au)*(csp*10**5)**2/(G*Mj)  # [M_jupyter] reference mass of Goodman & Rafikov 2001 eq.(19) ( = thermal mass Eq. 8 Bollati et al. 2021)
    betap = Mp/mth
    #C = (indad + 1)*betap*C0/2**(3/4)
