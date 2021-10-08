import struct           
import numpy as np


def get_data_from_S_file(fname, Verbose=False): 

##### READ PARODY 'S' FILE #####
    if Verbose is True: 
        print('reading parody_pdaf file ', fname, ' ...')
    f = open(fname,"rb")

### header
    head = f.read(108)
    h = struct.unpack("<idddddddddddiiii", head) ## little endian 
    par = {}
    par['version'] = h[0]
    par['time'] = h[1]
    par['dt'] = h[2]
    dt = par['dt'] 
    par['deltaU'] = h[3]
    par['Coriolis'] = h[4]
    par['Lorentz'] = h[5]
    par['Buoyancy'] = h[6]
    par['ForcingU'] = h[7]
    par['DeltaT'] = h[8]
    par['ForcingT'] = h[9]
    par['DeltaB'] = h[10]
    par['ForcingB'] = h[11]
# grid parameters
    par['NF'] = h[12]
    par['Nlat'] = h[13]
    par['Nlon'] = h[14]
    par['Mc'] = h[15]

    if par['version'] != 4:
             print("error, only version 4 is supported")
             exit()

### radial grid in the fluid
    nf = par['NF'] 
    r = np.fromfile( f, dtype="<d", count = nf )
    if Verbose is True: 
        print("radial grid from r=%f to %f" % (r[0], r[-1]))

    nlat = par['Nlat'] 
    colat = np.fromfile( f, dtype="<d", count = nlat )
    if Verbose is True: 
        print("colatitudinal grid from theta=%f to %f" % (colat[0], colat[-1]))

    nlon = par['Nlon']

# initial shape is Vt(1:nlong,1:nlat)
    Vt = np.fromfile( f, dtype="<d", count= nlat * nlon )
    Vp = np.fromfile( f, dtype="<d", count= nlat * nlon )
    Br = np.fromfile( f, dtype="<d", count= nlat * nlon )
    dBr = np.fromfile( f, dtype="<d", count= nlat * nlon )
    if Verbose is True: 
        print(f.tell(), " bytes parsed.")
    f.close()
### DONE READING PARODY ###
    Br_2D = np.reshape( Br, (nlat,nlon) ) 
    dBr = dBr / dt
    dBrdt_2D = np.reshape( dBr,  (nlat,nlon) )
    theta = colat
    phi = 2.*np.pi * np.linspace(0, 1, nlon, endpoint=False)
    
    return nlat, nlon, par['time'], Br_2D, dBrdt_2D

def compute_glmhlm_from_brlm(br_lm, sh, ltrunc = None, bscale = None):
#
#   inputs :
#   br_lm: the SH expansion of br at the surface of the dynamo region
#   sh: the current shtns framework
#
    a = 6371.2
    c = 3485.

    azsym = sh.mres

    if ltrunc == None:
        ltrunc = sh.lmax
    glm = np.zeros( (ltrunc+1,ltrunc+1), dtype=float)
    hlm = np.zeros( (ltrunc+1,ltrunc+1), dtype=float)

    for il in range(1,ltrunc+1):
        for im in range(0,il+1,azsym):
            fact = ((il+1)/np.sqrt(2*il+1))*(a/c)**(il+2)
            glm[il,im] =      np.real(br_lm[sh.idx(il,im)]) / fact
            hlm[il,im] = -1 * np.imag(br_lm[sh.idx(il,im)]) / fact

    if bscale != None:
        glm = bscale * glm
        hlm = bscale * hlm

    return glm, hlm
