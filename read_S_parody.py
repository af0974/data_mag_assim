import sys
import struct		# to read crazy fortran records
import numpy as np
import rev_process as rp
fname='St=30.11091768.truth'
job_out = 'parody'


##### READ PARODY 'S' FILE #####
print('reading parody_pdaf file ', fname, ' ...')
f = open(fname,"rb")

### header
head = f.read(108)
#h = struct.unpack(">iiiidiidiidiidiidiidiidiidiidiidiidiiiiiiiiiiiii", head) ## big endian
h = struct.unpack("<idddddddddddiiii", head) ## little endian 
par = {}
par['version'] = h[0]
par['time'] = h[1]
par['dt'] = h[2]
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
print("radial grid from r=%f to %f" % (r[0], r[-1]))

nlat = par['Nlat'] 
colat = np.fromfile( f, dtype="<d", count = nlat )
print("colatitudinal grid from theta=%f to %f" % (colat[0], colat[-1]))


nlon = par['Nlon']

# initial shape is Vt(1:nlong,1:nlat)
Vt = np.fromfile( f, dtype="<d", count= nlat * nlon )
Vp = np.fromfile( f, dtype="<d", count= nlat * nlon )
Br = np.fromfile( f, dtype="<d", count= nlat * nlon )
Bt = np.fromfile( f, dtype="<d", count= nlat * nlon )

print(f.tell(), " bytes parsed.")
f.close()
### DONE READING PARODY ###
F = np.reshape( Br, (nlat,nlon) ) 
theta = colat
phi = 2.*np.pi * np.linspace(0, 1, nlon, endpoint=False)
print(phi)
rp.mollweide_surface(F, theta, phi, fname='br_'+fname, vmax=None, vmin=None, Title=None, positive=False, cmap=None, unit="nondim")
