import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shtns
import glob
import cartopy.crs as ccrs
import cartopy.util as cutil
import readSparody as rs
import rev_process as rp
import pointwise_obs as ptw
import parody_toolbox_wf as pt
from tg_plots import *

# Load observation file

obs = pd.read_table("obs_time_1949.txt", sep='\s', \
        names = ['elem', 'year', 'lon', 'lat', 'val', 'dval', 'source'])
# Change to micro T
obs.val = obs.val * 1.e-3
obs.dval = obs.dval * 1.e-3

# Initialize SHTNS

lmax = 64
mmax = 64
nlon = 192
nlat = 96
sh = shtns.sht(lmax,mmax, norm=shtns.sht_fourpi | shtns.SHT_NO_CS_PHASE | \
	 shtns.SHT_REAL_NORM)
sh.set_grid(nlat, nlon)
theta = np.arccos(sh.cos_theta)
phi =  (2. * np.pi / float(nlon))*np.arange(nlon)

# Directories
path_ens = '/ptmp/sanchezs/Case_-1_newcode_NR50/'
path_tru = '/ptmp/sanchezs/Case_-1_newcode_NR50/'

# Read ensemble
files = glob.glob(path_ens + 'St*long')
dim_ens = len(files)
#dim_ens = 60

xf_ens = np.zeros((sh.nlm, dim_ens), dtype=complex)
br_lm_ens = np.zeros((sh.nlm, dim_ens), dtype=complex)
br_ens = np.zeros((nlat,nlon,dim_ens))
for ie in range(dim_ens):
    fname = files[ie]
    br, thetan, phin, r = rs.readS(fname, verbose = False)
    br_ens[:,:,ie] = -br
    br_lm = sh.analys(-br)
    br_lm_ens[:,ie] = br_lm
    xf = pt.brlm2ybpr(br_lm, sh, r[-1])
    xf[0] = 0.
    xf_ens[:,ie] = xf

# Calculate rescaling for field

xf_mean = np.mean(xf_ens, axis=1)
g10 = ((3485.0 / 6371.2)**3) * (np.sqrt(3)/r[-1]) * np.real(xf_mean[1])
b_c = 30.305 / np.absolute(g10) # in microT
print('calibrating factor in micro T : ', b_c)

xf_ens = xf_ens * b_c
xf_mean = xf_mean * b_c
br_lm_ens = br_lm_ens * b_c
br_ens = br_ens * b_c

# Plot mean field

br_lm_mean = np.mean(br_lm_ens, axis=1)
br_mean = np.mean(br_ens, axis=2)

fig = radialContour(br_mean.T, vmax=800., vmin=-800., levels=51)
plt.savefig('xf_Br_mean.png')
plt.show()

# Make synthetic observations for 1975.5 based on observatory data
# Make also synthetic spectral obs later

y_s = obs.dval * 1000.
print(y_s)
y_lon = (obs.lon + 180.)*np.pi/180. 
y_lat = (90. - obs.lat)*np.pi/180.
y_elem = obs.elem

dim_obs = len(y_s)
R = np.diag(y_s**2)

fname = path_tru + 'St=50.26811194.truthKm' 
brt, thetan, phin, r = rs.readS(fname, verbose = False)
brt = - brt * b_c
brt_lm = sh.analys(brt)
xt = pt.brlm2ybpr(brt_lm, sh, r[-1])
xt[0] = 0.

y_o = np.zeros_like(y_s)
for io in range(dim_obs):
    if y_elem[io] == 'X':
        y_o[io] = ptw.green_x(brt.T, y_lat[io], y_lon[io], sh)  
    if y_elem[io] == 'Y':
        y_o[io] = ptw.green_y(brt.T, y_lat[io], y_lon[io], sh)
    if y_elem[io] == 'Z':
        y_o[io] = ptw.green_z(brt.T, y_lat[io], y_lon[io], sh)
print(y_o)

# Plot synthetic obs against true values 

fig = radialContour(brt.T, vmax=800., vmin=-800., levels=51)
plt.savefig('xt_Br_cmb.png')
plt.show()

tx, ty, tz = ptw.surf_xyz(xt,phi,theta,r[-1],sh)

y_lat_x = np.array(y_lat[y_elem == 'X'])
y_lon_x = np.array(y_lon[y_elem == 'X'])
y_o_x = np.array(y_o[y_elem == 'X'])
y_lat_y = np.array(y_lat[y_elem == 'Y'])
y_lon_y = np.array(y_lon[y_elem == 'Y'])
y_o_y = np.array(y_o[y_elem == 'Y'])
y_lat_z = np.array(y_lat[y_elem == 'Z'])
y_lon_z = np.array(y_lon[y_elem == 'Z'])
y_o_z = np.array(y_o[y_elem == 'Z'])

fig = radialContour_data(tx.T, y_lon_x, y_lat_x, y_o_x, vmax=np.max(tx), \
	vmin=np.min(tx), levels=51)
plt.savefig('xt_xobs.png')
plt.show()

fig = radialContour_data(ty.T, y_lon_y, y_lat_y, y_o_y, vmax=np.max(ty), \
	vmin=np.min(ty), levels=51)
plt.savefig('xt_yobs.png')
plt.show()

fig = radialContour_data(tz.T, y_lon_z, y_lat_z, y_o_z, vmax=np.max(tz), \
	vmin=np.min(tz), levels=51)
plt.savefig('xt_zobs.png')
plt.show()

# Make obs from forecast ensemble

hxf_ens = np.zeros((dim_obs, dim_ens))
for ie in range(dim_ens):
#    print(ie)
    for io in range(dim_obs):
        if y_elem[io] == 'X':
            hxf_ens[io,ie] = ptw.green_x(br_ens[:,:,ie].T, y_lat[io], y_lon[io], sh)
        if y_elem[io] == 'Y':
            hxf_ens[io,ie] = ptw.green_y(br_ens[:,:,ie].T, y_lat[io], y_lon[io], sh)
        if y_elem[io] == 'Z':
            hxf_ens[io,ie] = ptw.green_z(br_ens[:,:,ie].T, y_lat[io], y_lon[io], sh)

hxf_mean = np.zeros_like(y_o)
for io in range(dim_obs):
    if y_elem[io] == 'X':
        hxf_mean[io] = ptw.green_x(br_mean.T, y_lat[io], y_lon[io], sh)
    if y_elem[io] == 'Y':
        hxf_mean[io] = ptw.green_y(br_mean.T, y_lat[io], y_lon[io], sh)
    if y_elem[io] == 'Z':
        hxf_mean[io] = ptw.green_z(br_mean.T, y_lat[io], y_lon[io], sh)

# Calculate and plot covariance matrix

Pf = np.zeros((sh.nlm, sh.nlm), dtype=complex)
xf_m1 = xf_ens - np.dot(np.expand_dims(xf_mean, 1), \
        np.ones((1, dim_ens), dtype=complex))
xf_std = np.std(xf_ens, axis=1, dtype=np.float64)
Pf = np.dot(xf_m1, np.conj(xf_m1).T)/(dim_ens - 1)
xf_corr = Pf / np.dot(np.expand_dims(xf_std, 1), \
         np.expand_dims(xf_std, 1).T)

fig = plt.figure()
plt.imshow(np.real(xf_corr[0:130,0:130]), cmap=plt.cm.seismic, \
        vmin=-1., vmax=1.)
plt.savefig('Pf_Br_covar.png')
plt.show()

# Build HP and HPH

hxf_m1 = hxf_ens - np.dot(np.expand_dims(hxf_mean, 1), np.ones((1, dim_ens)))
HPf = np.dot(hxf_m1, np.conj(xf_m1).T)/(dim_ens - 1)
HPfH = np.dot(hxf_m1, hxf_m1.T)/(dim_ens - 1)
HPfH = HPfH + R
print(HPfH.shape)
print(HPf.shape)

# Innovation matrix
# Next : add noise to observations ...

d = np.dot(np.expand_dims(y_o, 1), np.ones((1, dim_ens))) - hxf_ens
print(np.expand_dims(y_o, 1).shape, d.shape)

# Prepare analysis and calculate error

xa_ens = np.zeros((sh.nlm, dim_ens), dtype=complex)
for ie in range(dim_ens):
    # HPH * b = d
    b = np.linalg.solve(HPfH, d[:,ie])
    # analysis : x^a = x^f + HP' * (HPH +R)^-1 * (y^o - H(x^f))
    xa_ens[:,ie] = xf_ens[:,ie] + np.squeeze(np.dot(np.conj(HPf).T, np.expand_dims(b, 1)))
    #print(np.squeeze(np.dot(np.conj(HPf).T, np.expand_dims(b, 1))))

xa_mean = np.mean(xa_ens, axis=1)

bra_lm_mean = pt.ybpr2brlm(xa_mean, sh, r[-1])
bra_mean = sh.synth(bra_lm_mean)

fig = radialContour(bra_mean.T, vmax=800., vmin=-800., levels=51)
plt.savefig('xa_Br_cmb.png')
plt.show()

ax, ay, az = ptw.surf_xyz(xa_mean,phi,theta,r[-1],sh)

fig = radialContour_data(ax.T, y_lon_x, y_lat_x, y_o_x, vmax=np.max(tx), \
        vmin=np.min(tx), levels=51)
plt.savefig('xa_xobs.png')
plt.show()

fig = radialContour_data(ay.T, y_lon_y, y_lat_y, y_o_y, vmax=np.max(ty), \
        vmin=np.min(ty), levels=51)
plt.savefig('xa_yobs.png')
plt.show()

fig = radialContour_data(az.T, y_lon_z, y_lat_z, y_o_z, vmax=np.max(tz), \
        vmin=np.min(tz), levels=51)
plt.savefig('xa_zobs.png')
plt.show()

# Plot error with respect to ensemble size


# Apply to real data


# Create localization based on spectral decomposition


# If it works, see if possible to calculate directly
# green functions in spectral form


  
#rp.mollweide_surface(Br, thetan, phin, fname='br_'+fname, \
#       vmax=None, vmin=None, Title=None, positive=False, cmap=None, unit="nondim")

# Enter parameters for field and time rescaling
# In the future, this will be accessed via workflow outputs

# Select only one observation

"""

obst = obs.iloc[0]
print(obst)
print(obst.elem, (90. - obst.lat)*np.pi/180., (obst.lon + 180.)*np.pi/180.)

# Observations with old routine

olon = (obst.lon + 180.)*np.pi/180.
olat = (90. - obst.lat)*np.pi/180.

obsx = ptw.green_x(br.T, olat, olon, sh)
obsy = ptw.green_y(br.T, olat, olon, sh)
obsz = ptw.green_z(br.T, olat, olon, sh)

print(obsx,obsy,obsz)

# Observations with VL routine

# Calculate X, Y and Z at surface

br_lm = sh.analys(br)
print(br_lm)
ybpr_lm = pt.brlm2ybpr(br_lm, sh, r[-1])
print(ybpr_lm)
ybpr_lm[0] = 0.

fx, fy, fz = ptw.surf_xyz(ybpr_lm,phi,theta,r[-1],sh)

# Superimpose plot of X Y Z and point-wise observation

print('obsx:',obsx,obsx.size,br_lm.size)

vmax = np.max(fx)
vmin = np.min(fx)
fig = radialContour_data(fx.T, olon, olat, obsx, vmax=vmax, vmin=vmin, levels=51)
plt.savefig('x_surf_data.png')
plt.show()

vmax = np.max(fy)
vmin = np.min(fy)
fig = radialContour_data(fy.T, olon, olat, obsy, vmax=vmax, vmin=vmin, levels=51)
plt.savefig('y_surf_data.png')
plt.show()

vmax = np.max(fz)
vmin = np.min(fz)
fig = radialContour_data(fz.T, olon, olat, obsz, vmax=vmax, vmin=vmin, levels=51)
plt.savefig('z_surf_data.png')
plt.show()

# Next : routine for calculating covariance matrix and 
#	analysis step


"""
