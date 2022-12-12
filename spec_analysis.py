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

obst = obs.iloc[0:20]
dim_obs = 1
print(obst)

"""

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
path_sav = 'Test_1obs_loc/'

# Read ensemble
files = glob.glob(path_ens + 'St*long')
dim_ens = len(files)
dim_ens = 200

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
plt.savefig(path_sav + 'xf_Br_mean.png')
plt.show()

# Make synthetic observations for 1975.5 based on observatory data
# Make also synthetic spectral obs later

y_s = np.atleast_1d(obst.dval * 10.)
y_lon = np.atleast_1d(obst.lon + 180.)*np.pi/180. 
y_lat = np.atleast_1d(90. - obst.lat)*np.pi/180.
y_elem = obst.elem

print(y_s, np.isscalar(y_s))

dim_obs = len(y_s)
R = np.diag(y_s**2)

fname = path_tru + 'St=50.26811194.truthKm' 
brt, thetan, phin, r = rs.readS(fname, verbose = False)
brt = - brt * b_c
brt_lm = sh.analys(brt)
xt = pt.brlm2ybpr(brt_lm, sh, r[-1])
xt[0] = 0.

y_o = np.zeros_like(y_s)
y_2 = np.zeros_like(y_s)
Gxlm = np.zeros((sh.nlm, dim_obs), dtype=complex)
for io in range(dim_obs):
    if y_elem[io] == 'X':
        y_o[io] = ptw.green_x(brt.T, y_lat[io], y_lon[io], sh)  
        Gx = ptw.gf_x(brt.T, y_lat[io], y_lon[io], sh).T
        Gxlm[:,io] = sh.analys(Gx) * 4 * np.pi
        y_2[io] = np.sum((Gxlm[:,io]*np.conj(brt_lm) + np.conj(Gxlm[:,io])*brt_lm)/2.)  
print('y_o, y_2, y_s : ', y_o, y_2, y_s)

# Plot synthetic obs against true values 

fig = radialContour(brt.T, vmax=800., vmin=-800., levels=51)
plt.savefig(path_sav + 'xt_Br_cmb.png')
plt.show()

tx, ty, tz = ptw.surf_xyz(xt,phi,theta,r[-1],sh)

y_lat_x = np.array(y_lat[y_elem == 'X'])
y_lon_x = np.array(y_lon[y_elem == 'X'])
y_o_x = np.array(y_o[y_elem == 'X'])

fig = radialContour_data(tx.T, y_lon_x, y_lat_x, y_o_x, vmax=np.max(tx), \
	vmin=np.min(tx), levels=51)
plt.savefig(path_sav + 'xt_xobs.png')
plt.show()

fx, fy, fz = ptw.surf_xyz(xf_mean,phi,theta,r[-1],sh)

fig = radialContour_data(fx.T, y_lon_x, y_lat_x, y_o_x, vmax=np.max(tx), \
        vmin=np.min(tx), levels=51)
plt.savefig(path_sav + 'xf_xobs.png')
plt.show()

# Make obs from forecast ensemble

hxf_ens = np.zeros((dim_obs, dim_ens))
hxf2_ens = np.zeros((dim_obs, dim_ens))
for ie in range(dim_ens):
    for io in range(dim_obs):
        if y_elem[io] == 'X':
            hxf_ens[io,ie] = ptw.green_x(br_ens[:,:,ie].T, y_lat[io], y_lon[io], sh)
            hxf2_ens[io,ie] = np.sum((Gxlm[:,io]*np.conj(br_lm_ens[:,ie]) + \
			 np.conj(Gxlm[:,io])*br_lm_ens[:,ie])/2.)

hxf_mean = np.zeros_like(y_o)
hxf2_mean = np.zeros_like(y_o)
for io in range(dim_obs):
    if y_elem[io] == 'X':
        hxf_mean[io] = ptw.green_x(br_mean.T, y_lat[io], y_lon[io], sh)
        hxf2_mean[io] = np.sum((Gxlm[:,io]*np.conj(br_lm_mean) + \
			np.conj(Gxlm[:,io])*br_lm_mean)/2.)
 
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
plt.savefig(path_sav + 'Pf_Br_covar.png')
plt.show()

# Build covariance localization

ltrunc = sh.lmax
loc1 = np.zeros((sh.nlm, sh.nlm), dtype=complex)
loc2 = np.zeros((sh.nlm, sh.nlm), dtype=complex)
for lmt in range(sh.nlm):
    for lm in range(sh.nlm):
        if (sh.m[lmt] == sh.m[lm]):
            loc1[lmt, lm] = 1.
            if ((sh.l[lmt] + sh.m[lmt])%2) == ((sh.l[lm] + sh.m[lm])%2):
                loc2[lmt, lm] = 1.

# Localize covariance
#loc = loc1
loc = np.ones((sh.nlm, sh.nlm), dtype=complex)
Pfl = Pf #* loc

fig = plt.figure()
plt.imshow(np.real(loc[0:130,0:130]), cmap='gray', \
        vmin=0., vmax=1.)
plt.savefig(path_sav + 'Loc.png')
plt.show()

corr_loc = xf_corr * loc
fig = plt.figure()
plt.imshow(np.real(corr_loc[0:130,0:130]), cmap=plt.cm.seismic, \
        vmin=-1., vmax=1.)
plt.savefig(path_sav + 'Pf_loc.png')
plt.show()

# Tricky : have to expand the dimensions to use the complex conjugate

Gxlml = np.zeros_like(Gxlm, dtype=complex)
for io in range(dim_obs):
    for lm in range (sh.nlm):
        Gxlml[lm,io] = Gxlm[lm,io] * sh.l[lm] * (sh.l[lm] + 1) / r[-1]

HPfl = np.zeros((dim_obs, sh.nlm), dtype=complex)
for io in range(dim_obs):
    for lm in range (sh.nlm):
        HPfl[io,lm] = np.sum((Gxlml[:,io]*np.conj(Pf[:,lm]) + \
			np.conj(Gxlml[:,io])*Pf[:,lm])/2.)

HPflH = np.zeros((dim_obs, dim_obs), dtype=complex)
for io in range(dim_obs):
    for lm in range (sh.nlm):
        HPflH[io,io] = np.sum((HPfl[io,:]*np.conj(Gxlml[:,io]) + \
			np.conj(HPfl[io,:])*(Gxlml[:,io]))/2.)  
# Try to make HPflH then

y_3 = np.sum((np.conj(Gxlml[:,0])*xt + Gxlml[:,0]*np.conj(xt))/2.)
print('y test 3: ', y_3)

# Build HP and HPH

hxf_m1 = hxf_ens - np.dot(np.expand_dims(hxf_mean, 1), np.ones((1, dim_ens)))
HPf = np.dot(hxf_m1, np.conj(xf_m1).T)/(dim_ens - 1)
HPfH = np.dot(hxf_m1, hxf_m1.T)/(dim_ens - 1)
print('HPH:', HPfH)
HPfH = HPfH + R

hxf2_m1 = hxf2_ens - np.dot(np.expand_dims(hxf2_mean, 1), np.ones((1, dim_ens)))
HPf2 = np.dot(hxf2_m1, np.conj(xf_m1).T)/(dim_ens - 1)
HPfH2 = np.dot(hxf2_m1, hxf2_m1.T)/(dim_ens - 1)
print('HPfH2:', HPfH2)
HPfH2 = HPfH2 + R

print('HPfH3:', HPflH)

print(HPf)
print(HPf2)
print(HPfl)

# Innovation matrix
# Next : add noise to observations ...

d = np.dot(np.expand_dims(y_o, 1), np.ones((1, dim_ens))) - hxf_ens
d2 = np.dot(np.expand_dims(y_o, 1), np.ones((1, dim_ens))) - hxf2_ens

# Prepare analysis and calculate error

xa_ens = np.zeros((sh.nlm, dim_ens), dtype=complex)
xa2_ens = np.zeros((sh.nlm, dim_ens), dtype=complex)
for ie in range(dim_ens):
    b = np.linalg.solve(HPfH, d[:,ie])
    xa_ens[:,ie] = xf_ens[:,ie] + np.squeeze(np.dot(np.conj(HPf).T, np.expand_dims(b, 1)))
    b2 = np.linalg.solve(HPfH2, d2[:,ie])
    xa2_ens[:,ie] = xf_ens[:,ie] + np.squeeze(np.dot(np.conj(HPfl).T, np.expand_dims(b2, 1)))

xa_mean = np.mean(xa_ens, axis=1)
xa2_mean = np.mean(xa2_ens, axis=1)

bra_lm_mean = pt.ybpr2brlm(xa_mean, sh, r[-1])
bra_mean = sh.synth(bra_lm_mean)

fig = radialContour(bra_mean.T, vmax=800., vmin=-800., levels=51)
plt.savefig(path_sav + 'xa_Br_cmb.png')
plt.show()

ax, ay, az = ptw.surf_xyz(xa_mean,phi,theta,r[-1],sh)

fig = radialContour_data(ax.T, y_lon_x, y_lat_x, y_o_x, vmax=np.max(tx), \
        vmin=np.min(tx), levels=51)
plt.savefig(path_sav + 'xa_xobs.png')
plt.show()

bra2_lm_mean = pt.ybpr2brlm(xa2_mean, sh, r[-1])
bra2_mean = sh.synth(bra2_lm_mean)

fig = radialContour(bra2_mean.T, vmax=800., vmin=-800., levels=51)
plt.savefig(path_sav + 'xa2_Br_cmb.png')
plt.show()

ax2, ay2, az2 = ptw.surf_xyz(xa2_mean,phi,theta,r[-1],sh)

fig = radialContour_data(ax2.T, y_lon_x, y_lat_x, y_o_x, vmax=np.max(tx), \
        vmin=np.min(tx), levels=51)
plt.savefig(path_sav + 'xa2_xobs.png')
plt.show()


# Plot error with respect to ensemble size


# Apply to real data


# Create localization based on spectral decomposition


# If it works, see if possible to calculate directly
# green functions in spectral form

"""
 
