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

obst = obs.iloc[0]

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

# Make synthetic observations for 1975.5 based on observatory data
# Make also synthetic spectral obs later

o_lon = (obst.lon + 180.)
o_lat = (90. - obst.lat)

y_lon = (obst.lon + 180.)*np.pi/180. 
y_lat = (90. - obst.lat)*np.pi/180.

fname = path_tru + 'St=50.26811194.truthKm' 
brt, thetan, phin, r = rs.readS(fname, verbose = False)
brt = - brt
print(brt.shape)
brt_lm = sh.analys(brt)
xt = pt.brlm2ybpr(brt_lm, sh, r[-1])
xt[0] = 0.

# Observation in spat

y_1 = ptw.green_x(brt.T, y_lat, y_lon, sh) 
print('y_1 = ', y_1)

# Observatio in spec
 
Gx = ptw.gf_x(brt.T, y_lat, y_lon, sh).T
print(Gx.shape)
Gxlm = sh.analys(Gx) * 4 * np.pi
y_2 = np.sum((Gxlm*np.conj(brt_lm) + np.conj(Gxlm)*brt_lm)/2.)
print('y_2 = ', y_2)
print('difference : ', y_2 - y_1)

# Try V Lesur routine

print('brt_lm : ', brt_lm.shape)
glm, hlm, ghlm = rp.compute_glmhlm_from_brlm(brt_lm, sh)
#ghlm = rp.compute_ghlm_from_glmhlm(glm, hlm)
print('ghlm : ', ghlm.shape)

SHBX = rp.SHB_X(o_lat, o_lon, 6371.2, ll = sh.lmax)
print('SHBX :', SHBX.shape)

y_3 = np.dot(SHBX, ghlm)
print('y_3 = ', y_3)
print('difference : ', y_3 - y_1)



"""

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

# Plot error with respect to ensemble size


# Apply to real data


# Create localization based on spectral decomposition


# If it works, see if possible to calculate directly
# green functions in spectral form

"""
