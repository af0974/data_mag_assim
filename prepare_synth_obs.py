import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shtns
import cartopy.crs as ccrs
import cartopy.util as cutil
import readSparody as rs
import rev_process as rp
import pointwise_obs as ptw
import parody_toolbox_wf as pt
from tg_plots import *

# Load observation file

obs = pd.read_table("obs_time.txt", sep='\s', \
        names = ['elem', 'year', 'lon', 'lat', 'val', 'dval', 'source'])
#print(obs)

# Initialize SHTNS

lmax = 64
mmax = 64
nlon = 192
nlat = 96
sh = shtns.sht(lmax,mmax, norm=shtns.sht_fourpi | shtns.SHT_NO_CS_PHASE | shtns.SHT_REAL_NORM)
sh.set_grid(nlat, nlon)
theta = np.arccos(sh.cos_theta)
phi =  (2. * np.pi / float(nlon))*np.arange(nlon)

# Directories
path_ens = '/ptmp/sanchezs/Case_-1_newcode_NR50/'

# Load true state
fname = 'St=16.84071360.long'
br, thetan, phin, r = rs.readS(path_ens + fname, verbose = True)

br = -br

#rp.mollweide_surface(Br, thetan, phin, fname='br_'+fname, \
#       vmax=None, vmin=None, Title=None, positive=False, cmap=None, unit="nondim")

# Enter parameters for field and time rescaling
# In the future, this will be accessed via workflow outputs

# Select only one observation

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



