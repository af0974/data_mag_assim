import numpy as np
from psv10_class import *
import random
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit
from mpi4py import MPI


def compute_Delta_QPM(QPMsimu, QPMearth, Verbose=False):
#   check definition of confidence interval for the sigma Eq. (9) of the paper
    if ( QPMsimu.a_med > QPMearth.a_med):
        denom = (QPMearth.a_high - QPMearth.a_med) + (QPMsimu.a_med - QPMsimu.a_low)
    else:
        denom = (QPMearh.a_med - QPMearth.a_low) + (QPMsimu.a_high - QPMsimu.a_med)
    deltaQPM_a = np.abs( QPMsimu.a_med - QPMearth.a_med ) / denom 
#
    if ( QPMsimu.b_med > QPMearth.b_med):
        denom = (QPMearth.b_high - QPMearth.b_med) + (QPMsimu.b_med - QPMsimu.b_low)
    else:
        denom = (QPMearth.b_med - QPMearth.b_low) + (QPMsimu.b_high - QPMsimu.b_med)
    deltaQPM_b = np.abs( QPMsimu.b_med - QPMearth.b_med ) / denom 
#
    if ( QPMsimu.delta_Inc_med > QPMearth.delta_Inc_med):
        denom = (QPMearth.delta_Inc_high - QPMearth.delta_Inc_med) + (QPMsimu.delta_Inc_med - QPMsimu.delta_Inc_low)
    else:
        denom = (QPMearth.delta_Inc_med - QPMearth.delta_Inc_low) + (QPMsimu.delta_Inc_high - QPMsimu.delta_Inc_med)
    deltaQPM_delta_Inc = np.abs( QPMsimu.delta_Inc_med - QPMearth.delta_Inc_med ) / denom 
#   Vpercent
    if ( QPMsimu.Vpercent_med > QPMearth.Vpercent_med):
        denom = (QPMearth.Vpercent_high - QPMearth.Vpercent_med) + (QPMsimu.Vpercent_med - QPMsimu.Vpercent_low)
    else:
        denom = (QPMearth.Vpercent_med - QPMearth.Vpercent_low) + (QPMsimu.Vpercent_high - QPMsimu.Vpercent_med)
    deltaQPM_Vpercent = np.abs( QPMsimu.Vpercent_med - QPMearth.Vpercent_med ) / denom 

#   Rev
    denom = QPMearth.taut_high - QPMearth.taut_med
    deltaQPM_rev = np.abs( QPMsimu.taut - QPMearth.taut_med ) / denom 

    if Verbose is True:
        print('deltaQPM_a         = %10.2f' % (deltaQPM_a) )
        print('deltaQPM_b         = %10.2f' % (deltaQPM_b) )
        print('deltaQPM_delta_Inc = %10.2f' % (deltaQPM_delta_Inc) )
        print('deltaQPM_rev       = %10.2f ' % (deltaQPM_rev) )
        print('deltaQPM_Vpercent  = %10.2f ' % (deltaQPM_Vpercent) )

    DeltaQPM = np.array([ deltaQPM_a, deltaQPM_b, deltaQPM_delta_Inc,  deltaQPM_rev,  deltaQPM_Vpercent])
    mask = ( DeltaQPM < 1. )
    if Verbose is True: 
        print()
        print( ' DeltaQPM is %10.2f ' %(np.sum(DeltaQPM) ) )
        print( ' QPM is %i ' % (np.sum(mask) ) ) 

def quadratic_disp(lat, alpha, beta):
    return alpha**2 + (beta*lat)**2

def angular_distance_2sphere( lat1, lon1, lat2, lon2, Verbose=False):

    lam1       = np.deg2rad(lat1) 
    lam2       = np.deg2rad(lat2) 
    dphi       = np.deg2rad(lon2 - lon1)

    if Verbose is True:
        print('delta calculation')
        print(dphi)
        print(lam1)
        print(lam2)
    delta = np.rad2deg( np.arccos( np.sin(lam1)*np.sin(lam2) + np.cos(lam1)*np.cos(lam2)*np.cos(dphi) )  )
    return delta

def get_ghlm_1d( br_lm, sh, ltrunc, verbose=True):
#   returns the Gauss coefficients
    a = 6371.2
    c = 3485.
    lmax = sh.lmax
    mmax = lmax
    azsym = sh.mres
    lm = 0
    ghlm = np.zeros( ltrunc * ( ltrunc + 2) , dtype=float)
    lm = -1 
    for il in range(1,ltrunc+1):
        fact = ((il+1)/np.sqrt(2*il+1))*(a/c)**(il+2)
        for im in range(0,il+1,azsym):
            if im == 0: 
                lm = lm + 1
                ghlm[lm] =       np.real(br_lm[sh.idx(il,im)]) / fact
            else:
                lm = lm + 1
                ghlm[lm] =       np.real(br_lm[sh.idx(il,im)]) / fact
                lm = lm + 1
                ghlm[lm] =  -1 * np.imag(br_lm[sh.idx(il,im)]) / fact
    return ghlm
#
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0: 
    print('parallel size is ', size)

QPMearth = QPM(type='QPM_std', acro='earth')

Filename='Sprain_etal_EPSL_2019_1-s2.0-S0012821X19304509-mmc7.txt'
datPSV10 = DataTable(Filename, type='PSV10')
lmax = datPSV10.l_trunc
nloc = len(datPSV10.location)
if rank == 0: 
    print("nloc =", nloc)

QPMsimu = QPM(type='QPM_std', acro='case9')

npzfile = np.load('t_gh.npz')
t = npzfile['t']
ghlm_arr = npzfile['gh']

nsamp = int( len(t))
simulation_time = t[-1] - t[0] 
time_intervals = np.ediff1d(t, to_end=[0.])
if rank == 0: 
    print('total number of samples = ', nsamp)
    print('total simulation time =', simulation_time)
    print(np.shape(time_intervals))
# compute gauss coefficients 
#ghlm_arr = np.zeros( (nsamp, lmax*(lmax+2)), dtype=float)
dipole_latitude = np.zeros(nsamp, dtype=float)
for i in range(nsamp):
    g10 = ghlm_arr[ i, 0]
    g11 = ghlm_arr[ i, 1]
    h11 = ghlm_arr[ i, 2]
    dipole_latitude[i] = np.rad2deg(np.arctan( g10 / np.sqrt(g11**2+h11**2) ))

#Rev criterion
#"normal" polarity 
mask_n = dipole_latitude > 45.
tau_n = np.sum(time_intervals[mask_n]) / simulation_time
#"reverse" polarity
mask_r = dipole_latitude < -45.
tau_r = np.sum(time_intervals[mask_r]) / simulation_time
#excursion time
mask_t = np.abs(dipole_latitude) < 45.
tau_t = np.sum(time_intervals[mask_t]) / simulation_time

QPMsimu.taut = tau_t
QPMsimu.taur = tau_r
QPMsimu.taun = tau_n

#print('tau_n, tau_r, tau_t', tau_n, tau_r, tau_t)
Rev = False
if ( (tau_n > tau_t and tau_r > tau_t)  and ( tau_t > 0.0375 and tau_t < 0.15 ) ):
    Rev = True

#print('Rev is ', Rev)

nbins = 19
bins = np.linspace(-90,90,nbins)
#number of localities
nloc_tot = len(datPSV10.location)
if rank == 0: 
    print('total number of localities = ', nloc_tot)
ndraw = 10000  
inc_anom = np.zeros( (ndraw, nbins), dtype=float)
scatter_squared = np.zeros( (ndraw, nloc_tot), dtype=float)
Vpercent = np.zeros( (ndraw), dtype=float ) 
bin_lat = np.zeros( nbins, dtype=float)
empty_bin = np.zeros( nbins, dtype=bool)
empty_bin[:] = True
r_earth = 6371.2e3
vdm_fact = 1.e7 * r_earth**3
#
#mpi 1D domain decomposition
ndraw_per_process = int(ndraw / size)
mydraw_beg = rank * ndraw_per_process
mydraw_end = mydraw_beg + ndraw_per_process
print('beg end ', mydraw_beg, mydraw_end, 'for process ', rank)
###
#
for idraw in range(mydraw_beg, mydraw_end): 
    if np.mod(idraw+1-mydraw_beg,ndraw_per_process/10) == 0:
        if rank == 0: 
            print('rank ', rank, ' performed ', idraw+1-mydraw_beg, 'draws', flush=True)
    VDM = None
    iloc_glob = -1
    for ibin in range(len(bins)-1):
        #print(datPSV10.latitude_in_deg)
        lambda_min = bins[ibin]
        lambda_max = bins[ibin+1]
        mask =  ( datPSV10.latitude_in_deg > lambda_min ) *  ( datPSV10.latitude_in_deg < lambda_max) 
        my_location = datPSV10.location[mask]
        my_nloc = len(my_location)
        if my_nloc > 0: 
            empty_bin[ ibin ] = False
            my_number_of_sites =  datPSV10.number_of_sites[mask]
            my_latitude_in_deg = datPSV10.latitude_in_deg[mask]
            my_longitude_in_deg = datPSV10.longitude_in_deg[mask]
            my_SHB_X = datPSV10.SHB_X[mask,:]
            my_SHB_Y = datPSV10.SHB_Y[mask,:]
            my_SHB_Z = datPSV10.SHB_Z[mask,:]
            n_north = None
            n_east = None
            n_down = None
            bin_lat[ibin] = np.mean(my_latitude_in_deg)
            Inc_GAD = np.arctan( 2. * np.tan(np.deg2rad(np.mean(my_latitude_in_deg))))
            for iloc in range(my_nloc):
                iloc_glob = iloc_glob + 1
                nsite = my_number_of_sites[iloc]
                #take nsite random samples
                timesteps = random.sample(range(nsamp), nsite)
                VGP_lat = []
                VGP_lon = []
                #for istep in range(nsite):
                #print(np.shape( np.reshape( np.repeat( np.sign( ghlm_arr[ timesteps,0]), 120, axis=0 ), (nsite,120) ) ) )
                gh = np.transpose( np.reshape( np.repeat( -1. * np.sign( ghlm_arr[ timesteps,0]), 120, axis=0 ), (nsite,120) ) * ghlm_arr[ timesteps, :] )
                #print(np.shape( -1.* np.sign(ghlm_arr[ timesteps,0])))
                #print(np.shape(gh))
                """
                print(my_SHB_X[iloc,:].flags)
                sys.exit()
                """
                X = np.dot(my_SHB_X[iloc,:], gh)
                Y = np.dot(my_SHB_Y[iloc,:], gh)
                Z = np.dot(my_SHB_Z[iloc,:], gh)
                F = np.sqrt( X**2 + Y**2 + Z**2 )
                H = np.sqrt( X**2 + Y**2 )
                Inc = np.arctan2( Z , H)
                Dec = np.arctan2( Y , X)
                g10 = gh[0]
                g11 = gh[1]
                h11 = gh[2]
                theta_mag = np.arctan2( np.sqrt(g11**2+h11**2) , g10 ) 
                if VDM is None: 
                    VDM = vdm_fact * F / np.sqrt( 1. + 3.*(np.cos(theta_mag))**2 )
                else:
                    VDM = np.concatenate( ( VDM,  vdm_fact * F / np.sqrt( 1. + 3.*(np.cos(theta_mag))**2  ) ), axis=None) 
                #print(np.shape(VDM), np.shape(F))
#
                
#VGP stuff
                #p = np.arctan(2. / np.tan(Inc) )
                p = np.arctan2(2.* H, Z )
                VGP_lam = np.arcsin( np.sin(np.deg2rad(my_latitude_in_deg[iloc])) * np.cos(p)\
                                   + np.cos(np.deg2rad(my_latitude_in_deg[iloc]))*np.sin(p)*np.cos(Dec) ) 
                #print(np.shape(VGP_lam))
                if np.isnan(VGP_lam.any()):
                    print('I am in bin ', ibin)
                    print(p, my_latitude_in_deg[iloc], np.rad2deg(Dec), np.rad2deg(Inc)) 
                    print(np.sin(np.deg2rad(my_latitude_in_deg[iloc])))
                    print(np.cos(np.deg2rad(my_latitude_in_deg[iloc])))
                    print(np.sin(p))
                    print(np.cos(Dec))
                    print(np.sin(np.deg2rad(my_latitude_in_deg[iloc])) * np.cos(p)\
                                      + np.cos(np.deg2rad(my_latitude_in_deg[iloc])*np.sin(p)*np.cos(Dec) ))
                    sys.exit()

                beta = np.rad2deg( np.arcsin( np.sin(p) * np.sin(Dec) / np.cos(VGP_lam) ) ) 
                VGP_phi = np.zeros(nsite, dtype=float)
                for istep in range(nsite): 
                    if ( np.cos(p[istep]) > ( np.sin(np.deg2rad(my_latitude_in_deg[iloc])) * np.sin(VGP_lam[istep]) ) ):
                        VGP_phi[istep] = my_longitude_in_deg[iloc] + beta[istep]
                    else:
                        VGP_phi[istep] = my_longitude_in_deg[iloc] + 180. - beta[istep]
                VGP_lat = np.rad2deg(VGP_lam)
                VGP_lon = np.mod(VGP_phi, 360.)
                #n_VGP = n_VGP + 1
#
                if n_north is None:
                    n_north =  X/F 
                    n_east =  Y/F 
                    n_down =  Z/F 
                else: 
                    n_north = np.concatenate( (n_north, X/F), axis=None)
                    n_east = np.concatenate( (n_east, Y/F), axis=None)
                    n_down = np.concatenate( (n_down, Z/F), axis=None)
            #
                #VGP_lat = np.array(VGP_lat)
                #VGP_lon = np.array(VGP_lon)
                VGP_lat_avg = np.mean(VGP_lat)
                VGP_lon_avg = np.mean(VGP_lon)
                n_VGP = len(VGP_lat)
                Verbose = False
                """
                if ibin==1:
                    print('n_VGP=', n_VGP)
                    print('mean VGP lat', VGP_lat_avg)
                    print('VGP lat', VGP_lat)
                    print('mean VGP lon', VGP_lon_avg)
                    print('VGP lon', VGP_lon)
                    #Verbose = True
                """
                delta = angular_distance_2sphere( VGP_lat, VGP_lon, \
                                              VGP_lat_avg * np.ones_like(VGP_lat), VGP_lon_avg * np.ones_like(VGP_lat), Verbose=Verbose)
                """
                if ibin == 9: 
                    print(delta)
                    print('size delta = ', np.size(delta))
                """
                scatter_squared[idraw, iloc_glob] = np.sum( delta**2 ) / ( n_VGP - 1 )
                scatter = np.sqrt( np.sum( delta**2 ) / ( n_VGP - 1 ) )
                lambda_cut = 90. - ( 1.8 * scatter + 5.) 
                mask_lam = (VGP_lat > lambda_cut * np.ones_like(VGP_lat) )
                ncut = n_VGP - np.sum(mask_lam)
                #print('ibin : ', ibin, ' lambda_cut = ', lambda_cut, 'ngood = ', np.sum(mask_lam), 'ncut = ', ncut)
                while ncut > 0 and n_VGP > 1:
                    VGP_lat = VGP_lat[mask_lam]
                    VGP_lon = VGP_lon[mask_lam]
                    VGP_lat_avg = np.mean(VGP_lat)
                    VGP_lon_avg = np.mean(VGP_lon)
                    n_VGP = len(VGP_lat )
                    delta = angular_distance_2sphere( VGP_lat, VGP_lon, \
                                                      VGP_lat_avg * np.ones_like(VGP_lat), VGP_lon_avg * np.ones_like(VGP_lat), Verbose=False)
                    scatter = np.sqrt( np.sum( delta**2 ) / ( n_VGP - 1 ) )
                    lambda_cut = 90. - ( 1.8 * scatter + 5.) 
                    mask_lam = (VGP_lat > lambda_cut * np.ones_like(VGP_lat) )
                    ncut = n_VGP - np.sum(mask_lam)
                    #print('ibin : ', ibin, ' lambda_cut = ', lambda_cut, 'ngood = ', np.sum(mask_lam), 'ncut = ', ncut)
                scatter_squared[idraw, iloc_glob] = np.sum( delta**2 ) / ( n_VGP - 1 )
            """
            if ibin==9:
                print(VGP_lat)
                print(mask_lam)
            """
            # Nuts and bolts of paleomagnetism, Cox & Hart, page 310
            r_north = np.sum(n_north)
            r_east = np.sum(n_east)
            r_down = np.sum(n_down)
            inc_avg = np.arctan( r_down / np.sqrt( r_north**2 + r_east**2 ) ) 
            inc_anom[idraw, ibin] = np.rad2deg(inc_avg - Inc_GAD)
            #print(ibin, len(n_north),  np.mean(my_latitude_in_deg), inc_anom[idraw,ibin] )
    #print(np.shape(VDM))
    #VDM = np.array(VDM, dtype=float)
    Vmed = np.median(VDM, axis=None)
    VDM75, VDM25 = np.percentile(VDM, [75 ,25])
    Viqr = VDM75 - VDM25
    Vpercent[idraw] = Viqr / Vmed

#mask_bin = np.logical_not(empty_bin)
#my_lat = bin_lat[mask_bin]
a = np.zeros( ndraw, dtype=float)
b = np.zeros( ndraw, dtype=float)
for idraw in range(mydraw_beg, mydraw_end):
    my_scatter_squared = scatter_squared[idraw,:]
    mask_test = np.isfinite(my_scatter_squared)
    my_scatter_squared = my_scatter_squared[mask_test]
    my_latitude_in_deg = datPSV10.latitude_in_deg[mask_test]
    popt, pcov = curve_fit( quadratic_disp, my_latitude_in_deg, my_scatter_squared)
    a[idraw] = np.abs(popt[0])
    b[idraw] = np.abs(popt[1])
#
#Global gather if draw done in parallel
#Vpercent
if size>1:
    Vpercent = comm.allreduce(Vpercent, op=MPI.SUM)
    a = comm.allreduce(a, op=MPI.SUM)
    b = comm.allreduce(b, op=MPI.SUM)
    for ibin in range(len(bins)-1):
        inc_anom[:,ibin] = comm.allreduce(inc_anom[:,ibin], op=MPI.SUM)
#inspection of arrays
Vpercent = Vpercent[np.isfinite(Vpercent)]
a = a[np.isfinite(a)]
b = b[np.isfinite(b)]

QPMsimu.Vpercent_med = np.median( Vpercent)
QPMsimu.Vpercent_low = np.percentile( Vpercent, 2.5)
QPMsimu.Vpercent_high = np.percentile( Vpercent, 97.5)
if rank ==0: 
    print()
    print(' Vpercent med low high  = ', QPMsimu.Vpercent_med, QPMsimu.Vpercent_low, QPMsimu.Vpercent_high)
    print()
"""
plt.hist(VDM)
plt.show()
"""

#mask_bin = np.logical_not(empty_bin)
#my_lat = bin_lat[mask_bin]
"""
a = np.zeros( ndraw, dtype=float)
b = np.zeros( ndraw, dtype=float)
for idraw in range(ndraw):
    my_scatter_squared = scatter_squared[idraw,:]
    popt, pcov = curve_fit( quadratic_disp, datPSV10.latitude_in_deg, my_scatter_squared)
    a[idraw] = np.sqrt(popt[0])
    b[idraw] = np.sqrt(popt[1])
"""

QPMsimu.a_med = np.median(a)
QPMsimu.a_low = np.percentile(a, 2.5)
QPMsimu.a_high = np.percentile(a, 97.5)
QPMsimu.b_med = np.median(b)
QPMsimu.b_low = np.percentile(b, 2.5)
QPMsimu.b_high = np.percentile(b, 97.5)
if rank ==0: 
    print()
    print(' a med low high  = ', QPMsimu.a_med, QPMsimu.a_low, QPMsimu.a_high)
    print(' b med low high  = ', QPMsimu.b_med, QPMsimu.b_low, QPMsimu.b_high)
    print()

"""
scatter_median = np.zeros(nbins, dtype=float)
scatter_low = np.zeros(nbins, dtype=float)
scatter_high = np.zeros(nbins, dtype=float)
for ibin in range(len(bins)-1):
    this_scatter = sorted( scatter_squared[:,ibin] )
    if not empty_bin[ibin]:
        scatter_median[ibin] = np.median(this_scatter)
        scatter_low[ibin] = this_scatter[5*int(ndraw/100)-1]
        scatter_high[ibin] = this_scatter[95*int(ndraw/100)-1]
        print( bin_lat[ibin], np.sqrt(scatter_median[ibin]), np.sqrt(scatter_low[ibin]), np.sqrt(scatter_high[ibin]))
"""

inc_anom_median = np.zeros(nbins, dtype=float)
inc_anom_low = np.zeros(nbins, dtype=float)
inc_anom_high = np.zeros(nbins, dtype=float)
for ibin in range(len(bins)-1):
    if not empty_bin[ibin]:
        inc_anom_median[ibin] = np.median(     inc_anom[:,ibin])
        inc_anom_low[ibin]    = np.percentile( inc_anom[:,ibin], 2.5)
        inc_anom_high[ibin]   = np.percentile( inc_anom[:, ibin], 97.5)
        if rank ==0: 
            print( "%12.3f %12.3f %12.3f %12.3f " % (bin_lat[ibin], inc_anom_median[ibin], inc_anom_low[ibin], inc_anom_high[ibin]) )

inc_ind_max = np.argmax(np.abs(inc_anom_median))

QPMsimu.delta_Inc_med = inc_anom_median[inc_ind_max]
QPMsimu.delta_Inc_low = inc_anom_low[inc_ind_max]
QPMsimu.delta_Inc_high = inc_anom_high[inc_ind_max]
if rank ==0: 
    print()
    print('QPMsimu.delta_Inc_med = %10.2f QPMsimu.delta_Inc_low = %10.2f  QPMsimu.delta_Inc_high = %10.2f '\
          % (QPMsimu.delta_Inc_med, QPMsimu.delta_Inc_low, QPMsimu.delta_Inc_high))

Verbose = False
if rank == 0: 
    Verbose = True
compute_Delta_QPM(QPMsimu, QPMearth, Verbose=Verbose)
