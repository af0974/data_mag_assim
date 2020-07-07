import sys
import configparser
import numpy as np
import shtns
import matplotlib.pyplot as plt
import rev_process as revpro
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpi4py import MPI

def get_rescaling_factors(comm, size, rank, config_file):

    if rank == 0:
        print()
        print('  Determining rescaling factors for time and magnetic field strength ')
        print()

# initialize parameters
    config_file = config_file
    config = configparser.ConfigParser()
    config.read(config_file)
    fname = config['Common']['filename']
    tag = config['Common']['tag']
    Verbose = config['Common'].getboolean('Verbose')
    dump_spectra = config['Rescaling'].getboolean('dump_spectra')
    plot_tausv = config['Rescaling'].getboolean('plot_tausv')
    percent = int(config['Rescaling']['percent_scales'])

    if rank == 0: 
        if Verbose is True: 
            print('    mpi parallel size is ', size)

    if Verbose is True and rank==0: 
        print('    filename is ', fname)
    raw = np.fromfile(fname, dtype=np.float64)
    ltrunc = 13
    sh = shtns.sht(13)
    sh_schmidt = shtns.sht(13, norm=shtns.sht_fourpi | shtns.SHT_REAL_NORM)
    nlm = shtns.nlm_calc(13,13,1)
    raw = raw.reshape((-1,2*nlm+1))
    raw.shape
#
    if Verbose is True and rank==0:
        print('    considering the first ', percent,' percent of data to establish rescaling factors')
    tag = tag+'_'+str(percent)+'%'
#
    twork = raw[:,0]
    end = int( percent * len(twork) / 100 )
    t = raw[:end,0]
    keep = revpro.clean_series(t, Verbose=Verbose, myrank=rank)
    t = t[keep]
    br_lm = (raw[:end,1::2] + 1j*raw[:end,2::2])*sh.l*(sh.l+1)   # multiply by l(l+1)
    br_lm = br_lm[keep,:]
    sh.set_grid(nlat=48, nphi=96)#, flags=shtns.sht_reg_poles)
    sh_schmidt.set_grid(nlat=48, nphi=96)#, flags=shtns.sht_reg_poles)
    if rank == 0 and Verbose is True:
        print('    total number of samples = ', len(t))
#nsamp = int( len(t)/1000)
    nsamp = len(t)
    time = np.zeros( nsamp )
    g10 = np.zeros( nsamp )
    sp_b = np.zeros( (ltrunc+1, nsamp) )
    sp_bdot = np.zeros( (ltrunc+1, nsamp) )
    tau_l = np.zeros( (ltrunc+1, nsamp) )
    tau_sv_avg = np.zeros( ltrunc+1 )
    mask = np.zeros(nsamp, dtype=bool)
    mask[:] = False
#one_percent = nsamp/100
    t_max = -0.1
#
#mpi 1D domain decomposition
    nsamp_per_process = int( nsamp / size)
    mysamp_beg = rank * nsamp_per_process
    mysamp_end = mysamp_beg + nsamp_per_process
    if (rank == size-1):
        mysamp_end = nsamp
    if size >1:
        comm.Barrier()
    if Verbose is True:
        if rank == 0: 
            print('    1D domain decomposition for processing:', flush=True)
    if Verbose is True:
       print('        beg end ', mysamp_beg, mysamp_end, ' for process ', rank)
###
#
    for i in range(mysamp_beg,mysamp_end):
        br = sh.synth(br_lm[i,:])
        br_lm_schmidt = sh_schmidt.analys(br)
        glm, hlm, ghlm = revpro.compute_glmhlm_from_brlm( br_lm_schmidt, sh_schmidt, ltrunc = None, bscale = None)
        time[i] = t[i]
        deltat = t[i] - t[i-1]
        test = (deltat > 0. and (i > mysamp_beg) )
        if test is True:
            mask[i] = True
            glmdot = ( glm - glm_old ) / deltat
            hlmdot = ( hlm - hlm_old ) / deltat
            for il in range(1, 14):
                for im in range(0, il+1):
                    sp_b[ il, i] = sp_b[ il, i] + (il+1) * ( glm[ il, im]**2 + hlm[il, im]**2 )
                    sp_bdot[ il, i] = sp_bdot[ il, i] + (il+1) * ( glmdot[ il, im]**2 + hlmdot[il, im]**2 )
                tau_l[il, i] = np.sqrt ( sp_b[ il, i] / sp_bdot[ il, i] )
        glm_old = glm
        hlm_old = hlm
        g10[i] = glm[1,0]

    if size>1:
        g10 = comm.allreduce(g10, op=MPI.SUM)
        time = comm.allreduce(time, op=MPI.SUM)
        tau_l = comm.allreduce(tau_l, op=MPI.SUM)
        mask = comm.allreduce(mask, op=MPI.SUM)
        sp_b = comm.allreduce(sp_b, op=MPI.SUM)
        sp_bdot = comm.allreduce(sp_bdot, op=MPI.SUM)

    if rank == 0: 
        my_g10 = g10[mask]
        my_time = time[mask]-time[0] # start at t=0. 
        my_tau_l = tau_l[:,mask]
        my_sp_b = sp_b[:,mask]
        my_sp_bdot = sp_bdot[:,mask]
        if dump_spectra is True: 
           np.savez_compressed('extended_spectra_unprocessed_'+tag,  sp_b = my_sp_b, sp_bdot = my_sp_bdot, tau_l = my_tau_l)
        for il in range(1,ltrunc+1):
            tau_sv_avg[ il] = np.sqrt( np.average(my_sp_b[:,:], axis =1)[il] / np.average(my_sp_bdot[:,:], axis =1)[il] )

        def one_over_l(x,tau_sv):

            return tau_sv / x

# fit with a 1 / ell law for tau_ell
        popt, pcov = curve_fit(one_over_l, np.arange(2, 14), tau_sv_avg[2:14])
        if Verbose is True: 
            print('    secular variation time scales ')
            print('       {}      {}    {}'.format('SH degree', 'tau_l', 'tau_sv / l') ) 
            for il in range(2,14):
                print('           {:2d}    {:>10f}     {:>10f}'.format(il, tau_sv_avg[il], float(one_over_l(il, popt))) ) 
        scaling_factor_time = float( 415. / popt)
        if Verbose is True: 
            print( '    Scaling factors: ')
            print( '        time conversion factor = ', scaling_factor_time)

        if plot_tausv is True: 
            plt.scatter( range(2, 13), tau_sv_avg[2:13], marker='s', color='r', label=r'average $\tau_\ell$ ')
            plt.plot( range(2, 13), one_over_l( range(2,13), popt), lw=2, label='(1/$\ell$) fit')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('spherical harmonic degree $\ell$')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig('sv_timescale'+tag+'.pdf')

        g10_mean = np.mean(abs(g10))
        vadm_earth = 7.46*1.e22
        r_earth = 6371.2e3
        mu0 = 4. * np.pi * 1.e-7
        g10_mean_earth = vadm_earth * mu0 / (4. *np.pi * r_earth**3)
        scaling_factor_mag = g10_mean_earth * 1.e3 / g10_mean # in mT
        if Verbose is True: 
            print( '        magnetic field conversion factor (to obtain mT) = ', scaling_factor_mag)
            print( '        time average abs(g10) = {:>3f} nT'.format( scaling_factor_mag * 1e6 * g10_mean) )
        np.savez('conversion_factors_'+tag, scaling_factor_time = scaling_factor_time, scaling_factor_mag = scaling_factor_mag) 
        config.add_section('Rescaling factors and units')
        config.set('Rescaling factors and units', 'scaling_factor_mag', str(scaling_factor_mag))
        config.set('Rescaling factors and units', 'mag unit', 'mT')
        config.set('Rescaling factors and units', 'scaling_factor_time', str(scaling_factor_time))
        config.set('Rescaling factors and units', 'time unit', 'yr')
        config.set('Common', 'rescaling_done', 'True')
        config_file = open(config_file, 'w')
        config.write(config_file)

def make_gauss_history(comm, size, rank, config_file):

    if rank == 0:
        print()
        print('  Constructing history of Gauss coefficients ')
        print()

# initialize parameters
    config_file = config_file
    config = configparser.ConfigParser()
    config.read(config_file)
    fname = config['Common']['filename']
    tag = config['Common']['tag']
    Verbose = config['Common'].getboolean('Verbose')
    nskip_analysis = int(config['Common']['nskip_analysis'])
    scaling_factor_time = float(config['Rescaling factors and units']['scaling_factor_time'])
    scaling_factor_mag = float(config['Rescaling factors and units']['scaling_factor_mag'])
    mag_unit = config['Rescaling factors and units']['mag unit']
    time_unit = config['Rescaling factors and units']['time unit']

    if rank == 0:
        if Verbose is True:
            print('    mpi parallel size is ', size)

    if Verbose is True: 
        if rank == 0: 
            print('        scaling_factor_time', scaling_factor_time)
            print('        scaling_factor_mag', scaling_factor_mag)

    raw = np.fromfile(fname, dtype=np.float64)
    ltrunc = 13
    sh = shtns.sht(13)
    sh_schmidt = shtns.sht(13, norm=shtns.sht_fourpi | shtns.SHT_REAL_NORM)
    nlm = shtns.nlm_calc(13,13,1)
    raw = raw.reshape((-1,2*nlm+1))
    raw.shape
#
    nskip = nskip_analysis
#
    t = raw[::nskip,0]
    keep = revpro.clean_series(t, Verbose=Verbose, myrank=rank)
    t = t[keep]
    br_lm = (raw[::nskip,1::2] + 1j*raw[::nskip,2::2])*sh.l*(sh.l+1)   # multiply by l(l+1)
    br_lm = br_lm[keep,:]
    sh.set_grid(nlat=48, nphi=96)#, flags=shtns.sht_reg_poles)
    sh_schmidt.set_grid(nlat=48, nphi=96)#, flags=shtns.sht_reg_poles)

    if rank == 0 and Verbose is True:
        print('    total number of samples = ', len(t))

    nsamp = len(t)
    t = t * scaling_factor_time
    br_lm = br_lm * scaling_factor_mag

    glm = np.zeros( (nsamp, ltrunc+1, ltrunc+1) )
    hlm = np.zeros( (nsamp, ltrunc+1, ltrunc+1) )
    ghlm = np.zeros( (nsamp, ltrunc *(ltrunc+2) ) )
    time = np.zeros( nsamp )
    mask = np.zeros(nsamp, dtype=bool)
    mask[:] = False
#
#mpi 1D domain decomposition
    nsamp_per_process = int(nsamp / size)
    mysamp_beg = rank * nsamp_per_process
    mysamp_end = mysamp_beg + nsamp_per_process
    if (rank == size-1):
        mysamp_end = nsamp
    if size > 1:
        comm.Barrier()
    if Verbose is True:
        if rank == 0:
            print('    1D domain decomposition for processing:', flush=True)
    if Verbose is True:
       print('        beg end ', mysamp_beg, mysamp_end, ' for process ', rank, flush=True)

###
    for i in range(mysamp_beg,mysamp_end):
        br = sh.synth(br_lm[i,:])
        br_lm_schmidt = sh_schmidt.analys(br)
        if mag_unit == 'mT':
            bscale = 1.e6
            gauss_unit = 'nT'
        glm[ i, :, :], hlm[ i, :, :], ghlm[ i, :] = revpro.compute_glmhlm_from_brlm( br_lm_schmidt, sh_schmidt, ltrunc = ltrunc, bscale = bscale)
        time[i] = t[i]
        deltat = t[i] - t[i-1]
        test = (deltat > 0. and (i > mysamp_beg) )
        if test == True:
             mask[i] = True

    if size>1:
        glm = comm.allreduce(glm, op=MPI.SUM)
        hlm = comm.allreduce(hlm, op=MPI.SUM)
        ghlm = comm.allreduce(ghlm, op=MPI.SUM)
        mask = comm.allreduce(mask, op=MPI.SUM)
        time = comm.allreduce(time, op=MPI.SUM)

    if rank == 0:
        my_glm = glm[mask, :, :]
        my_hlm = hlm[mask, :, :]
        my_ghlm = ghlm[mask, :]
        my_time = time[mask]-time[0] # start at t=0.
        gauss_fname = 't_gauss_nskip%i_'%nskip+tag
        np.savez(gauss_fname, time = my_time, glm = my_glm, hlm = my_hlm, ghlm = my_ghlm)
        config.add_section('Gauss coefficients')
        config.set('Gauss coefficients', 'ltrunc', str(ltrunc) )
        config.set('Gauss coefficients', 'unit', gauss_unit)
        config.set('Gauss coefficients', 'filename', gauss_fname+'.npz')
        config.set('Common', 'gauss_done', 'True')
        config_file = open(config_file, 'w')
        config.write(config_file)
#
def prepare_SHB_plot( comm, size, rank, config_file):

    if rank == 0:
        print()
        print('  building design matrices on regular grid ')
        print()

    config_file = config_file
    config = configparser.ConfigParser()
    config.read(config_file)
    Verbose = config['Common'].getboolean('Verbose')

    l_trunc = 10
    sh = shtns.sht(l_trunc)
    sh.set_grid(nlat=48, nphi=96)
    theta = np.arccos( sh.cos_theta )
    phi = np.arange(0,sh.nphi) * 2 * np.pi / sh.nphi

    ntheta = len(theta)
    nphi = len(phi)
    npt = ntheta * nphi 

    colatitude_in_deg = np.zeros( npt, dtype = float )
    longitude_in_deg = np.zeros( npt, dtype = float )
    ipt = -1
    for itheta in range(ntheta):
        for iphi in range(nphi):
            ipt = ipt + 1
            colatitude_in_deg[ ipt] = np.rad2deg(theta[itheta])
            longitude_in_deg[ ipt] = np.rad2deg(phi[iphi])

    radius = 6371.2 * np.ones_like(colatitude_in_deg)

    nsh = l_trunc*(l_trunc+2)
    SHBX = np.zeros( (npt, nsh ), dtype=float)
    SHBY = np.zeros( (npt, nsh ), dtype=float)
    SHBZ = np.zeros( (npt, nsh ), dtype=float)

    npt_per_process = int(npt / size)
    mypt_beg = rank * npt_per_process
    mypt_end = mypt_beg + npt_per_process
    if (rank == size-1):
        mypt_end = npt
    if size > 1:
        comm.Barrier()
    if Verbose is True:
        if rank == 0:
            print('    1D domain decomposition for processing:', flush=True)
    if Verbose is True:
       print('        beg end ', mypt_beg, mypt_end, ' for process ', rank, flush=True)

    for ipt in range( mypt_beg, mypt_end): 
        SHBX[ipt,:] = revpro.SHB_X(colatitude_in_deg[ipt], longitude_in_deg[ipt], radius[ipt], ll=l_trunc )
        SHBY[ipt,:] = revpro.SHB_Y(colatitude_in_deg[ipt], longitude_in_deg[ipt], radius[ipt], ll=l_trunc )
        SHBZ[ipt,:] = revpro.SHB_Z(colatitude_in_deg[ipt], longitude_in_deg[ipt], radius[ipt], ll=l_trunc )

    if size>1:
        SHBX = comm.allreduce(SHBX, op=MPI.SUM)
        SHBY = comm.allreduce(SHBY, op=MPI.SUM)
        SHBZ = comm.allreduce(SHBZ, op=MPI.SUM)

    if rank == 0: 
        filename = 'SHB_nlat%i_nlon%i'%(ntheta,nphi)
        np.savez(filename, SHBX = SHBX, SHBY = SHBY, SHBZ = SHBZ, l_trunc=l_trunc, npt=npt, theta=theta,phi=phi)
        config.add_section('Design matrices on regular grid')
        config.set('Design matrices on regular grid', 'l_trunc', str(l_trunc) )
        config.set('Design matrices on regular grid', 'nlat', str(ntheta) )
        config.set('Design matrices on regular grid', 'nlon', str(nphi) )
        config.set('Design matrices on regular grid', 'npt', str(npt) )
        config.set('Design matrices on regular grid', 'filename', filename+'.npz')
        config.set('Common', 'SHB_plot_done', 'True')
        config_file = open(config_file, 'w')
        config.write(config_file)

def get_rms_intensity( comm, size, rank, config_file):

    if rank == 0:
        print()
        print('  Analysis of geomagnetic intensity ')
        print()

    config_file = config_file
    config = configparser.ConfigParser()
    config.read(config_file)
    Verbose = config['Common'].getboolean('Verbose')
    fname_gauss = config['Gauss coefficients']['filename']
    gauss_unit = config['Gauss coefficients']['unit']
    ltrunc_gauss = int(config['Gauss coefficients']['ltrunc'])
    tag = config['Common']['tag']
    time_unit = config['Rescaling factors and units']['time unit']
    ltrunc_SHB = int(config['Design matrices on regular grid']['l_trunc'])
    nlat = int(config['Design matrices on regular grid']['nlat'])
    nlon = int(config['Design matrices on regular grid']['nlon'])
    fname_SHB = config['Design matrices on regular grid']['filename']

    npzfile =  np.load(fname_gauss)
    time = npzfile['time']
    ghlm = npzfile['ghlm']

    npzfile_SHB = np.load(fname_SHB)
    SHBX = npzfile_SHB['SHBX']
    SHBY = npzfile_SHB['SHBY']
    SHBZ = npzfile_SHB['SHBZ']
    sh = shtns.sht(ltrunc_SHB, ltrunc_SHB, norm=shtns.sht_fourpi | shtns.SHT_NO_CS_PHASE | shtns.SHT_REAL_NORM)
    sh.set_grid(nlat=nlat, nphi=nlon)
    theta = np.arccos( sh.cos_theta )
    phi = np.arange(0,sh.nphi) * 2 * np.pi / sh.nphi
    nsh = ltrunc_SHB * ( ltrunc_SHB + 2 ) 
    
    ntime = int ( len(time) ) # / 5 )
    ntime_per_process = int(ntime / size)
    mytime_beg = rank * ntime_per_process
    mytime_end = mytime_beg + ntime_per_process
    F_rms = np.zeros(ntime, dtype=float)
    if (rank == size-1):
        mytime_end = ntime
    if size > 1:
        comm.Barrier()
    if Verbose is True:
        if rank == 0:
            print('    1D domain decomposition for processing:', flush=True)
    if Verbose is True:
       print('        beg end ', mytime_beg, mytime_end, ' for process ', rank, flush=True)
    for itime in range(mytime_beg, mytime_end):
        X = np.dot(SHBX, ghlm[itime,0:nsh])
        Y = np.dot(SHBY, ghlm[itime,0:nsh])
        Z = np.dot(SHBZ, ghlm[itime,0:nsh])
        F = np.sqrt( X**2 + Y**2 + Z**2 )
        F = np.reshape( F, (nlat,nlon) )
        F_lm = sh.analys(F)
        F_rms[itime] = np.sqrt( np.sum( (np.abs(F_lm))**2 ) )

    if size > 1:
        F_rms = comm.allreduce(F_rms, op=MPI.SUM)
#
    return time, F_rms, gauss_unit, time_unit    

#
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
config_file = "config_revproc.ini"
config = configparser.ConfigParser()
config.read(config_file)
ier = comm.Barrier()
rescaling_done = config['Common'].getboolean('rescaling_done')
if rescaling_done is not True:
    get_rescaling_factors( comm, size, rank, config_file)
ier = comm.Barrier()
gauss_done = config['Common'].getboolean('gauss_done')
if gauss_done is not True:
    make_gauss_history( comm, size, rank, config_file)
ier = comm.Barrier()
SHB_plot_done = config['Common'].getboolean('SHB_plot_done')
if SHB_plot_done is not True:
    prepare_SHB_plot( comm, size, rank, config_file)
ier = comm.Barrier()
if config['Diags'].getboolean('rms_intensity') is True:
    time, F_rms, gauss_unit, time_unit = get_rms_intensity( comm, size, rank, config_file)
    if rank==0:
        np.savez('F_rms', time=time, F_rms=F_rms, gauss_unit = gauss_unit, time_unit = time_unit)
