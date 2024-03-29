import sys
import configparser
import numpy as np
import shtns
import matplotlib.pyplot as plt
import rev_process as revpro
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpi4py import MPI

def compute_chi2_components(comm, size, rank, config_file):

    if rank == 0:
        print()
        print('  Computing Earth-likeness components according to Christensen et al. (EPSL, 2010) ')
        print()
	
    config_file = config_file
    config = configparser.ConfigParser(interpolation=None)
    config.read(config_file)
    Verbose = config['Common'].getboolean('Verbose')
    fname_gauss = config['Gauss coefficients']['filename']
    gauss_unit = config['Gauss coefficients']['unit']
    outdir = config['Common']['output_directory']
    ltrunc_gauss = int(config['Gauss coefficients']['ltrunc'])
    tag = config['Common']['tag']
    time_unit = config['Rescaling factors and units']['time unit']

    npzfile =  np.load(outdir+'/'+fname_gauss)
    time = npzfile['time']
    glm = npzfile['glm']
    hlm = npzfile['hlm'] 

    a = 6371.2
    c = 3485.
    ltrunc = 8
    sh_par = shtns.sht(9, 9, norm=shtns.sht_fourpi | shtns.SHT_NO_CS_PHASE | shtns.SHT_REAL_NORM)
    sh_par.set_grid(nlat=48, nphi=96)
    ad_over_nad = np.zeros_like(time, dtype=float)
    O_over_E = np.zeros_like(time, dtype=float)
    z_over_nz = np.zeros_like(time, dtype=float)
    fcf = np.zeros_like(time, dtype=float)
#
# mpi 1D domain decomposition
    nsamp = len(time)
    nsamp_per_process = int(nsamp / size)
    mysamp_beg = rank * nsamp_per_process
    mysamp_end = mysamp_beg + nsamp_per_process
    if (rank == size-1):
        mysamp_end = nsamp
    if size > 1:
        ier = comm.Barrier()
    if Verbose is True:
        if rank == 0:
            print('    1D domain decomposition for processing:', flush=True)
    if Verbose is True:
       print('        beg end ', mysamp_beg, mysamp_end, ' for process ', rank, flush=True)

    #for itime in range(len(time)):
    for itime in range(mysamp_beg, mysamp_end):
    	if np.mod(itime+1-mysamp_beg, int(nsamp_per_process/10)) == 0 and Verbose is True:
             if rank==0:
                 print('        rank ', rank, ' performed ', itime+1-mysamp_beg, ' analyses', flush=True)
	# AD/NAD ratio
    	nad = 0. 
    	nad = nad + 2.*(glm[itime,1,1]**2 + hlm[itime,1,1]**2)
        for il in range(2,ltrunc+1):
    		toto = 0.
    		for im in range(0,il+1):
    			toto = toto + (glm[itime,il,im]**2+hlm[itime,il,im]**2)
    		nad = nad + (il+1.) * (a/c)**(2.*il-2.)*toto
    	ad_over_nad[itime] = (2.* glm[itime, 1,0]**2 ) / nad

	# Equatorial symmetry
    	odd = 0.
    	even = 0. 
    	for il in range(2,ltrunc+1):
    		for im in range(0,il+1):
    			if ( (il+im) % 2 == 0 ):
    				even = even + (a/c)**(2*il-2) * (il+1.) * (glm[itime,il,im]**2+hlm[itime,il,im]**2)
    			else:
    				odd  = odd + (a/c)**(2*il-2) * (il+1.) * (glm[itime,il,im]**2+hlm[itime,il,im]**2)
    	O_over_E[itime] = odd / even
	# Zonality
    	zonal = 0.
    	nonzonal = 0.
    	for il in range(2,ltrunc+1):
    		zonal = zonal + (a/c)**(2*il-2) * (il+1.) * (glm[itime,il,0]**2+hlm[itime,il,0]**2)
    		#print(zonal, glm[itime,il,0], hlm[itime,il,0])
    		for im in range(1,il+1):
    			nonzonal = nonzonal + (a/c)**(2*il-2) * (il+1.) * (glm[itime,il,im]**2+hlm[itime,il,im]**2)
    			#print(nonzonal, glm[itime,il,im], hlm[itime,il,im])
    	z_over_nz[itime] = zonal / nonzonal
	# Flux concentration factor
	# need br_lm_trunc at degree 8
    	br_lm_trunc = revpro.compute_brlm_from_glmhlm(glm[itime,:,:], hlm[itime,:,:], sh_par, ltrunc = ltrunc, bscale =None, radius=None)
    	br2_rms = np.sum( (np.abs(br_lm_trunc))**2 )
    	brf = sh_par.synth(br_lm_trunc)
    	brf2 = brf*brf
    	br2_lm = sh_par.analys(brf2)
    	br4_rms = np.sum( (np.abs(br2_lm))**2 )
    	fcf[itime] = (br4_rms - br2_rms**2) / br2_rms**2   

    if size>1:
        ad_over_nad = comm.allreduce( ad_over_nad, op=MPI.SUM)
        O_over_E = comm.allreduce( O_over_E, op=MPI.SUM)
        z_over_nz = comm.allreduce( z_over_nz, op=MPI.SUM)
        fcf = comm.allreduce( fcf, op=MPI.SUM)
    # safety net with z_over_nz that can go inf during transients
    mask = ~np.isinf(z_over_nz)
    ad_over_nad = ad_over_nad[mask]
    O_over_E = O_over_E[mask]
    z_over_nz = z_over_nz[mask]
    fcf = fcf[mask]

    return ad_over_nad, O_over_E, z_over_nz, fcf

def compute_chi2(comm, size, rank, config_file):

    ad_over_nad, O_over_E, z_over_nz, fcf = compute_chi2_components(comm, size, rank, config_file)
#   Earth values are there
    ad_over_nad_earth = 1.4
    O_over_E_earth = 1.
    z_over_nz_earth = 0.15
    fcf_earth = 1.5
    static_chi2 = ( np.log(np.mean(ad_over_nad)/ad_over_nad_earth) / np.log(2.0) ) **2 \
                 +( np.log(np.mean(O_over_E)/O_over_E_earth) / np.log(2.0) ) **2 \
                 +( np.log(np.mean(z_over_nz)/z_over_nz_earth) / np.log(2.5) ) **2 \
                 +( np.log(np.mean(fcf)/fcf_earth) / np.log(1.75) ) **2
    all_chi2 =    ( np.log(ad_over_nad/ad_over_nad_earth) / np.log(2.0) ) **2 \
                 +( np.log(O_over_E/O_over_E_earth) / np.log(2.0) ) **2 \
                 +( np.log(z_over_nz/z_over_nz_earth) / np.log(2.5) ) **2 \
                 +( np.log(fcf/fcf_earth) / np.log(1.75) ) **2
    if rank == 0:
        print()
        print('			AD_OVER_NAD = ', np.mean(ad_over_nad))
        print('			O_OVER_E = ', np.mean(O_over_E))
        print('			Z_OVER_NZ = ', np.mean(z_over_nz))
        print('			Flux Conc. Fact. = ', np.mean(fcf))
        print()
        print('  	 chi2 = ', static_chi2)
        print(' 	 median  chi2 over time = ', np.median(all_chi2))

    return static_chi2

def get_rescaling_factors(comm, size, rank, config_file):

    if rank == 0:
        print()
        print('  Determining rescaling factors for time and magnetic field strength ')
        print()

# initialize parameters
    config_file = config_file
    config = configparser.ConfigParser(interpolation=None)
    config.read(config_file)
    dynamo_code = config['Common']['dynamo_code']
    outdir = config['Common']['output_directory']
    tag = config['Common']['tag']
    Verbose = config['Common'].getboolean('Verbose')
    dump_spectra = config['Rescaling'].getboolean('dump_spectra')
    plot_tausv = config['Rescaling'].getboolean('plot_tausv')

    ltrunc = 13

    if rank == 0: 
        if Verbose is True: 
            print('    mpi parallel size is ', size)
            print('    dynamo code is ', dynamo_code)

    if dynamo_code == "xshells": 
        percent = int(config['Rescaling']['percent_scales'])
        fname = config['Common']['filename'] # in case xshells is post-processed

        if Verbose is True and rank==0: 
            print('    filename is ', fname)
        raw = np.fromfile(fname, dtype=np.float64)
        sh = shtns.sht(13)
        sh_schmidt = shtns.sht(13, norm=shtns.sht_fourpi | shtns.SHT_REAL_NORM)
        nlm = shtns.nlm_calc(13,13,1)
        raw = raw.reshape((-1,2*nlm+1))
        raw.shape
#
        if Verbose is True and rank==0:
            print('    considering the first ', percent,' percent of data to establish rescaling factors', flush=True)
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
# nsamp = int( len(t)/1000)
        nsamp = len(t)
        time = np.zeros( nsamp )
        g10 = np.zeros( nsamp )
        sp_b = np.zeros( (nsamp, ltrunc+1) )
        sp_bdot = np.zeros( (nsamp, ltrunc+1) )
        tau_l = np.zeros( (nsamp, ltrunc+1) )
        tau_sv_avg = np.zeros( ltrunc+1 )
        mask = np.zeros(nsamp, dtype=bool)
        mask[:] = False
# one_percent = nsamp/100
        t_max = -0.1
#
# mpi 1D domain decomposition
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
           print('        beg end ', mysamp_beg, mysamp_end, ' for process ', rank, flush=True)
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
                        sp_b[ i, il] = sp_b[ i, il] + (il+1) * ( glm[ il, im]**2 + hlm[il, im]**2 )
                        sp_bdot[ i,  il] = sp_bdot[ i, il] + (il+1) * ( glmdot[ il, im]**2 + hlmdot[il, im]**2 )
                    tau_l[i, il] = np.sqrt ( sp_b[ i, il] / sp_bdot[ i, il] )
            glm_old = glm
            hlm_old = hlm
            g10[i] = glm[1,0]
#
    elif dynamo_code == "parody":
#
        ltrunc = 12
        import parody_toolbox_wf as ptool
        import glob
        path = config['Parody']['data_location']
        fname = config['Parody']['surface_fname']
        files = glob.glob( path + '/' + fname )
        nlat, nphi, time, br, dbrdt = ptool.get_data_from_S_file(files[rank], Verbose=False)
        sh_schmidt = shtns.sht(ltrunc, norm=shtns.sht_fourpi | shtns.SHT_REAL_NORM)
        sh_schmidt.set_grid(nlat=nlat, nphi=nphi)
        #sh = shtns.sht( lmax, mmax, mres, norm=shtns.sht_fourpi | shtns.SHT_REAL_NORM)
        #sh.set_grid(nlat, nphi)
        theta = np.arccos(sh_schmidt.cos_theta)
        phi =  (2. * np.pi / np.float(nphi))*np.arange(nphi) - np.pi
        #revpro.mollweide_surface(br, theta, phi, fname=None, vmax=None, vmin=None, Title=None, positive=False, cmap=None, unit="nondim")
        nsamp = len(files)
        if Verbose is True:
            print('number of surface files', nsamp)
        time = np.zeros( nsamp )
        g10 = np.zeros( nsamp )
        br = np.zeros( (nsamp, nlat, nphi) )
        dbrdt = np.zeros( (nsamp, nlat, nphi) )
        sp_b = np.zeros( (nsamp, ltrunc+1) )
        sp_bdot = np.zeros( (nsamp, ltrunc+1) )
        tau_l = np.zeros( (nsamp, ltrunc+1) )
        tau_sv_avg = np.zeros( ltrunc+1 )
        mask = np.zeros(nsamp, dtype=bool)
        mask[:] = False
        # mpi 1D domain decomposition
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
           print('        beg end ', mysamp_beg, mysamp_end, ' for process ', rank, flush=True)
###
        for i in range(mysamp_beg,mysamp_end):
            nlat, nphi, this_time, this_br, this_dbrdt = ptool.get_data_from_S_file(files[i], Verbose=False)
            time[i] = this_time
            br[i,:,:] = this_br[:,:]
            dbrdt[i,:,:] = this_dbrdt[:,:]
        if size > 1:
            time = comm.allreduce( time, op=MPI.SUM)
            br = comm.allreduce( br, op=MPI.SUM)
            dbrdt = comm.allreduce( dbrdt, op=MPI.SUM)
        time = time[ np.argsort(time) ]
        br = br[ np.argsort(time) ]
        dbrdt = dbrdt[ np.argsort(time) ]

        for i in range( mysamp_beg, mysamp_end):
            br_lm = sh_schmidt.analys( br[i,:,:] )
            glm, hlm, ghlm = revpro.compute_glmhlm_from_brlm(br_lm, sh_schmidt, ltrunc = ltrunc, bscale = None)
            brdot_lm = sh_schmidt.analys( dbrdt[i,:,:] )
            glmdot, hlmdot, ghlmdot = revpro.compute_glmhlm_from_brlm(brdot_lm, sh_schmidt, ltrunc = ltrunc, bscale = None)
            test = True
            if test is True:
                mask[i] = True
                for il in range(1, ltrunc+1):
                    for im in range(0, il+1):
                        sp_b[ i, il] = sp_b[ i, il] + (il+1) * ( glm[ il, im]**2 + hlm[il, im]**2 )
                        sp_bdot[ i,  il] = sp_bdot[ i, il] + (il+1) * ( glmdot[ il, im]**2 + hlmdot[il, im]**2 )
                    tau_l[i, il] = np.sqrt ( sp_b[ i, il] / sp_bdot[ i, il] )
                """    
                plt.semilogy( np.arange(1, 14), sp_b[i, 1:14], label='field')
                plt.semilogy( np.arange(1, 14), sp_bdot[i, 1:14], label='SV')
                plt.semilogy( np.arange(1, 14), tau_l[i, 1:14], label=r'$\tau_\ell$')
                plt.legend()
                plt.show()
                sys.exit()
                """
            g10[i] = glm[1,0]

    if size>1:
        g10 = comm.allreduce(g10, op=MPI.SUM)
        tau_l = comm.allreduce(tau_l, op=MPI.SUM)
        mask = comm.allreduce(mask, op=MPI.SUM)
        sp_b = comm.allreduce(sp_b, op=MPI.SUM)
        sp_bdot = comm.allreduce(sp_bdot, op=MPI.SUM)

    if rank == 0: 
        my_g10 = g10[mask]
        my_time = time[mask]-time[0] # start at t=0. 
        my_tau_l = tau_l[mask,:]
        my_sp_b = sp_b[mask,:]
        my_sp_bdot = sp_bdot[mask,:]
        if dump_spectra is True: 
            fname = 'extended_spectra_unprocessed_'+tag
            np.savez_compressed(outdir+'/'+fname,  sp_b = my_sp_b, sp_bdot = my_sp_bdot, tau_l = my_tau_l)
            fname = fname+'.npz'
            config.set('Diags', 'spectra_file', fname)
            lfile = open(config_file, 'w')
            config.write(lfile)
            lfile.close()
        for il in range(1,ltrunc+1):
            tau_sv_avg[ il] = np.sqrt( np.average(my_sp_b[:,:], axis =0)[il] / np.average(my_sp_bdot[:,:], axis =0)[il] )

        def one_over_l(x,tau_sv):

            return tau_sv / x

# fit with a 1 / ell law for tau_ell
        popt, pcov = curve_fit(one_over_l, np.arange(2, ltrunc+1), tau_sv_avg[2:ltrunc+1])
        if Verbose is True: 
            print('    secular variation time scales ')
            print('       {}      {}    {}'.format('SH degree', 'tau_l', 'tau_sv / l') ) 
            for il in range(2,ltrunc+1):
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
            plt.savefig(outdir+'/'+'sv_timescale_'+tag+'.pdf')
            plt.close()

        g10_mean = np.mean(abs(g10))
        vadm_earth = 7.46*1.e22
        r_earth = 6371.2e3
        mu0 = 4. * np.pi * 1.e-7
        g10_mean_earth = vadm_earth * mu0 / (4. *np.pi * r_earth**3)
        scaling_factor_mag = g10_mean_earth * 1.e3 / g10_mean # in mT
        if Verbose is True: 
            print( '        magnetic field conversion factor (to obtain mT) = ', scaling_factor_mag)
            print( '        time average abs(g10) = {:>3f} nT'.format( scaling_factor_mag * 1e6 * g10_mean) )
        np.savez(outdir+'/'+'conversion_factors_'+tag, scaling_factor_time = scaling_factor_time, scaling_factor_mag = scaling_factor_mag) 


        if config.has_section('Rescaling factors and units') is False: 
	        config.add_section('Rescaling factors and units')
        config.set('Rescaling factors and units', 'scaling_factor_mag', str(scaling_factor_mag))
        config.set('Rescaling factors and units', 'mag unit', 'mT')
        config.set('Rescaling factors and units', 'scaling_factor_time', str(scaling_factor_time))
        config.set('Rescaling factors and units', 'time unit', 'yr')
        config.set('Common', 'rescaling_done', 'True')
        config_file = open(config_file, 'w')
        config.write(config_file)
        config_file.close()

def make_gauss_history(comm, size, rank, config_file):

    if rank == 0:
        print()
        print('  Constructing history of Gauss coefficients ')
        print(flush=True)

# initialize parameters
    config_gauss = configparser.ConfigParser(interpolation=None)
    config_gauss.read(config_file)
    dynamo_code = config_gauss['Common']['dynamo_code']
    fname = config_gauss['Common']['filename']
    tag = config_gauss['Common']['tag']
    outdir = config_gauss['Common']['output_directory']
    Verbose = config_gauss['Common'].getboolean('Verbose')
    nskip_analysis = int(config_gauss['Common']['nskip_analysis'])
    scaling_factor_time = float(config_gauss['Rescaling factors and units']['scaling_factor_time'])
    scaling_factor_mag = float(config_gauss['Rescaling factors and units']['scaling_factor_mag'])
    mag_unit = config_gauss['Rescaling factors and units']['mag unit']
    time_unit = config_gauss['Rescaling factors and units']['time unit']

    if rank == 0:
        if Verbose is True:
            print('    mpi parallel size is ', size)

    if Verbose is True: 
        if rank == 0: 
            print('        scaling_factor_time', scaling_factor_time)
            print('        scaling_factor_mag', scaling_factor_mag)


    if dynamo_code == "xshells": 

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

    elif dynamo_code == "parody":
#
        ltrunc = 13
        import parody_toolbox_wf as ptool
        import glob
        path = config_gauss['Parody']['data_location']
        fname = config_gauss['Parody']['surface_fname']
        files = glob.glob( path + '/' +fname )
        nlat, nphi, time, br, dbrdt = ptool.get_data_from_S_file(files[rank], Verbose=False)
        sh_schmidt = shtns.sht(ltrunc, norm=shtns.sht_fourpi | shtns.SHT_REAL_NORM)
        sh_schmidt.set_grid(nlat=nlat, nphi=nphi)
        theta = np.arccos(sh_schmidt.cos_theta)
        phi =  (2. * np.pi / np.float(nphi))*np.arange(nphi) - np.pi
        nsamp = len(files)
        if Verbose is True:
            print('number of surface files', nsamp)

    glm = np.zeros( (nsamp, ltrunc+1, ltrunc+1) )
    hlm = np.zeros( (nsamp, ltrunc+1, ltrunc+1) )
    ghlm = np.zeros( (nsamp, ltrunc *(ltrunc+2) ) )
    time = np.zeros( nsamp )
    if dynamo_code == "parody":
        br = np.zeros( (nsamp, nlat, nphi), dtype=float)
    mask = np.zeros(nsamp, dtype=bool)
    mask[:] = False
#
# mpi 1D domain decomposition
    nsamp_per_process = int(nsamp / size)
    mysamp_beg = rank * nsamp_per_process
    mysamp_end = mysamp_beg + nsamp_per_process
    if (rank == size-1):
        mysamp_end = nsamp
    if size > 1:
        ier = comm.Barrier()
        samp_beg = np.zeros( size, dtype=int)
        samp_end = np.zeros( size, dtype=int)
        samp_beg[rank] = mysamp_beg
        samp_end[rank] = mysamp_end
        samp_beg = comm.allreduce(samp_beg, op=MPI.SUM)
        samp_end = comm.allreduce(samp_end, op=MPI.SUM)
    if Verbose is True:
        if rank == 0:
            print('    1D domain decomposition for processing:', flush=True)
    if Verbose is True:
       print('        beg end ', mysamp_beg, mysamp_end, ' for process ', rank, flush=True)

###
    if dynamo_code == "xshells": 
        for i in range(mysamp_beg,mysamp_end):
            br = sh.synth(br_lm[i,:])
            br_lm_schmidt = sh_schmidt.analys(br)
            if mag_unit == 'mT':
                bscale = 1.e6
                gauss_unit = 'nT'
            else:
                bscale = None
                gauss_unit = 'ND'
            glm[ i, :, :], hlm[ i, :, :], ghlm[ i, :] = revpro.compute_glmhlm_from_brlm( br_lm_schmidt, sh_schmidt, ltrunc = ltrunc, bscale = bscale)
            time[i] = t[i]
            deltat = t[i] - t[i-1]
            test = (deltat > 0. and (i > mysamp_beg) )
            if test == True:
                 mask[i] = True
    elif dynamo_code == "parody":
        for i in range(mysamp_beg,mysamp_end):
            nlat, nphi, this_time, this_br, this_dbrdt = ptool.get_data_from_S_file(files[i], Verbose=False)
            time[i] = this_time
            br[i,:,:] = this_br[:,:]
        if size > 1:
            time = comm.allreduce( time, op=MPI.SUM)
            br = comm.allreduce( br, op=MPI.SUM)
        time = time[ np.argsort(time) ] * scaling_factor_time
        br = br[ np.argsort(time) ] * scaling_factor_mag

        for i in range( mysamp_beg, mysamp_end):
            br_lm = sh_schmidt.analys( br[i,:,:] )
            mask[i] = True
            if mag_unit == 'mT':
                bscale = 1.e6
                gauss_unit = 'nT'
            else:
                bscale = None
                gauss_unit = 'ND'
            glm[ i, :, :], hlm[ i, :, :], ghlm[ i, :] = revpro.compute_glmhlm_from_brlm(br_lm, sh_schmidt, ltrunc = ltrunc, bscale = bscale)
        
    if size>1:
        ier = comm.Barrier()
        if rank==0:
            for sender in range(1,size):
                data = comm.recv(source=sender, tag=sender)
                print('glm sender=', sender, 'shape=',np.shape(data))
                glm[ samp_beg[sender]:samp_end[sender] ] = data
        else:
            comm.send(glm[mysamp_beg:mysamp_end], dest=0, tag=rank)
        ier = comm.Barrier()
        if rank==0:
            for sender in range(1,size):
                data = comm.recv(source=sender, tag=sender)
                print('hlm sender=', sender, 'shape=',np.shape(data))
                hlm[ samp_beg[sender]:samp_end[sender] ] = data
        else:
            comm.send(hlm[mysamp_beg:mysamp_end], dest=0, tag=rank)
        ier = comm.Barrier()
        if rank==0:
            for sender in range(1,size):
                data = comm.recv(source=sender, tag=sender)
                print('ghlm sender=', sender, 'shape=',np.shape(data))
                ghlm[ samp_beg[sender]:samp_end[sender] ] = data
        else:
            comm.send(ghlm[mysamp_beg:mysamp_end], dest=0, tag=rank)
        ier = comm.Barrier()
        if rank==0:
            for sender in range(1,size):
                data = comm.recv(source=sender, tag=sender)
                print('sender=', sender, 'shape=',np.shape(data))
                mask[ samp_beg[sender]:samp_end[sender] ] = data
        else:
            comm.send(mask[mysamp_beg:mysamp_end], dest=0, tag=rank)
        ier = comm.Barrier()
        if rank==0:
            for sender in range(1,size):
                data = comm.recv(source=sender, tag=sender)
                print('sender=', sender, 'shape=',np.shape(data))
                time[ samp_beg[sender]:samp_end[sender] ] = data
        else:
            comm.send(time[mysamp_beg:mysamp_end], dest=0, tag=rank)

    if rank == 0:
        my_glm = glm[mask, :, :]
        my_hlm = hlm[mask, :, :]
        my_ghlm = ghlm[mask, :]
        my_time = time[mask]-time[0] # start at t=0.
        if dynamo_code == "xshells":
            gauss_fname = 't_gauss_nskip%i_'%nskip+tag
        elif dynamo_code == "parody":
            gauss_fname = 't_gauss_'+tag
        np.savez(outdir+'/'+gauss_fname, time = my_time, glm = my_glm, hlm = my_hlm, ghlm = my_ghlm)
        if config_gauss.has_section('Gauss coefficients') is False:
            config_gauss.add_section('Gauss coefficients')
        config_gauss.set('Gauss coefficients', 'ltrunc', str(ltrunc) )
        config_gauss.set('Gauss coefficients', 'unit', gauss_unit)
        config_gauss.set('Gauss coefficients', 'filename', gauss_fname+'.npz')
        config_gauss.set('Common', 'gauss_done', 'True')
        config_file = open(config_file, 'w')
        config_gauss.write(config_file)
        config_file.close()
#
def prepare_SHB_plot( comm, size, rank, config_file):

    if rank == 0:
        print()
        print('  building design matrices on regular grid ')
        print()

    config_file = config_file
    config = configparser.ConfigParser(interpolation=None)
    config.read(config_file)
    Verbose = config['Common'].getboolean('Verbose')
    outdir = config['Common']['output_directory']

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
        ier = comm.Barrier()
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
        np.savez(outdir+'/'+filename, SHBX = SHBX, SHBY = SHBY, SHBZ = SHBZ, l_trunc=l_trunc, npt=npt, theta=theta,phi=phi)
        if config.has_section('Design matrices on regular grid') is False:
            config.add_section('Design matrices on regular grid')
        config.set('Design matrices on regular grid', 'l_trunc', str(l_trunc) )
        config.set('Design matrices on regular grid', 'nlat', str(ntheta) )
        config.set('Design matrices on regular grid', 'nlon', str(nphi) )
        config.set('Design matrices on regular grid', 'npt', str(npt) )
        config.set('Design matrices on regular grid', 'filename', filename+'.npz')
        config.set('Common', 'SHB_plot_done', 'True')
        config_file = open(config_file, 'w')
        config.write(config_file)

def get_transition_time( comm, size, rank, config_file):

    if rank == 0:
        print()
        print('  Detection of polarity transitions ')
        print()

    config_file = config_file
    config = configparser.ConfigParser(interpolation=None)
    config.read(config_file)
    outdir = config['Common']['output_directory']
    fname = outdir+'/'+config['Diags']['pole_latitude_file']	

    npz = np.load(fname)
    pole_lat = npz['pole_latitude']
    mask_tra = ( np.abs(pole_lat) <= 45. )
    mask_stb = ( np.abs(pole_lat) > 45. )

    return mask_tra, mask_stb

def analyze_transitional_field( comm, size, rank, config_file):

    config_file = config_file
    config = configparser.ConfigParser(interpolation=None)
    config.read(config_file)
    outdir = config['Common']['output_directory']
    Verbose = config['Common'].getboolean('Verbose')
    
    mask_tra, mask_stb = get_transition_time( comm, size, rank, config_file)
    if rank == 0:
        print()
        print('  Spectra of stable and transitional fields ')
        print()
    spectra_file = config['Diags']['spectra_file']
    fname = outdir+'/'+spectra_file
    scaling_factor_mag = float(config['Rescaling factors and units']['scaling_factor_mag'])
    mag_unit = config['Rescaling factors and units']['mag unit']
    if rank == 0:
        npz = np.load(fname)
        sp_b = npz['sp_b']
        if Verbose is True:
            print('   Conformity of arrays   ')
            print('           ',np.shape(sp_b))
            print('           ',np.shape(mask_tra))
        sp_b_tra = sp_b[mask_tra,:] * scaling_factor_mag**2
        sp_b_stb = sp_b[mask_stb,:] * scaling_factor_mag**2
        plt.semilogy(range(1,14), np.mean(sp_b_tra, axis=0)[1:14], 'o', label='transitional')
        plt.semilogy(range(1,14), np.mean(sp_b_stb, axis=0)[1:14], 'x', label='stable' )
        plt.xlabel('SH degree')
        plt.ylabel(r'B$^2$ in '+mag_unit+'$^2$')
        plt.xticks(range(1,14))
        plt.legend(loc='best')
        plt.savefig(outdir+'/'+'sp_a.pdf')
        plt.close()
        a = 6371.2
        c = 3485.0
        for il in range(1,14):
            sp_b_tra[:,il] = (a/c)**(2*il+4) *  sp_b_tra[:,il]
            sp_b_stb[:,il] = (a/c)**(2*il+4) *  sp_b_stb[:,il]
        plt.semilogy(range(1,14), np.mean(sp_b_tra, axis=0)[1:14], 'o', label='transitional')
        plt.semilogy(range(1,14), np.mean(sp_b_stb, axis=0)[1:14], 'x', label='stable' )
        plt.xlabel('SH degree')
        plt.ylabel(r'B$^2$ in '+mag_unit+'$^2$')
        plt.xticks(range(1,14))
        plt.legend(loc='best')
        plt.savefig(outdir+'/'+'sp_c.pdf')
        plt.close()
        for iens in range(np.shape(sp_b_tra)[0]):
                plt.semilogy(range(1,14), sp_b_tra[iens,1:14], color='r',lw=0.1, alpha=0.3)
        for iens in range(np.shape(sp_b_tra)[0]):
                plt.semilogy(range(1,14), sp_b_stb[iens,1:14], color='b' ,lw=0.1, alpha=0.3)
        plt.semilogy(range(1,14), np.mean(sp_b_tra, axis=0)[1:14], 'o', label='transitional', color='r')
        plt.semilogy(range(1,14), np.mean(sp_b_stb, axis=0)[1:14], 'x', label='stable', color='b' )
        plt.xlabel('SH degree')
        plt.ylabel(r'B$^2$ in '+mag_unit+'$^2$')
        plt.xticks(range(1,14))
        plt.legend(loc='best')
        plt.savefig(outdir+'/'+'sp_c_ens.pdf')
        plt.close()
    
    return
	
def get_pole_latitude( comm, size, rank, config_file):

    if rank == 0:
        print()
        print('  Computation of geomagnetic pole latitude ')
        print()

    config_file = config_file
    config = configparser.ConfigParser(interpolation=None)
    config.read(config_file)
    Verbose = config['Common'].getboolean('Verbose')
    fname_gauss = config['Gauss coefficients']['filename']
    gauss_unit = config['Gauss coefficients']['unit']
    outdir = config['Common']['output_directory']
    ltrunc_gauss = int(config['Gauss coefficients']['ltrunc'])
    tag = config['Common']['tag']
    time_unit = config['Rescaling factors and units']['time unit']

    npzfile =  np.load(outdir+'/'+fname_gauss)
    time = npzfile['time']
    ghlm = npzfile['ghlm']

    ntime = int ( len(time) ) # / 5 )
    ntime_per_process = int(ntime / size)
    mytime_beg = rank * ntime_per_process
    mytime_end = mytime_beg + ntime_per_process
    if (rank == size-1):
        mytime_end = ntime
    if size > 1:
        ier = comm.Barrier()
    if Verbose is True:
        if rank == 0:
            print('    1D domain decomposition for processing:', flush=True)
    if Verbose is True:
       print('        beg end ', mytime_beg, mytime_end, ' for process ', rank, flush=True)

    pole_latitude = np.zeros( ntime, dtype=float)

    for itime in range(mytime_beg, mytime_end):
        g10 = ghlm[itime, 0]
        g11 = ghlm[itime, 1]
        h11 = ghlm[itime, 2]
        pole_latitude[itime] = np.rad2deg( np.arctan2( g10, np.sqrt(g11**2+h11**2) )  )

    if size>1:
        pole_latitude = comm.allreduce( pole_latitude, op=MPI.SUM)

    if rank==0:
        fname = 'pole_latitude'
        np.savez(outdir+'/'+fname, time=time, pole_latitude=pole_latitude, time_unit = time_unit)
        fname = fname + '.npz'
        config.set('Diags', 'pole_latitude_file', fname)
        lfile = open(config_file, 'w')
        config.write(lfile)
        lfile.close()

    if rank==0:
        #distibute plot over 3 rows
        nrows = 3
        ndat = len(time)
        nstep = int( ndat / nrows )
        yticks = np.linspace( -90, 90, 5, dtype=float)
        fig, ax = plt.subplots( nrows, 1, figsize=(15,6))
        for i in range(nrows):
            istart = i * nstep
            if i < nrows -1:
                iend = istart + nstep
            else:
                iend = ndat-1
            ax[i].plot( time[istart:iend], pole_latitude[istart:iend])
            ax[i].set_xlim(time[istart],time[iend])
            ax[i].set_ylim(-90,90)
            ax[i].set_yticks(yticks)
            ax[i].set_ylabel('pole latitude (deg)')
        ax[nrows-1].set_xlabel('time in ' + time_unit)
        plt.savefig(outdir+'/pole_latitude.pdf')
        plt.savefig(outdir+'/pole_latitude.png')
        plt.close()
	
    return time, pole_latitude, time_unit

def get_eccentricity( comm, size, rank, config_file):

    if rank == 0:
        print()
        print('  Computation of dipole eccentricity ')
        print()

    config_file = config_file
    config = configparser.ConfigParser(interpolation=None)
    config.read(config_file)
    Verbose = config['Common'].getboolean('Verbose')
    fname_gauss = config['Gauss coefficients']['filename']
    gauss_unit = config['Gauss coefficients']['unit']
    outdir = config['Common']['output_directory']
    ltrunc_gauss = int(config['Gauss coefficients']['ltrunc'])
    tag = config['Common']['tag']
    time_unit = config['Rescaling factors and units']['time unit']

    npzfile =  np.load(outdir+'/'+fname_gauss)
    time = npzfile['time']
    ghlm = npzfile['ghlm']

    ntime = int ( len(time) ) # / 5 )
    ntime_per_process = int(ntime / size)
    mytime_beg = rank * ntime_per_process
    mytime_end = mytime_beg + ntime_per_process
    if (rank == size-1):
        mytime_end = ntime
    if size > 1:
        ier = comm.Barrier()
    if Verbose is True:
        if rank == 0:
            print('    1D domain decomposition for processing:', flush=True)
    if Verbose is True:
       print('        beg end ', mytime_beg, mytime_end, ' for process ', rank, flush=True)

    sc = np.zeros( ntime, dtype=float)
    zc = np.zeros( ntime, dtype=float)
    a = 6371.2 # mean Earth radius

    for itime in range(mytime_beg, mytime_end):
        g10 = ghlm[itime, 0]
        g11 = ghlm[itime, 1]
        h11 = ghlm[itime, 2]
        g20 = ghlm[itime, 3]
        g21 = ghlm[itime, 4]
        h21 = ghlm[itime, 5]
        g22 = ghlm[itime, 6]
        h22 = ghlm[itime, 7]
        # Gallet et al. EPSL 2009 (appendix)
        L0 =  2. * g10 * g20 + np.sqrt(3.) * ( g11 * g21 + h11 * h21             )
        L1 = -1. * g11 * g20 + np.sqrt(3.) * ( g10 * g21 + g11 * g22 + h11 * h22 )
        L2 = -1. * h11 * g20 + np.sqrt(3.) * ( g10 * h21 + g11 * h22 - h11 * g22 )
        m2 = g10**2 + g11**2 + h11**2 
        E = ( L0 * g10 + L1 * g11 + L2 * h11  )/ ( 4. * m2 )
        xc = a * ( L1 - E * g11 ) / (3. * m2 )
        yc = a * ( L2 - E * h11 ) / (3. * m2 )
        zc[itime] = a * ( L0 - E * g10 ) / (3. * m2 )
        sc[itime] = np.sqrt( xc**2 + yc**2 )

    if size>1:
        sc = comm.allreduce( sc, op=MPI.SUM)
        zc = comm.allreduce( zc, op=MPI.SUM)

    if rank==0:
        fname = 'eccentricity'
        np.savez(outdir+'/'+fname, time=time, s_ecc=sc, z_ecc=zc, time_unit = time_unit)
        fname = fname + '.npz'
        config.set('Diags', 'eccentricity_file', fname)
        lfile = open(config_file, 'w')
        config.write(lfile)
        lfile.close()

    if rank==0:
        #distibute plot over 3 rows
        nrows = 3
        ndat = len(time)
        nstep = int( ndat / nrows )
        yticks = np.linspace( -90, 90, 5, dtype=float)
        fig, ax = plt.subplots( nrows, 1, figsize=(15,6), sharey=True)
        for i in range(nrows):
            istart = i * nstep
            if i < nrows -1:
                iend = istart + nstep
            else:
                iend = ndat-1
            ax[i].plot( time[istart:iend], sc[istart:iend], label='s_c')
            ax[i].plot( time[istart:iend], zc[istart:iend], label='z_c')
            ax[i].set_xlim(time[istart],time[iend])
            ax[i].set_ylabel('km')
            ax[i].legend(loc='best')
        ax[nrows-1].set_xlabel('time in ' + time_unit)
        plt.savefig(outdir+'/eccentricity.pdf')
        plt.savefig(outdir+'/eccentricity.png')
        plt.close()

	
    return time, sc, zc, time_unit

def get_rms_intensity( comm, size, rank, config_file):

    if rank == 0:
        print()
        print('  Analysis of geomagnetic intensity ')
        print()

    config_file = config_file
    config = configparser.ConfigParser(interpolation=None)
    config.read(config_file)
    Verbose = config['Common'].getboolean('Verbose')
    fname_gauss = config['Gauss coefficients']['filename']
    gauss_unit = config['Gauss coefficients']['unit']
    outdir = config['Common']['output_directory']
    ltrunc_gauss = int(config['Gauss coefficients']['ltrunc'])
    tag = config['Common']['tag']
    time_unit = config['Rescaling factors and units']['time unit']
    ltrunc_SHB = int(config['Design matrices on regular grid']['l_trunc'])
    nlat = int(config['Design matrices on regular grid']['nlat'])
    nlon = int(config['Design matrices on regular grid']['nlon'])
    fname_SHB = config['Design matrices on regular grid']['filename']

    npzfile =  np.load(outdir+'/'+fname_gauss)
    time = npzfile['time']
    ghlm = npzfile['ghlm']

    npzfile_SHB = np.load(outdir+'/'+fname_SHB)
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
        ier = comm.Barrier()
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
    if rank == 0:
        fname = 'F_rms'
        np.savez(outdir+'/'+fname, time=time, F_rms=F_rms, gauss_unit = gauss_unit, time_unit = time_unit)
        fname = fname + '.npz'
        config.set('Diags', 'rms_intensity_file', fname)
        lfile = open(config_file, 'w')
        config.write(lfile)
        lfile.close()

    if rank==0:
        #distibute plot over 3 rows
        nrows = 3
        ndat = len(time)
        nstep = int( ndat / nrows )
        #yticks = np.linspace( -90, 90, 5, dtype=float)
        fig, ax = plt.subplots( nrows, 1, figsize=(15,6), sharey=True)
        for i in range(nrows):
            istart = i * nstep
            if i < nrows -1:
                iend = istart + nstep
            else:
                iend = ndat-1
            ax[i].plot( time[istart:iend], F_rms[istart:iend])
            ax[i].set_xlim(time[istart],time[iend])
            #ax[i].set_yticks(yticks)
            ax[i].set_ylabel('rms Intensity in ' + gauss_unit)
        ax[nrows-1].set_xlabel('time in ' + time_unit)
        plt.savefig(outdir+'/F_rms.pdf')
        plt.savefig(outdir+'/F_rms.png')
        plt.close()

        
    return time, F_rms, gauss_unit, time_unit    

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

def compute_Delta_QPM(QPMsimu, QPMearth, Verbose=False):
    if ( np.abs(QPMsimu.a_med) > QPMearth.a_med):
        denom = (QPMearth.a_high - QPMearth.a_med) + (np.abs(QPMsimu.a_med) - np.abs(QPMsimu.a_low))
        deltaQPM_a = ( np.abs( QPMsimu.a_med) - QPMearth.a_med ) / denom
    else:
        denom = (QPMearth.a_med - QPMearth.a_low) + (np.abs(QPMsimu.a_high) - np.abs(QPMsimu.a_med))
        deltaQPM_a = ( QPMearth.a_med - np.abs(QPMsimu.a_med) ) / denom
#
    if ( np.abs(QPMsimu.b_med) > QPMearth.b_med):
        denom = (QPMearth.b_high - QPMearth.b_med) + (np.abs(QPMsimu.b_med) - np.abs(QPMsimu.b_low))
        deltaQPM_b = ( np.abs( QPMsimu.b_med) - QPMearth.b_med ) / denom
    else:
        denom = (QPMearth.b_med - QPMearth.b_low) + (np.abs(QPMsimu.b_high) - np.abs(QPMsimu.b_med))
        deltaQPM_b = ( QPMearth.b_med - np.abs(QPMsimu.b_med) ) / denom
#
    if ( np.abs(QPMsimu.delta_Inc_med) > QPMearth.delta_Inc_med):
        diff = np.abs( QPMsimu.delta_Inc_med) - QPMearth.delta_Inc_med
        denom = (QPMearth.delta_Inc_high - QPMearth.delta_Inc_med) + (np.abs(QPMsimu.delta_Inc_med) - np.abs(QPMsimu.delta_Inc_low))
        deltaQPM_delta_Inc = diff / denom
    else:
        diff = QPMearth.delta_Inc_med - np.abs( QPMsimu.delta_Inc_med)        
        denom = (QPMearth.delta_Inc_med - QPMearth.delta_Inc_low) + (np.abs(QPMsimu.delta_Inc_high) - np.abs(QPMsimu.delta_Inc_med))
        deltaQPM_delta_Inc = diff / denom
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

def compute_mean_VGP( VGP_lat, VGP_lon):
#
    l = np.cos(np.deg2rad(VGP_lat)) * np.cos(np.deg2rad(VGP_lon))
    m = np.cos(np.deg2rad(VGP_lat)) * np.sin(np.deg2rad(VGP_lon))
    n = np.sin(np.deg2rad(VGP_lat))
    r = np.sqrt( np.sum(l, axis=-1)**2 + np.sum(m, axis=-1)**2 + np.sum(n, axis=-1)**2 )
    l_site = np.sum(l, axis=-1) / r
    m_site = np.sum(m, axis=-1) / r
    n_site = np.sum(n, axis=-1) / r

    VGP_lam_avg = np.arcsin(n_site)
    VGP_phi_avg = np.arctan2(m_site,l_site)
    VGP_lat_avg = np.rad2deg(VGP_lam_avg)
    VGP_lon_avg = np.mod(np.rad2deg(VGP_phi_avg), 360.)

    return VGP_lat_avg, VGP_lon_avg

def compute_S( VGP_lat, VGP_lon, VGP_lat_avg, VGP_lon_avg):
#
    delta = angular_distance_2sphere( VGP_lat, VGP_lon, \
                                      np.transpose(np.tile(VGP_lat_avg, (np.shape(VGP_lat)[-1],1))), \
                                      np.transpose(np.tile(VGP_lon_avg, (np.shape(VGP_lon)[-1],1))), Verbose=False)

    ASD = np.sqrt(np.sum(delta**2,axis=-1)/(np.shape(delta)[-1]-1))
    return ASD

def compute_Svd( VGP_lat, VGP_lon, VGP_lat_avg, VGP_lon_avg):
#   For future vectorized version
    """
#
    delta = angular_distance_2sphere( VGP_lat, VGP_lon, \
                                      np.transpose(np.tile(VGP_lat_avg, (np.shape(VGP_lat)[-1],1))), \
                                      np.transpose(np.tile(VGP_lon_avg, (np.shape(VGP_lon)[-1],1))), Verbose=False)

    ASD = np.sqrt(np.sum(delta**2,axis=-1)/(np.shape(delta)[-1]-1))
    A = (1.8 * ASD + 5.)
    delta_max = np.max(delta, axis=-1)
    Ndat = np.ones( len(delta_max), dtype=int)
    for i in range(len(delta_max)):
        VGP_lat_loc = VGP_lat[i,:]
        VGP_lon_loc = VGP_lon[i,:]
        Aloc = A[i]
        ASD_loc = ASD[i]
        dmax_loc = delta_max[i]
        delta_loc = delta[i,:]
        Ndat_loc = Ndat[i]
        while dmax_loc > Aloc:
            mask_delta = ( delta_loc < dmax_loc )
            VGP_lat_loc = VGP_lat_loc[ mask_delta ]
            VGP_lon_loc = VGP_lon_loc[ mask_delta ]
            VGP_lat_avg_loc, VGP_lon_avg_loc = compute_mean_VGP( VGP_lat_loc, VGP_lon_loc)
            delta_loc = angular_distance_2sphere( VGP_lat_loc, VGP_lon_loc, \
                                      VGP_lat_avg_loc * np.ones( (len(VGP_lat_loc)), dtype=float), \
                                      VGP_lon_avg_loc * np.ones( (len(VGP_lon_loc)), dtype=float), Verbose=False)
            ASD_loc = np.sqrt( np.sum(delta_loc**2)/(np.shape(delta_loc)[0]-1) )
            Aloc = 1.8 * ASD_loc + 5.
            dmax_loc = np.max(delta_loc)
            Ndat_loc = len(delta_loc)
        ASD[i] = ASD_loc
        A[i] = Aloc
        Ndat[i] = Ndat_loc
    """

    VGP_lat_loc = VGP_lat
    VGP_lon_loc = VGP_lon
    VGP_lat_avg_loc, VGP_lon_avg_loc = compute_mean_VGP( VGP_lat_loc, VGP_lon_loc)
    delta_loc = angular_distance_2sphere( VGP_lat_loc, VGP_lon_loc, \
                                  VGP_lat_avg_loc * np.ones( (len(VGP_lat_loc)), dtype=float), \
                                  VGP_lon_avg_loc * np.ones( (len(VGP_lon_loc)), dtype=float), Verbose=False)
    ASD_loc = np.sqrt( np.sum(delta_loc**2)/(np.shape(delta_loc)[0]-1) )
    Aloc = 1.8 * ASD_loc + 5.
    dmax_loc = np.max(delta_loc)
    Ndat_loc = len(delta_loc)
    while dmax_loc > Aloc:
        mask_delta = ( delta_loc < dmax_loc )
        VGP_lat_loc = VGP_lat_loc[ mask_delta ]
        VGP_lon_loc = VGP_lon_loc[ mask_delta ]
        VGP_lat_avg_loc, VGP_lon_avg_loc = compute_mean_VGP( VGP_lat_loc, VGP_lon_loc)
        delta_loc = angular_distance_2sphere( VGP_lat_loc, VGP_lon_loc, \
                                  VGP_lat_avg_loc * np.ones( (len(VGP_lat_loc)), dtype=float), \
                                  VGP_lon_avg_loc * np.ones( (len(VGP_lon_loc)), dtype=float), Verbose=False)
        ASD_loc = np.sqrt( np.sum(delta_loc**2)/(np.shape(delta_loc)[0]-1) )
        Aloc = 1.8 * ASD_loc + 5.
        dmax_loc = np.max(delta_loc)
        Ndat_loc = len(delta_loc)
    ASD = ASD_loc
    A = Aloc
    Ndat = Ndat_loc

    return ASD,  A, Ndat

def compute_QPM(comm, size, rank, config_file):
    import psv10_class 
    import random
    from scipy.optimize import curve_fit
    if rank == 0:
        print()
        print('  Calculation of QPM (Sprain et al. EPSL 2019) ')
        print()

	# paleomag reference  
    QPMearth = psv10_class.QPM(type='QPM_std', acro='earth')
    Filename='Sprain_etal_EPSL_2019_1-s2.0-S0012821X19304509-mmc7.txt'
    datPSV10 = psv10_class.DataTable(Filename, type='PSV10')
    lmax = datPSV10.l_trunc
    nloc = len(datPSV10.location)

    # this dynamo simulation
    config_file = config_file
    config = configparser.ConfigParser(interpolation=None)
    config.read(config_file)
    tag = config['Common']['tag']
    Verbose = config['Common'].getboolean('Verbose')
    fname_gauss = config['Gauss coefficients']['filename']
    outdir = config['Common']['output_directory']
    gauss_unit = config['Gauss coefficients']['unit']
    time_unit = config['Rescaling factors and units']['time unit']
    ltrunc_gauss = int(config['Gauss coefficients']['ltrunc'])

    QPMsimu = psv10_class.QPM(type='QPM_std', acro=tag)
    npzfile = np.load(outdir+'/'+fname_gauss)
    t = npzfile['time']
    ghlm_arr = npzfile['ghlm'][:,0:lmax*(lmax+2)]
    nsamp = len(t)
    simulation_time = t[-1] - t[0]
    time_intervals = np.ediff1d(t, to_end=[0.])
    if rank == 0:
        print('    total number of samples = ', nsamp)
        print('    total simulation time =', simulation_time ,' ', time_unit)

    dipole_latitude = np.zeros(nsamp, dtype=float)
    for i in range(nsamp):
        g10 = ghlm_arr[ i, 0]
        g11 = ghlm_arr[ i, 1]
        h11 = ghlm_arr[ i, 2]
        dipole_latitude[i] = np.rad2deg(np.arctan2( g10 , np.sqrt(g11**2+h11**2) ))
    #
    # Rev criterion
    #
    # "normal" polarity 
    mask_n = dipole_latitude > 45.
    tau_n = np.sum(time_intervals[mask_n]) / simulation_time
    # "reverse" polarity
    mask_r = dipole_latitude < -45.
    tau_r = np.sum(time_intervals[mask_r]) / simulation_time
    # excursion time
    mask_t = np.abs(dipole_latitude) < 45.
    tau_t = np.sum(time_intervals[mask_t]) / simulation_time

    QPMsimu.taut = tau_t
    QPMsimu.taur = tau_r
    QPMsimu.taun = tau_n
    if rank == 0 and Verbose is True:
        print('     tau_t = ', tau_t)
	
    nbins = 19
    bins = np.linspace( -90, 90, nbins)
#   number of localities
    nloc_tot = len(datPSV10.location)
    if rank == 0 and Verbose is True:
        print('    total number of localities = ', nloc_tot)
    ndraw = 10000 
    inc_anom = np.zeros( (ndraw, nbins), dtype=float)
    scatter_squared = np.zeros( (ndraw, nloc_tot), dtype=float)
    Vpercent = np.zeros( (ndraw), dtype=float )
    bin_lat = np.zeros( nbins, dtype=float)
    empty_bin = np.zeros( nbins, dtype=bool)
    empty_bin[:] = True
    r_earth = 6371.2e3
    vdm_fact = 1.e7 * r_earth**3
   # mpi 1D domain decomposition
    ndraw_per_process = int(ndraw / size)
    mydraw_beg = rank * ndraw_per_process
    mydraw_end = mydraw_beg + ndraw_per_process
    if rank==size-1:
        mydraw_end = ndraw

    for idraw in range(mydraw_beg, mydraw_end):
        if np.mod(idraw+1-mydraw_beg,ndraw_per_process/10) == 0 and Verbose is True:
            if rank == 0:
                print('        rank ', rank, ' performed ', idraw+1-mydraw_beg, 'draws', flush=True)
        VDM = None
        iloc_glob = -1
        for ibin in range(len(bins)-1):
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
                    # take nsite random samples
                    timesteps = random.sample(range(nsamp), nsite)
                    VGP_lat = []
                    VGP_lon = []
                    gh = np.transpose( np.reshape( np.repeat( -1. * np.sign( ghlm_arr[ timesteps,0]), 120, axis=0 ), (nsite,120) ) * ghlm_arr[ timesteps, :] )
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
                    p = np.arctan2(2.* H, Z )
                    VGP_lam = np.arcsin( np.sin(np.deg2rad(my_latitude_in_deg[iloc])) * np.cos(p)\
					                    + np.cos(np.deg2rad(my_latitude_in_deg[iloc]))*np.sin(p)*np.cos(Dec) )
                    beta = np.rad2deg( np.arcsin( np.sin(p) * np.sin(Dec) / np.cos(VGP_lam) ) )
                    VGP_phi = np.zeros(nsite, dtype=float)
                    for istep in range(nsite):
                        if ( np.cos(p[istep]) > ( np.sin(np.deg2rad(my_latitude_in_deg[iloc])) * np.sin(VGP_lam[istep]) ) ):
                            VGP_phi[istep] = my_longitude_in_deg[iloc] + beta[istep]
                        else:
                            VGP_phi[istep] = my_longitude_in_deg[iloc] + 180. - beta[istep]
                        if ( VGP_lam[istep] < 0. ):
                            VGP_lam[istep] = -1. * VGP_lam[istep]
                            VGP_phi[istep] = VGP_phi[istep] + 180.
                    VGP_lat = np.rad2deg(VGP_lam)
                    VGP_lon = np.mod(VGP_phi, 360.)
                    VGP_lat_avg, VGP_lon_avg = compute_mean_VGP( VGP_lat, VGP_lon)
                    if n_north is None:
                        n_north =  X/F
                        n_east =  Y/F
                        n_down =  Z/F
                    else:
                        n_north = np.concatenate( (n_north, X/F), axis=None)
                        n_east = np.concatenate( (n_east, Y/F), axis=None)
                        n_down = np.concatenate( (n_down, Z/F), axis=None)
                    """
                    delta = 90. - VGP_lat
                    ASD = np.sqrt(np.sum(delta**2)/(np.shape(delta)[0]-1))
                    A = 1.8 * ASD + 5.
                    delta_max = np.max(delta)
                    while delta_max > A:
                        mask_delta = delta < delta_max * np.ones_like(delta)
                        delta = delta[ mask_delta ]
                        ASD = np.sqrt(np.sum(delta**2)/(np.shape(delta)[0]-1))
                        A = 1.8 * ASD + 5.
                        delta_max = np.max(delta)
                    scatter_squared[idraw, iloc_glob] = np.sum(delta**2)/(np.shape(delta)[0]-1)
				# Nuts and bolts of paleomagnetism, Cox & Hart, page 310
                    """
                    Svd, cutoff, Ndat = compute_Svd( VGP_lat, VGP_lon, VGP_lat_avg, VGP_lon_avg)
                    scatter_squared[idraw, iloc_glob] = Svd**2
                r_north = np.sum(n_north)
                r_east = np.sum(n_east)
                r_down = np.sum(n_down)
                inc_avg = np.arctan2( r_down , np.sqrt( r_north**2 + r_east**2 ) )
                inc_anom[idraw, ibin] = np.rad2deg(inc_avg - Inc_GAD)				
        Vmed = np.median(VDM, axis=None)
        VDM75, VDM25 = np.percentile(VDM, [75 ,25])
        Viqr = VDM75 - VDM25
        Vpercent[idraw] = Viqr / Vmed

    a = np.zeros( ndraw, dtype=float)
    b = np.zeros( ndraw, dtype=float)
    for idraw in range(mydraw_beg, mydraw_end):
        my_scatter_squared = scatter_squared[idraw,:]
        mask_test = np.isfinite(my_scatter_squared)
        my_scatter_squared = my_scatter_squared[mask_test]
        my_latitude_in_deg = datPSV10.latitude_in_deg[mask_test]
        # popt, pcov = curve_fit( quadratic_disp, np.abs(my_latitude_in_deg), my_scatter_squared, check_finite=True, p0=[25.,0.5], method='dogbox', bounds=([0.,0.],[100.,1.]))
        popt, pcov = curve_fit( quadratic_disp, np.abs(my_latitude_in_deg), my_scatter_squared, check_finite=True, p0=[25.,0.5], bounds=([0.,0.],[100.,2.]))
        a[idraw] = np.abs(popt[0])
        b[idraw] = np.abs(popt[1])
	#
	# Global gather if draw done in parallel
    # Vpercent
    if size>1:
        Vpercent = comm.allreduce(Vpercent, op=MPI.SUM)
        a = comm.allreduce(a, op=MPI.SUM)
        b = comm.allreduce(b, op=MPI.SUM)
# dbg
        scatter_squared = comm.allreduce(scatter_squared, op=MPI.SUM) 
#       scatter_squared = np.reshape(scatter_squared, (ndraw,nloc_tot))
#
        for ibin in range(len(bins)-1):
            inc_anom[:,ibin] = comm.allreduce(inc_anom[:,ibin], op=MPI.SUM)
	# inspection of arrays
    Vpercent = Vpercent[np.isfinite(Vpercent)]	
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
# dbg
#   scatter_squared = scatter_squared[np.isfinite(scatter_squared)]
#   mean_Ssq = np.mean(scatter_squared, axis=0)
#   if rank == 0:
#        print(np.shape(mean_Ssq), flush=True)
#        print(np.shape(scatter_squared), flush=True)
#        Sfile = open('S_vd_mean.dat','w+')
#        for i in range(np.size(mean_Ssq)):
#             Sfile.write( '%g %g \n' % (my_latitude_in_deg[i], mean_Ssq[i] ) )
#        Sfile.close()
    
    QPMsimu.Vpercent_med = np.median( Vpercent)
    QPMsimu.Vpercent_low = np.percentile( Vpercent, 2.5)
    QPMsimu.Vpercent_high = np.percentile( Vpercent, 97.5)
    if rank ==0 and Verbose is True:
        print()
        print('        Vpercent med low high  = ', QPMsimu.Vpercent_med, QPMsimu.Vpercent_low, QPMsimu.Vpercent_high)
        print()
    QPMsimu.a_med = np.median(a)
    QPMsimu.a_low = np.percentile(a, 2.5)
    QPMsimu.a_high = np.percentile(a, 97.5)
    QPMsimu.b_med = np.median(b)
    QPMsimu.b_low = np.percentile(b, 2.5)
    QPMsimu.b_high = np.percentile(b, 97.5)
    if rank ==0 and Verbose is True: 
        print()
        print('        a med low high  = ', QPMsimu.a_med, QPMsimu.a_low, QPMsimu.a_high)
        print('        b med low high  = ', QPMsimu.b_med, QPMsimu.b_low, QPMsimu.b_high)
        print()

    inc_anom_median = np.zeros(nbins, dtype=float)
    inc_anom_low = np.zeros(nbins, dtype=float)
    inc_anom_high = np.zeros(nbins, dtype=float)
    for ibin in range(len(bins)-1):
        if not empty_bin[ibin]:
            inc_anom_median[ibin] = np.median(     inc_anom[:,ibin])
            if inc_anom_median[ibin] > 0:
                inc_anom_low[ibin]    = np.percentile( inc_anom[:,ibin], 2.5)
                inc_anom_high[ibin]   = np.percentile( inc_anom[:, ibin], 97.5)
            else:
                inc_anom_low[ibin]    = np.percentile( inc_anom[:,ibin], 97.5)
                inc_anom_high[ibin]   = np.percentile( inc_anom[:, ibin], 2.5)
            if rank ==0 and Verbose is True:
                print( "        %12.3f %12.3f %12.3f %12.3f " % (bin_lat[ibin], inc_anom_median[ibin], inc_anom_low[ibin], inc_anom_high[ibin]) )
    inc_ind_max = np.argmax(np.abs(inc_anom_median))
    QPMsimu.delta_Inc_med = inc_anom_median[inc_ind_max]
    QPMsimu.delta_Inc_low = inc_anom_low[inc_ind_max]
    QPMsimu.delta_Inc_high = inc_anom_high[inc_ind_max]
    if rank == 0 and Verbose is True:
        print()
        print('QPMsimu.delta_Inc_med = %10.2f QPMsimu.delta_Inc_low = %10.2f  QPMsimu.delta_Inc_high = %10.2f '\
        % (QPMsimu.delta_Inc_med, QPMsimu.delta_Inc_low, QPMsimu.delta_Inc_high))
	# to complete

    Verbose = False
    if rank == 0:
       Verbose = True
       compute_Delta_QPM(QPMsimu, QPMearth, Verbose=Verbose)

    QPM_results = [] 
    return QPM_results
#
