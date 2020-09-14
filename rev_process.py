import sys
import numpy as np
import shtns
import cartopy.crs as ccrs
import cartopy.util as cutil
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

def clean_series(time, Verbose, myrank):
#
#   make sure time is an increasing function 
#   
    if Verbose is True and myrank==0: 
        print('    checking for redundant data by inspection of time axis ')
        print('        initial size of time array = ', len(time))
    keep = np.zeros_like(time, dtype=bool)
    maxval = time[0]
    keep[0] = True
    for i in range(1,len(time)):
        if time[i] > (maxval + 1.e-5):
            keep[i] = True
            maxval = time[i]
        else:
            keep[i] = False
    if Verbose is True and myrank==0: 
        print('        size of kept time array = ', np.sum(keep))
    return keep

def filter_br(br_in, sh_parody,ltrunc):

    ylm_in = sh_parody.analys(br_in)
    ylm_out = sh_parody.spec_array()
    keep = ( sh_parody.l <= ltrunc )
    for lm in range(len(ylm_in)):
        if keep[lm]:
            ylm_out[lm] = ylm_in[lm]
    br_out = sh_parody.synth(ylm_out)
    return br_out

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
    ghlm =  np.zeros( ltrunc * ( ltrunc + 2) , dtype=float)
    lm = -1

    for il in range(1,ltrunc+1):
        fact = ((il+1)/np.sqrt(2*il+1))*(a/c)**(il+2)
        for im in range(0,il+1,azsym):
            glm[il,im] =      np.real(br_lm[sh.idx(il,im)]) / fact
            hlm[il,im] = -1 * np.imag(br_lm[sh.idx(il,im)]) / fact
            if im == 0:
                lm = lm + 1
                ghlm[lm] =       np.real(br_lm[sh.idx(il,im)]) / fact
            else:
                lm = lm + 1
                ghlm[lm] =       np.real(br_lm[sh.idx(il,im)]) / fact
                lm = lm + 1
                ghlm[lm] =  -1 * np.imag(br_lm[sh.idx(il,im)]) / fact

    if bscale != None:
        glm = bscale * glm
        hlm = bscale * hlm
        ghlm = bscale * ghlm

    return glm, hlm, ghlm

def compute_brlm_from_glmhlm(glm, hlm, sh, ltrunc = None, bscale =None, radius=None):
#
#   inputs :
#   glm, hlm: the Gauss coefficients expressed in nanoTeslas
#   sh: the current shtns framework
#
    a = 6371.2
    c = 3485. # default value for radius where br_lm is estimated
    if radius != None:
        c = radius

    azsym = sh.mres
    if ltrunc == None:
        ltrunc = sh.lmax

    br_lm = sh.spec_array()

    for il in range(1,ltrunc+1):
        for im in range(0,il+1,azsym):
            fact = ((il+1)/np.sqrt(2*il+1))*(a/c)**(il+2)
            br_lm[sh.idx(il,im)] = fact * complex ( glm[il,im] , -1. * hlm[il,im] )

    if bscale != None:
        br_lm = bscale * br_lm

    return br_lm

def compute_brlm_from_glmhlm(glm, hlm, sh, ltrunc = None, bscale =None, radius=None):
#
#   inputs :
#   glm, hlm: the Gauss coefficients expressed in nanoTeslas
#   sh: the current shtns framework
#
    a = 6371.2
    c = 3485. # default value for radius where br_lm is estimated
    if radius != None:
        c = radius

    azsym = sh.mres
    if ltrunc == None:
        ltrunc = sh.lmax

    br_lm = sh.spec_array()

    for il in range(1,ltrunc+1):
        for im in range(0,il+1,azsym):
            fact = ((il+1)/np.sqrt(2*il+1))*(a/c)**(il+2)
            br_lm[sh.idx(il,im)] = fact * complex ( glm[il,im] , -1. * hlm[il,im] )

    if bscale != None:
        br_lm = bscale * br_lm

    return br_lm




def mollweide_dual_surface(F,Z,theta,phi,fname=None,vmax1=None,vmin1=None,vmax2=None,vmin2=None,Title1=None, Title2=None,\
positive1=False, positive2=False, cmap1=None, cmap2=None):

    lats = .5 * np.pi - theta
    lats = np.rad2deg(lats)
    lons = np.rad2deg(phi)
    fig = plt.figure( figsize=(10,7) )
#
    cyclic_data, cyclic_lons = cutil.add_cyclic_point(F, coord=lons)
    if vmax1 is None:
        vmax1 = np.max(cyclic_data)
        if vmin1 is None: 
            vmin1 = np.min(cyclic_data)
        vmax1 = max(vmax1,-1.*vmin1)
        print('vmax1 = ', vmax1)
    if positive1 is False:
        levels =  np.linspace(-vmax1,vmax1, 21, endpoint=True)
        ticks =  np.linspace(-vmax1,vmax1, 11, endpoint=True)
    else:
        levels =  np.linspace(vmin1,vmax1, 21, endpoint=True)
        ticks =  np.linspace(vmin1,vmax1, 11, endpoint=True)
    if cmap1 is not None:
        cmap1 = cmap1
    else:
        cmap1 = 'seismic'
    ax1 = plt.subplot(1, 2, 1, projection = ccrs.Mollweide(central_longitude=0))
    x, y = np.meshgrid(cyclic_lons, lats)
    m1 = ax1.contourf(x, y, cyclic_data, levels=levels, transform = ccrs.PlateCarree(), cmap=cmap1, extend='both')
    ax1.set_global()
    ax1.coastlines()
    if Title1 is not None:
        ax1.set_title(Title1)
#
    cyclic_data, cyclic_lons = cutil.add_cyclic_point(Z, coord=lons)
    if vmax2 is None:
        vmax2 = np.max(cyclic_data)
        if vmin2 is None: 
            vmin2 = np.min(cyclic_data)
        vmax2 = max(vmax2,-1.*vmin2)
        print('vmax2 = ', vmax2)
    if positive2 is False:
        levels =  np.linspace(-vmax2,vmax2, 21, endpoint=True)
        ticks =  np.linspace(-vmax2,vmax2, 11, endpoint=True)
    else:
        levels =  np.linspace(vmin2,vmax2, 21, endpoint=True)
        ticks =  np.linspace(vmin2,vmax2, 11, endpoint=True)
    if cmap2 is not None:
        cmap2 = cmap2
    else:
        cmap2 = 'seismic'
    ax2 = plt.subplot(1, 2, 2, projection = ccrs.Mollweide(central_longitude=0))
    x, y = np.meshgrid(cyclic_lons, lats)
    m2 = ax2.contourf(x, y, cyclic_data, levels=levels, transform = ccrs.PlateCarree(), cmap=cmap2, extend='both')
    ax2.set_global()
    ax2.coastlines()
    if Title2 is not None:
        ax2.set_title(Title2)
#   colorbars
    cax1 = fig.add_axes([0.05, .25, .4, 0.03])
    cm = fig.colorbar(m1, cax=cax1, ticks=ticks, orientation='horizontal', )
    cm.set_label(r"$\mu$T")
    cax2 = fig.add_axes([0.55, .25, .4, 0.03])
    cm = fig.colorbar(m2, cax=cax2, ticks=ticks, orientation='horizontal', )
    cm.set_label(r"$\mu$T")
    fig.tight_layout()
    if fname is not None:
       plt.savefig(fname+'.png', bbox_inches='tight', dpi=300)
       #plt.savefig(fname+'.pdf')
    else:
        plt.show()
    plt.close()
    return vmax1, vmax2

def mollweide_surface(F,theta,phi,fname=None,vmax=None,vmin=None,Title=None, positive=False, cmap=None):

    lats = .5 * np.pi - theta
    lats = np.rad2deg(lats)
    lons = np.rad2deg(phi)
    cyclic_data, cyclic_lons = cutil.add_cyclic_point(F, coord=lons)
    if vmax is None:
        vmax = np.max(cyclic_data)
        if vmin is None: 
            vmin = np.min(cyclic_data)
        vmax = max(vmax,-1.*vmin)
        print('vmax = ', vmax)
    if positive is False:
        levels =  np.linspace(-vmax,vmax, 21, endpoint=True)
        ticks =  np.linspace(-vmax,vmax, 11, endpoint=True)
    else:
        levels =  np.linspace(vmin,vmax, 21, endpoint=True)
        ticks =  np.linspace(vmin,vmax, 11, endpoint=True)
    if cmap is not None:
        cmap = cmap
    else:
        cmap = 'seismic'
    fig = plt.figure()
    ax = plt.axes( projection = ccrs.Mollweide(central_longitude=0))
    x, y = np.meshgrid(cyclic_lons, lats)
    m = ax.contourf(x, y, cyclic_data, levels=levels, transform = ccrs.PlateCarree(), cmap=cmap, extend='both')
    ax.set_global()
    ax.coastlines()
    if Title is not None:
        plt.title(Title)
    plt.tight_layout()
    cax = fig.add_axes([0.05, .1, .9, 0.05])
    cm = fig.colorbar(m, cax=cax, ticks=ticks, orientation='horizontal', )
    cm.set_label(r"$\mu$T")
    plt.tight_layout()
    if fname is not None:
       plt.savefig(fname+'.png')
       #plt.savefig(fname+'.pdf')
    else:
        plt.show()
    plt.close()
    return vmax

def mollweide_scatter(lon,lat,fname=None,Title=None):

    fig = plt.figure()
    ax = plt.axes( projection = ccrs.Mollweide(central_longitude=0))
    #ax.scatter(lon, lat, marker='x', color='k', s=4, transform = ccrs.Geodetic() )
    for i in range(20000,25000):
        ax.plot([lon[i],lon[i+1]], [lat[i], lat[i+1] ] , color='r', lw=2, transform = ccrs.Geodetic())
    ax.set_global()
    ax.stock_img()
    ax.gridlines()
    ax.coastlines()
    if Title is not None:
        plt.title(Title)
    plt.tight_layout()
    if fname is not None:
       plt.savefig(fname+'.png')
       plt.savefig(fname+'.pdf')
    else:
        plt.show()
    plt.close()

def SHB_X(tt, pp, rr, ll, ra=6371.2):
    # author: Vincent Lesur
    #
    # tt (theta), pp (phi), rr (radius) .... all float
    #
    # ll maximum degreee of SH expension
    #  Computes the elements of the equations of condition for the
    #  North component of the magnetic field at a point (theta,phi,r)
    #
    # CALLED: mk_dlf
    #           ra reference radius
    #           np.pi 3.14159...
    try: 
        dtr = np.pi / 180.
        aax = np.zeros(ll*(ll+2), dtype=float)
        rw = ra/rr
        # im = 0
        im = 0
        plm = mk_dlf( ll, im, tt)
        for il in range(1, ll+1, 1):
            k = il*il - 1
            aax[k] = plm[il]*np.power(rw, float(il+2))
        #  im != 0
        for im in range(1, ll+1, 1):
            plm = mk_dlf( ll, im, tt)
            dc = np.cos(float(im)*pp*dtr)
            ds = np.sin(float(im)*pp*dtr)
            for il in range(im, ll+1, 1):
                k = il*il+2*im-2
                ww = plm[il]*np.power(rw, float(il+2))
                aax[k] = ww*dc
                aax[k+1] = ww*ds
        return aax
    except Exception as msg:
        print ('**** ERROR:SHB_X')
        print (msg)

def SHB_Y(tt, pp, rr, ll, ra=6371.2):
    # author: Vincent Lesur
    #
    # tt (theta), pp (phi), rr (radius) .... all float
    #
    #  Computes the elements of the equations of condition for the
    #  East component of the magnetic field at a point (theta,phi,r)
    #
    # ATTENTION : for theta=0 then sin(theta) set to 1.e-10
    #
    # CALLED: mk_lf
    #           ra reference radius
    #           np.pi 3.14159...
    try:
        dtr = np.pi / 180.
        st = np.sin(tt*dtr)
        if st == 0:
            st = 1.e-10
        aay = np.zeros(ll*(ll+2), dtype=float)
        rw = ra/rr
        #  im != 0
        for im in range(1, ll+1, 1):
            plm = mk_lf( ll, im, tt)
            dc = -float(im)*np.sin(float(im)*pp*dtr)
            ds = float(im)*np.cos(float(im)*pp*dtr)
            for il in range(im, ll+1, 1):
                k = il*il+2*im-2
                ww = plm[il]*np.power(rw, float(il+2))/st
                aay[k] = -ww*dc
                aay[k+1] = -ww*ds
        return aay
    except Exception as msg:
        print ('**** ERROR:SHB_Y')
        print (msg)

def SHB_Z(tt, pp, rr, ll, ra=6371.2):
    # author: Vincent Lesur
    #
    # tt (theta), pp (phi), rr (radius) .... all float
    #
    #  Computes the elements of the equations of condition for the
    #  Down component of the magnetic field at a point (theta,phi,r)
    #
    # CALLED: mk_lf
    #           ra reference radius
    #           np.pi 3.14159...
    try:
        dtr = np.pi / 180.
        aaz = np.zeros(ll*(ll+2), dtype=float)
        rw = ra/rr
        # im = 0
        im = 0
        plm = mk_lf( ll, im, tt)
        for il in range(1, ll+1, 1):
            k = il*il - 1
            aaz[k] = -float(il+1)*plm[il]*np.power(rw, float(il+2))
        #  im != 0
        for im in range(1, ll+1, 1):
            plm = mk_lf( ll, im, tt)
            dc = np.cos(float(im)*pp*dtr)
            ds = np.sin(float(im)*pp*dtr)
            for il in range(im, ll+1, 1):
                k = il*il+2*im-2
                ww = -float(il+1)*plm[il]*np.power(rw, float(il+2))
                aaz[k] = ww*dc
                aaz[k+1] = ww*ds
        return aaz
    except Exception as msg:
        print ('**** ERROR:SHB_Z')
        print (msg)

def mk_dlf(ll, im, tt):
    #
    # im integer, tt float
    #
    # Computes the derivative along theta of all legendre functions for
    # a given order up to degree ll at a given position : theta (in degree)
    #
    # CALLED: mk_lf
    # DEFINED:  ll maximum degreee of SH expension
    #           np.pi 3.14159...
    dlf = np.zeros(ll+1, dtype=float)
    #
    dtr = np.pi / 180.
    dc = np.cos(tt*dtr)
    ds = np.sin(tt*dtr)
    if ds == 0.:
        if im == 1:
            dlf[1] = -1.
            dlf[2] = -np.sqrt(3.)
            for ii in range(3, ll+1, 1):
                d1 = float(2*ii-1)/np.sqrt(float(ii*ii-1))
                d2 = np.sqrt(float(ii*(ii-2))/float(ii*ii-1))
                dlf[ii] = d1*dlf[ii-1]-d2*dlf[ii-2]
    else:
        dlf = mk_lf(ll, im, tt)
        for ii in range(ll, im, -1):
            d1 = np.sqrt(float((ii-im)*(ii+im)))
            d2 = float(ii)
            dlf[ii] = (d2*dc*dlf[ii]-d1*dlf[ii-1])/ds
        dlf[im] = float(im)*dc*dlf[im]/ds
    #
    return dlf

def mk_lf( ll, im, tt):
    #
    # im integer, tt float
    #
    # Computes all legendre functions for a given order up to degree ll
    # at a given position : theta (in degree)
    # recurrence 3.7.28 Fundations of geomagetism, Backus 1996
    #
    # CALLED: logfac
    # DEFINED:  ll maximum degreee of SH expension
    #           np.pi 3.14159...
    lf = np.zeros(ll+1, dtype=float)
    #
    dtr = np.pi / 180.
    dc = np.cos(tt*dtr)
    ds = np.sin(tt*dtr)
    #
    dm = float(im)
    d1 = logfac(2.*dm)
    d2 = logfac(dm)
    #
    d2 = 0.5*d1 - d2
    d2 = np.exp(d2 - dm*np.log(2.0))
    if im != 0:
        d2 = d2*np.sqrt(2.0)
    #
    d1 = ds
    if d1 != 0.:
        d1 = np.power(d1, im)
    elif im == 0:
        d1 = 1.
    #
    # p(m,m), p(m+1,m)
    lf[im] = d1*d2
    if im != ll:
        lf[im+1] = lf[im]*dc*np.sqrt(2.*dm+1.)
    #
    # p(m+2,m), p(m+3,m).....
    for ii in range(2, ll-im+1, 1):
        d1 = float((ii-1)*(ii+2*im-1))
        d2 = float((ii)*(ii+2*im))
        db = np.sqrt(d1/d2)
        d1 = float(2*(ii+im)-1)
        da = d1/np.sqrt(d2)
        #
        lf[im+ii] = da*lf[im+ii-1]*dc-db*lf[im+ii-2]
    #
    return lf

def logfac(dd):
    #
    # Calculate log(dd!)
    # dd is a float, but is truncated to an integer before calculation
    #
    id = np.int(dd)
    lgfc = np.sum(np.log(range(1, id+1, 1)))
    return lgfc
