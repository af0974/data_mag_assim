import numpy as np
import copy

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
    id = int(dd)
    lgfc = np.sum(np.log(range(1, id+1, 1)))
    return lgfc

class QPM:

    def __init__(self, type='QPM_std', acro=None):
        if type == 'QPM_std':
            if acro == 'earth':
                self.a_med = 11.33
                self.a_high = 13.26
                self.a_low = 9.69
                self.b_med = 0.256
                self.b_high = 0.299
                self.b_low = 0.206
                self.delta_Inc_med = 7.04
                self.delta_Inc_high = 8.39
                self.delta_Inc_low = 5.64
                self.Vpercent_1Ma = 0.534
                self.Vpercent_10Ma = 0.863
                self.Vpercent_high = 0.863
                self.Vpercent_low = 0.534
                self.Vpercent_med = .5 * ( 0.534 + 0.863 )
                self.taut_low = 0.0375
                self.taut_high = 0.15
                self.taut_med = .5 * ( 0.0375 + 0.15 ) 
            else:
                self.a_med = 0. 
                self.a_high = 0. 
                self.a_low = 0. 
                self.b_med = 0.
                self.b_high = 0.
                self.b_low = 0.
                self.delta_Inc_med = 0.
                self.delta_Inc_high = 0. 
                self.delta_Inc_low = 0. 
                self.Vpercent_med = 0.
                self.Vpercent_high = 0.
                self.Vpercent_low = 0.
                self.taut = 0.
                self.taur = 0.
                self.taun = 0.
                self.Rev = False
                self.VDMvar = False
                self.VGPa = False
                self.VGPb = False

class DataTable: 

    def __init__(self, fileName, type='PSV10', acro=None, ltrunc=None):
        if type == 'PSV10': 
            self.read_PSV10(fileName, acro, ltrunc)

    def read_PSV10(self, fileName, acro, ltrunc): 

        data = np.loadtxt(fileName)
        self.location = data[:,0]
        nloc = len(self.location)
        self.latitude_in_deg = data[:,1]
        self.colatitude_in_deg = 90. - self.latitude_in_deg 
        self.longitude_in_deg = data[:,2]
        self.number_of_sites = data[:,3].astype('int')
        self.radius = 6371.2 * np.ones_like(self.location)
        if ltrunc is not None:
            self.l_trunc = ltrunc 
        else:
            self.l_trunc = 10 
        self.SHB_X = np.zeros( ( nloc,  self.l_trunc * (self.l_trunc + 2) ) )
        self.SHB_Y = np.zeros( ( nloc,  self.l_trunc * (self.l_trunc + 2) ) )
        self.SHB_Z = np.zeros( ( nloc,  self.l_trunc * (self.l_trunc + 2) ) )
        for iloc in range(nloc): 
            self.SHB_X[iloc,:] = SHB_X(self.colatitude_in_deg[iloc], self.longitude_in_deg[iloc], self.radius[iloc], ll=self.l_trunc)
            self.SHB_Y[iloc,:] = SHB_Y(self.colatitude_in_deg[iloc], self.longitude_in_deg[iloc], self.radius[iloc], ll=self.l_trunc)
            self.SHB_Z[iloc,:] = SHB_Z(self.colatitude_in_deg[iloc], self.longitude_in_deg[iloc], self.radius[iloc], ll=self.l_trunc)

    def __add__(self, new):
        out = copy.deepcopy(new)
        out.location = np.hstack( ( self.location, new.location ) )
        out.latitude_in_deg = np.hstack( ( self.latitude_in_deg, new.latitude_in_deg ) )
        out.colatitude_in_deg = np.hstack( ( self.colatitude_in_deg, new.colatitude_in_deg ) )
        out.longitude_in_deg = np.hstack( ( self.longitude_in_deg, new.longitude_in_deg ) )
        out.number_of_sites = np.hstack( ( self.number_of_sites, new.number_of_sites ) )
        out.l_trunc = np.hstack( ( self.l_trunc, new.l_trunc ) )
        out.radius = np.hstack( ( self.radius, new.radius ) )
        out.SHB_X = np.hstack( ( self.SHB_X, new.SHB_X ) )
        out.SHB_Y = np.hstack( ( self.SHB_Y, new.SHB_Y ) )
        out.SHB_Z = np.hstack( ( self.SHB_Z, new.SHB_Z ) )

        return out
