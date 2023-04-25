import numpy as np
import shtns 

def surfint(f,sh):

    flm = sh.analys(f.T) #.T)
    summ = flm[0] * 4. * np.pi

    return np.real(summ)

def green_x(field,t0,p0,sh):  

    rc = 3485.0
    ra = 6371.2
    b = rc / ra 
    bb = b * b

    cost = np.mat(sh.cos_theta)
    sint = np.sqrt(1.-np.power(cost,2))
    sintt = np.transpose(sint)
    costt = np.transpose(cost)
    nlon = field.shape[0]
    phi = np.mat((2. * np.pi / np.float(nlon)) * np.arange(1,nlon+1))
    phit = np.transpose(phi)
    cosp = np.mat(np.cos(phi))
    onesp = np.mat(np.ones((1,phi.shape[1])))
    onespt = np.transpose(onesp)
        
    mu = np.sin(t0) * (np.cos(p0-phit) * sint)
    mu = mu + np.cos(t0) * (onespt * cost);
    f  = np.sqrt(np.ones(field.shape) * (1 + bb) - (2 * b * mu));
 
    G = (np.ones(field.shape) + (2 * f) - (bb * np.ones(field.shape))) * (b**3);
    G = G / np.multiply(np.power(f,3),np.ones(field.shape) + f - (b * mu));
    G = np.multiply(G,np.cos(t0) * (np.cos(phit-p0) * sint) - np.sin(t0) * (onespt * cost));
    G = G / (4 * np.pi);

    G = np.multiply(G,field)
    x = surfint(G,sh)

    return x    

def gf_x(field,t0,p0,sh):

    rc = 3485.0
    ra = 6371.2
    b = rc / ra
    bb = b * b

    cost = np.mat(sh.cos_theta)
    sint = np.sqrt(1.-np.power(cost,2))
    sintt = np.transpose(sint)
    costt = np.transpose(cost)
    nlon = field.shape[0]
    phi = np.mat((2. * np.pi / np.float(nlon)) * np.arange(1,nlon+1))
    phit = np.transpose(phi)
    cosp = np.mat(np.cos(phi))
    onesp = np.mat(np.ones((1,phi.shape[1])))
    onespt = np.transpose(onesp)

    mu = np.sin(t0) * (np.cos(p0-phit) * sint)
    mu = mu + np.cos(t0) * (onespt * cost);
    f  = np.sqrt(np.ones(field.shape) * (1 + bb) - (2 * b * mu));

    G = (np.ones(field.shape) + (2 * f) - (bb * np.ones(field.shape))) * (b**3);
    G = G / np.multiply(np.power(f,3),np.ones(field.shape) + f - (b * mu));
    G = np.multiply(G,np.cos(t0) * (np.cos(phit-p0) * sint) - np.sin(t0) * (onespt * cost));
    G = G / (4 * np.pi);

    return G

def green_y(field,t0,p0,sh):

    rc = 3485.0
    ra = 6371.2
    b = rc / ra
    bb = b * b

    cost = np.mat(sh.cos_theta)
    sint = np.sqrt(1.-np.power(cost,2))
    sintt = np.transpose(sint)
    costt = np.transpose(cost)
    nlon = field.shape[0]
    phi = np.mat((2. * np.pi / np.float(nlon)) * np.arange(1,nlon+1))
    phit = np.transpose(phi)
    cosp = np.mat(np.cos(phi))
    onesp = np.mat(np.ones((1,phi.shape[1])))
    onespt = np.transpose(onesp)

    mu = np.sin(t0) * (np.cos(p0-phit) * sint)
    mu = mu + np.cos(t0) * (onespt * cost);
    f  = np.sqrt(np.ones(field.shape) * (1 + bb) - (2 * b * mu));

    G = (np.ones(field.shape) + (2 * f) - (bb * np.ones(field.shape))) * (b**3);
    G = G / np.multiply(np.power(f,3),np.ones(field.shape) + f - (b * mu));
    G = -1. * np.multiply(G,np.sin(phit-p0) * sint) 
    G = G / (4 * np.pi);

    G = np.multiply(G,field)
    y = surfint(G,sh)

    return y

def green_z(field,t0,p0,sh):

    rc = 3485.0
    ra = 6371.2
    b = rc / ra
    bb = b * b

    cost = np.mat(sh.cos_theta)
    sint = np.sqrt(1.-np.power(cost,2))
    sintt = np.transpose(sint)
    costt = np.transpose(cost)
    nlon = field.shape[0]
    phi = np.mat((2. * np.pi / np.float(nlon)) * np.arange(1,nlon+1))
    phit = np.transpose(phi)
    cosp = np.mat(np.cos(phi))
    onesp = np.mat(np.ones((1,phi.shape[1])))
    onespt = np.transpose(onesp)

    mu = np.sin(t0) * (np.cos(p0-phit) * sint)
    mu = mu + np.cos(t0) * (onespt * cost);
    f  = np.sqrt(np.ones(field.shape) * (1 + bb) - (2 * b * mu));

    G = -1. * (bb - bb**2)*np.ones(field.shape)
    G = (G / np.power(f,3)) + (bb * np.ones(field.shape))
    G = G / (4 * np.pi);

    G = np.multiply(G,field)
    z = surfint(G,sh)

    return z

def surf_xyz(ybpr,phi,theta,r,sh):

    rc = 3485.0
    ra = 6371.2

    nlat = theta.shape[0]
    nlon = phi.shape[0]

    blm1 = np.zeros_like(ybpr)
    blm2 = np.zeros_like(ybpr)
 
    for lm in range(sh.nlm):
        l = sh.l[lm]
        m = sh.m[lm]
        blm1[lm] = ((rc/ra)**(l+2)) * l * (l+1) * ybpr[lm] / r
        blm2[lm] = ((rc/ra)**(l+2)) * l * -ybpr[lm] / r

    br = np.zeros((nlon,nlat))
    bt = np.zeros((nlon,nlat))
    bp = np.zeros((nlon,nlat))
    x = np.zeros((nlon,nlat))
    y = np.zeros((nlon,nlat))
    z = np.zeros((nlon,nlat))

    br = sh.synth(blm1)
    bt, bp = sh.synth_grad(blm2) 

    x = - bt
    y = bp 
    z = - br

    return x,y,z



