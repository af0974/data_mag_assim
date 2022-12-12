import numpy as np
import matplotlib.pyplot as plt


def hammer2cart(ttheta, pphi, colat=False):
    """
    This function is used to define the Hammer projection used when
    plotting surface contours in :py:class:`magic.Surf`

    >>> # Load Graphic file
    >>> gr = MagicGraph()
    >>> # Meshgrid
    >>> pphi, ttheta = mgrid[-np.pi:np.pi:gr.nphi*1j, np.pi/2.:-np.pi/2.:gr.ntheta*1j]
    >>> x,y = hammer2cart(ttheta, pphi)
    >>> # Contour plots
    >>> contourf(x, y, gr.vphi)

    :param ttheta: meshgrid [nphi, ntheta] for the latitudinal direction
    :type ttheta: numpy.ndarray
    :param pphi: meshgrid [nphi, ntheta] for the azimuthal direction
    :param colat: colatitudes (when not specified a regular grid is assumed)
    :type colat: numpy.ndarray
    :returns: a tuple that contains two [nphi, ntheta] arrays: the x, y meshgrid
              used in contour plots
    :rtype: (numpy.ndarray, numpy.ndarray)
    """

    if not colat: # for lat and phi \in [-pi, pi]
        xx = 2.*np.sqrt(2.) * np.cos(ttheta)*np.sin(pphi/2.)\
             /np.sqrt(1.+np.cos(ttheta)*np.cos(pphi/2.))
        yy = np.sqrt(2.) * np.sin(ttheta)\
             /np.sqrt(1.+np.cos(ttheta)*np.cos(pphi/2.))
    else:  # for colat and phi \in [0, 2pi]
        xx = -2.*np.sqrt(2.) * np.sin(ttheta)*np.cos(pphi/2.)\
             /np.sqrt(1.+np.sin(ttheta)*np.sin(pphi/2.))
        yy = np.sqrt(2.) * np.cos(ttheta)\
             /np.sqrt(1.+np.sin(ttheta)*np.sin(pphi/2.))
    return xx, yy


def radialContour(data, rad=0.85, label=None, proj='hammer', lon_0=0., vmax=None,
                  vmin=None, lat_0=30., levels=32, cm='seismic',
                  normed=True, cbar=True, tit=True, lines=False):
    """
    Plot the radial cut of a given field

    :param data: the input data (an array of size (nphi,ntheta))
    :type data: numpy.ndarray
    :param rad: the value of the selected radius
    :type rad: float
    :param label: the name of the input physical quantity you want to
                  display
    :type label: str
    :param proj: the type of projection. Default is Hammer, in case
                 you want to use 'ortho' or 'moll', then Basemap is
                 required.
    :type proj: str
    :param levels: the number of levels in the contour
    :type levels: int
    :param cm: name of the colormap ('jet', 'seismic', 'RdYlBu_r', etc.)
    :type cm: str
    :param tit: display the title of the figure when set to True
    :type tit: bool
    :param cbar: display the colorbar when set to True
    :type cbar: bool
    :param lines: when set to True, over-plot solid lines to highlight
                  the limits between two adjacent contour levels
    :type lines: bool
    :param vmax: maximum value of the contour levels
    :type vmax: float
    :param vmin: minimum value of the contour levels
    :type vmin: float
    :param normed: when set to True, the colormap is centered around zero.
                   Default is True, except for entropy/temperature plots.
    :type normed: bool
    """

    nphi, ntheta = data.shape

    phi = np.linspace(-np.pi, np.pi, nphi)
    theta = np.linspace(np.pi/2, -np.pi/2, ntheta)
    pphi, ttheta = np.mgrid[-np.pi:np.pi:nphi*1j, np.pi/2.:-np.pi/2.:ntheta*1j]
    lon2 = pphi * 180./np.pi
    lat2 = ttheta * 180./np.pi

    circles = np.r_[-60., -30., 0., 30., 60.]
    delon = 60.
    meridians = np.arange(-180+delon, 180, delon)

    if proj == 'moll' or proj == 'hammer':
        if tit and label is not None:
            if cbar:
                fig = plt.figure(figsize=(9,4.5))
                ax = fig.add_axes([0.01, 0.01, 0.87, 0.87])
            else:
                fig = plt.figure(figsize=(8,4.5))
                ax = fig.add_axes([0.01, 0.01, 0.98, 0.87])
            ax.set_title('%s: r/ro = %.3f' % (label, rad),
                         fontsize=24)
        else:
            if cbar:
                fig = plt.figure(figsize=(9,4))
                ax = fig.add_axes([0.01, 0.01, 0.87, 0.98])
            else:
                fig = plt.figure(figsize=(8,4))
                ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
            #tit1 = r'%.2f Ro' % rad
            #ax.text(0.12, 0.9, tit1, fontsize=16,
                  #horizontalalignment='right',
                  #verticalalignment='center',
                  #transform = ax.transAxes)
    else:
        if tit and label is not None:
            if cbar:
                fig = plt.figure(figsize=(6,5.5))
                ax = fig.add_axes([0.01, 0.01, 0.82, 0.9])
            else:
                fig = plt.figure(figsize=(5,5.5))
                ax = fig.add_axes([0.01, 0.01, 0.98, 0.9])
            ax.set_title('%s: r/ro = %.3f' % (label, rad),
                         fontsize=24)
        else:
            if cbar:
                fig = plt.figure(figsize=(6,5))
                ax = fig.add_axes([0.01, 0.01, 0.82, 0.98])
            else:
                fig = plt.figure(figsize=(5,5))
                ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
            tit1 = r'%.2f Ro' % rad
            ax.text(0.12, 0.9, tit1, fontsize=16,
                  horizontalalignment='right',
                  verticalalignment='center',
                  transform = ax.transAxes)

    if proj != 'hammer':
        from mpl_toolkits.basemap import Basemap
        map = Basemap(projection=proj, lon_0=lon_0, lat_0=lat_0,
                      resolution='c')
        map.drawparallels([0.], dashes=[2, 3], linewidth=0.5)
        map.drawparallels(circles, dashes=[2,3], linewidth=0.5)
        map.drawmeridians(meridians, dashes=[2,3], linewidth=0.5)
        map.drawmeridians([-180], dashes=[20,0], linewidth=0.5)
        map.drawmeridians([180], dashes=[20,0], linewidth=0.5)
        x, y = list(map(lon2, lat2))
    else:
        x, y = hammer2cart(ttheta, pphi)
        for lat0 in circles:
            x0, y0 = hammer2cart(lat0*np.pi/180., phi)
            ax.plot(x0, y0, 'k:', linewidth=0.7)
        for lon0 in meridians:
            x0, y0 = hammer2cart(theta, lon0*np.pi/180.)
            ax.plot(x0, y0, 'k:', linewidth=0.7)
        xxout, yyout  = hammer2cart(theta, -np.pi-1e-3)
        xxin, yyin  = hammer2cart(theta, np.pi+1e-3)
        ax.plot(xxin, yyin, 'k-')
        ax.plot(xxout, yyout, 'k-')
        ax.axis('off')

    cmap = plt.get_cmap(cm)

    if proj == 'ortho':
        lats = np.linspace(-90., 90., ntheta)
        dat = map.transform_scalar(data.T, phi*180/np.pi, lats,
                                   nphi, ntheta, masked=True)
        im = map.imshow(dat, cmap=cmap)
    else:
        if vmax is not None or vmin is not None:
            normed = False
            cs = np.linspace(vmin, vmax, levels)
            im = ax.contourf(x, y, data, cs, cmap=cmap, extend='both')
            if lines:
                ax.contour(x, y, data, cs, colors=['k'], linewidths=0.5,
                           extend='both', linestyles=['-'])
                #ax.contour(x, y, data, 1, colors=['k'])
            #im = ax.pcolormesh(x, y, data, cmap=cmap, antialiased=True)
        else:
            cs = levels
            im = ax.contourf(x, y, data, cs, cmap=cmap)
            if lines:
                ax.contour(x, y, data, cs, colors=['k'], linewidths=0.5,
                           linestyles=['-'])
            #im = ax.pcolormesh(x, y, data, cmap=cmap, antialiased=True)


    # Add the colorbar at the right place
    pos = ax.get_position()
    l, b, w, h = pos.bounds
    if cbar:
        if tit and label is not None:
            cax = fig.add_axes([0.9, 0.46-0.7*h/2., 0.03, 0.7*h])
        else:
            cax = fig.add_axes([0.9, 0.51-0.7*h/2., 0.03, 0.7*h])
        mir = fig.colorbar(im, cax=cax)

    # Normalise around zero
    if normed:
        im.set_clim(-max(abs(data.max()), abs(data.min())),
                     max(abs(data.max()), abs(data.min())))

    #To avoid white lines on pdfs

    for c in im.collections:
        c.set_edgecolor("face")

    return fig

def radialContour_data(data, dlon, dlat, dval, rad=0.85, label=None, proj='hammer', lon_0=0., vmax=None,
                  vmin=None, lat_0=30., levels=32, cm='seismic',
                  normed=True, cbar=True, tit=True, lines=False):

    nphi, ntheta = data.shape

    phi = np.linspace(-np.pi, np.pi, nphi)
    theta = np.linspace(np.pi/2, -np.pi/2, ntheta)
    pphi, ttheta = np.mgrid[-np.pi:np.pi:nphi*1j, np.pi/2.:-np.pi/2.:ntheta*1j]
    lon2 = pphi * 180./np.pi
    lat2 = ttheta * 180./np.pi

    circles = np.r_[-60., -30., 0., 30., 60.]
    delon = 60.
    meridians = np.arange(-180+delon, 180, delon)

    if proj == 'moll' or proj == 'hammer':
        if tit and label is not None:
            if cbar:
                fig = plt.figure(figsize=(9,4.5))
                ax = fig.add_axes([0.01, 0.01, 0.87, 0.87])
            else:
                fig = plt.figure(figsize=(8,4.5))
                ax = fig.add_axes([0.01, 0.01, 0.98, 0.87])
            ax.set_title('%s: r/ro = %.3f' % (label, rad),
                         fontsize=24)
        else:
            if cbar:
                fig = plt.figure(figsize=(9,4))
                ax = fig.add_axes([0.01, 0.01, 0.87, 0.98])
            else:
                fig = plt.figure(figsize=(8,4))
                ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
            #tit1 = r'%.2f Ro' % rad
            #ax.text(0.12, 0.9, tit1, fontsize=16,
                  #horizontalalignment='right',
                  #verticalalignment='center',
                  #transform = ax.transAxes)
    else:
        if tit and label is not None:
            if cbar:
                fig = plt.figure(figsize=(6,5.5))
                ax = fig.add_axes([0.01, 0.01, 0.82, 0.9])
            else:
                fig = plt.figure(figsize=(5,5.5))
                ax = fig.add_axes([0.01, 0.01, 0.98, 0.9])
            ax.set_title('%s: r/ro = %.3f' % (label, rad),
                         fontsize=24)
        else:
            if cbar:
                fig = plt.figure(figsize=(6,5))
                ax = fig.add_axes([0.01, 0.01, 0.82, 0.98])
            else:
                fig = plt.figure(figsize=(5,5))
                ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
            tit1 = r'%.2f Ro' % rad
            ax.text(0.12, 0.9, tit1, fontsize=16,
                  horizontalalignment='right',
                  verticalalignment='center',
                  transform = ax.transAxes)

    if proj != 'hammer':
        from mpl_toolkits.basemap import Basemap
        map = Basemap(projection=proj, lon_0=lon_0, lat_0=lat_0,
                      resolution='c')
        map.drawparallels([0.], dashes=[2, 3], linewidth=0.5)
        map.drawparallels(circles, dashes=[2,3], linewidth=0.5)
        map.drawmeridians(meridians, dashes=[2,3], linewidth=0.5)
        map.drawmeridians([-180], dashes=[20,0], linewidth=0.5)
        map.drawmeridians([180], dashes=[20,0], linewidth=0.5)
        x, y = list(map(lon2, lat2))
    else:
        x, y = hammer2cart(ttheta, pphi)
        for lat0 in circles:
            x0, y0 = hammer2cart(lat0*np.pi/180., phi)
            ax.plot(x0, y0, 'k:', linewidth=0.7)
        for lon0 in meridians:
            x0, y0 = hammer2cart(theta, lon0*np.pi/180.)
            ax.plot(x0, y0, 'k:', linewidth=0.7)
        xxout, yyout  = hammer2cart(theta, -np.pi-1e-3)
        xxin, yyin  = hammer2cart(theta, np.pi+1e-3)
        ax.plot(xxin, yyin, 'k-')
        ax.plot(xxout, yyout, 'k-')
        ax.axis('off')

    cmap = plt.get_cmap(cm)

    if proj == 'ortho':
        lats = np.linspace(-90., 90., ntheta)
        dat = map.transform_scalar(data.T, phi*180/np.pi, lats,
                                   nphi, ntheta, masked=True)
        im = map.imshow(dat, cmap=cmap)
    else:
        if vmax is not None or vmin is not None:
            normed = False
            cs = np.linspace(vmin, vmax, levels)
            im = ax.contourf(x, y, data, cs, cmap=cmap, extend='both')
            if lines:
                ax.contour(x, y, data, cs, colors=['k'], linewidths=0.5,
                           extend='both', linestyles=['-'])
                #ax.contour(x, y, data, 1, colors=['k'])
            #im = ax.pcolormesh(x, y, data, cmap=cmap, antialiased=True)
        else:
            cs = levels
            im = ax.contourf(x, y, data, cs, cmap=cmap)
            if lines:
                ax.contour(x, y, data, cs, colors=['k'], linewidths=0.5,
                           linestyles=['-'])
            #im = ax.pcolormesh(x, y, data, cmap=cmap, antialiased=True)

    # Add obs atop projection
    if dval.size>1:
        for io in range(len(dval)):
            xd, yd = hammer2cart(np.pi/2 - dlat[io], dlon[io] - np.pi)    
            ax.scatter(xd, yd, c=dval[io], s=100, cmap=cmap, vmin=vmin, \
                      vmax=vmax, edgecolors='black')
    else:
        xd, yd = hammer2cart(np.pi/2 - dlat, dlon - np.pi)
        ax.scatter(xd, yd, c=dval, s=100, cmap=cmap, vmin=vmin, \
                   vmax=vmax, edgecolors='black')

    # Add the colorbar at the right place
    pos = ax.get_position()
    l, b, w, h = pos.bounds
    if cbar:
        if tit and label is not None:
            cax = fig.add_axes([0.9, 0.46-0.7*h/2., 0.03, 0.7*h])
        else:
            cax = fig.add_axes([0.9, 0.51-0.7*h/2., 0.03, 0.7*h])
        mir = fig.colorbar(im, cax=cax)

    # Normalise around zero
    if normed:
        im.set_clim(-max(abs(data.max()), abs(data.min())),
                     max(abs(data.max()), abs(data.min())))

    #To avoid white lines on pdfs

    for c in im.collections:
        c.set_edgecolor("face")

    return fig

