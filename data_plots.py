import numpy as np
import cartopy.crs as ccrs
import cartopy.util as cutil
import matplotlib.pyplot as plt


def mollweide_data_dist_sphere(lat,lon,fname=None,Title=None):
    fig = plt.figure()
    ax = plt.axes( projection = ccrs.Mollweide(central_longitude=0))
    ax.set_global()
    ax.coastlines()
    ax.scatter( lon, lat, transform = ccrs.PlateCarree() )
    plt.tight_layout()
    plt.show()
