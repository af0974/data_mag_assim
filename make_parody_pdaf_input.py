import numpy as np
from data_class import *
from data_plots import *
import seaborn as sns
import pickle
from time_tools import *
import datetime

arch_data = pickle.load(open("dataArch", "rb"))
hist_data = pickle.load(open("dataHist", "rb"))
obs_data = pickle.load(open("dataObs", "rb"))

# Time in decimal years (only for hist data)
# not taking the hour into account yet, but when we do it, consider day-1 

year = hist_data.year[(~np.isnan(hist_data.month)) & (hist_data.month!=0.)]
month = hist_data.month[(~np.isnan(hist_data.month)) & (hist_data.month!=0.)]
day = hist_data.day[(~np.isnan(hist_data.month)) & (hist_data.month!=0.)]
source = hist_data.source[(~np.isnan(hist_data.month)) & (hist_data.month!=0.)]

# Dirty fix for problematic data - Patrick will update those in the future
day[16263] = 30.
day[16796] = 30.
month[85699] = 10.
day[86030] = 28.
day[88109] = 30.
day[119304] = 30.
day[120813] = 28.
day[121060] = 28.
day[123013] = 30.

year_new = np.zeros_like(year)
for i in range(0,len(year)):
    if (day[i]!=0.) & (~np.isnan(day[i])):  
       #print(i,year[i], month[i], day[i], source[i]) 
       year_new[i] = DecYear(year[i].astype(int), month[i].astype(int), day[i].astype(int)) 
    if (day[i]==0.) | (np.isnan(day[i])):
       #print(i,year[i], month[i], 1, source[i]) 
       year_new[i] = DecYear(year[i].astype(int), month[i].astype(int), 1)

hist_data.year[(~np.isnan(hist_data.month)) & (hist_data.month!=0.)] = year_new 

# Check observatory data given at hours: average over time?
# Check land and ship survey trajectories

# Gather whole database and sort by date

all_data = arch_data + hist_data + obs_data

# Time binning and averaging

dt = 10.
init = 1000.
fint = 2000.
delta = (fint - init)/ dt

# Remove data previous to init

all_data = all_data.select_indx(all_data.year > init)

# Remove data without lat lon and unc

all_data = all_data.select_indx(~np.isnan(all_data.lat))
all_data = all_data.select_indx(~np.isnan(all_data.lon))
all_data = all_data.select_indx(~np.isnan(all_data.sigma))

# Sort data with respect to time

all_data = all_data.sort_year()

bins = np.linspace(init+dt, fint, num = delta, endpoint = True, dtype = int)
inds = np.digitize(all_data.year, bins, right = True)

# Calculate ndata

ndata = np.zeros(len(bins))
for i in range(0,len(bins)):
    ndata[i] = sum(inds == i)

# Save input for Parody_pdaf

print(len(all_data.year))

f = open('obs_parody_pdaf.txt', 'w')
f.write("pointwise\n")
f.write("%i\n" % (len(bins)))
for i in range(0,len(bins)):
    f.write("%i %i %i\n" % (i+1, bins[i], ndata[i]))
for i in range(0,len(all_data.year)):
    f.write("%c %f %f %f %f %f\n" % ( \
           all_data.element[i], all_data.year[i], all_data.lon[i], \
           all_data.lat[i], all_data.value[i], all_data.sigma[i]))
f.close()

# Lat lon dec inc have to be changed from original with degrees to
# radians in parody format
