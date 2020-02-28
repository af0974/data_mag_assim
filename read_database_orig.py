import numpy as np
from data_class import *
from data_plots import *
import seaborn as sns
import pickle

# Reads databases Histmag (including Geomagia) and Obs from gufm1

dataHistArch = DataTable('results_complete_output.tsv', type='histmag_complete')
dataObs = DataTable('all_obs.dat', type='obsmag_gufm1')

# Separate datasets in terms of origin (archeo, hist, obs)

#print(len(dataHistArch.origin))
#print(type(dataHistArch))

dataHist, dataArch = dataHistArch.separate_origin()

# Save in hdf5 format

pickle.dump(dataArch, open("dataArch", "wb"))
pickle.dump(dataHist, open("dataHist", "wb"))
pickle.dump(dataObs, open("dataObs", "wb"))
