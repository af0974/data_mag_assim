import numpy as np
from data_class import *
from data_plots import *
import seaborn as sns

datToto = DataTable('results_compact_output.tsv', type='histmag_compact')
#print(datToto.source)
#mask = (datToto.source == "Brick")
#myyear = datToto.year[mask]
#mollweide_data_dist_sphere( datToto.lat[mask], datToto.lon[mask])
#print(len(myyear))
dataObs = DataTable('all_obs.dat', type='obsmag_gufm1')
print(len(dataObs.year))
#mollweide_data_dist_sphere( dataObs.lat, dataObs.lon)

datTot = datToto+dataObs
print(len(datTot.year))
#mollweide_data_dist_sphere( datTot.lat, datTot.lon)

fig = plt.figure(figsize=(5,5))
ax1 = fig.add_subplot(111)
#sns.distplot(datTot.year, hist = True, kde = False,
#                     kde_kws = {'shade': True, 'linewidth': 2},
#                     label = 'all_data', color='blue')
sns.distplot(datTot.year[datTot.origin==1], hist = True, kde = False,
                     kde_kws = {'shade': True, 'linewidth': 2},
                     label = 'archeo/volcanic', color='red')
sns.distplot(datTot.year[datTot.origin==0], hist = True, kde = False,
                     kde_kws = {'shade': True, 'linewidth': 2},
                     label = 'historic', color='blue')
sns.distplot(datTot.year[datTot.origin==2], hist = True, kde = False,
                     kde_kws = {'shade': True, 'linewidth': 2},
                     label = 'observatory', color='green')
ax1.legend(loc="best")
ax1.set_xlabel('year AD')
plt.tight_layout()
plt.yscale('log')
plt.show()


