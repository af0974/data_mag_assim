import numpy as np
import pandas as pd

# Read histmag database
histmag = pd.read_csv("histmag_02_12_2021.tsv", sep='\t', \
        usecols = ['origin','year','posdyear', \
	'dyearCalc','lat','lon','decl','ddecl','inc','dinc', \
	'inten','dinten','source','inten_code'])
histmag.dtypes
histmag.info()

# Unfold d, i and f data for new dataframe
header = ['origin', 'year', 'posdyear', 'dyearCalc','lat','lon', \
	'val', 'dval', 'source', 'elem']

# Declination

d = histmag[(histmag['decl'].notnull())][['origin', 'year', \
	'posdyear', 'dyearCalc','lat','lon', 'decl', 'ddecl', 'source']]
d =  d.assign(elem='D')
d.columns = header

# Inclination

i = histmag[(histmag['inc'].notnull())][['origin', 'year', \
	'posdyear', 'dyearCalc','lat','lon', 'inc', 'dinc', 'source']]
i = i.assign(elem='I')
i.columns = header

# Total intensity

f = histmag[(histmag['inten'].notnull()) & 
        ((histmag['inten_code'].isnull()) | (histmag['inten_code'].str.contains('T')))]  \
	[['origin', 'year','posdyear', 'dyearCalc','lat','lon','inten', 'dinten', 'source']]
f = f.assign(elem='F')
f.columns = header

# Horizontal intensity

h = histmag[(histmag['inten'].notnull()) & (histmag['inten_code'].str.contains('H'))]  \
        [['origin', 'year','posdyear', 'dyearCalc','lat','lon','inten', 'dinten', 'source']]
h = h.assign(elem='H')
h.columns = header

df = pd.concat([d, i, f, h])

# Update dyearCalc for numeric values
df['dyearC'] = 0
df.loc[df['dyearCalc'].str.contains('Estimate', na=False),'dyearC'] = 1
df.loc[df['dyearCalc'].str.contains('Deviation', na=False),'dyearC'] = 2
df = df.drop(columns = 'dyearCalc')

# Take average over the assimilation interval for time-series
dt = 10.
dt2 = 5.

obs = df[df['source'].str.contains('observatory|monastery|mining', na=False)]
df = df.drop(df[df['source'].str.contains('observatory|monastery|mining', na=False)].index)

t_ini = np.floor(obs['year'].min()/dt)*dt
t_fin = np.ceil(obs['year'].max()/dt)*dt
delta = int((t_fin - t_ini)/ dt) + 1
t_obs = np.linspace(t_ini, t_fin, num = delta, endpoint =True)

# change year to nearest round
obs['year'] = (obs['year']/dt).apply(np.ceil)*dt

# Calculate mean and stdv
s = obs.groupby(['lat','lon','elem','year','posdyear','dyearC', \
	'origin','source']).agg(mean_val=('val', 'mean'), \
	std_val=('val', 'std'), mean_dval=('dval', 'mean'))	

# Compare existing uncertainties to the STD of the aggregation
# if uncertainties don't exist, substitute with STD
s['mean_dval'] = s['mean_dval'].fillna(s['std_val'])
s = s.reset_index()
s = s.drop(columns = 'std_val')
s = s.rename({'mean_val': 'val', 'mean_dval': 'dval'}, axis='columns')

# Add mean values to main dataframe
df = pd.concat([df, s], ignore_index=True)

# Time intervals (frequency has been increased between 1820 and 1860)
time_i = 1000.
time_ii = 1820.
time_ff = 1860.
time_f = 2000.

df_dt = df[(df['year'] > time_i) & (df['year'] <= time_f)]

# If no uncertainties, set one degree for now
df_dt[df_dt==""] = np.NaN
df_dt = df_dt.fillna(-999)

df_dt = df_dt.drop(df_dt[(df_dt.posdyear == -999) & (df_dt.origin==1.0)].index)

df_dt = df_dt.drop(df_dt[df_dt.posdyear > 100].index)

# For indirect data with no uncertainties, F unc 8.0 muT and 3.0 degrees for D and F

df_dt.loc[((df_dt.dval==-999.0) | (df_dt.dval==0)) & (df_dt.origin==1.0) \
                & ((df_dt.elem=='I')), 'dval'] = 3.0
df_dt.loc[((df_dt.dval==-999.0) | (df_dt.dval==0)) & (df_dt.origin==1.0) \
		& ((df_dt.elem=='D')), 'dval'] = 3.0
df_dt.loc[((df_dt.dval==-999.0) | (df_dt.dval==0)) & (df_dt.origin==1.0) \
		& ((df_dt.elem=='F')), 'dval'] = 8000.0
df_dt.loc[((df_dt.dval==-999.0) | (df_dt.dval==0)) & (df_dt.origin==1.0) \
                & ((df_dt.elem=='H')), 'dval'] = 8000.0

# Lower bounds for indirect data

df_dt.loc[(df_dt.dval<0.5) & (df_dt.origin==1.0) \
                & ((df_dt.elem=='I')), 'dval'] = 0.5
df_dt.loc[(df_dt.dval<0.5) & (df_dt.origin==1.0) \
                & ((df_dt.elem=='D')), 'dval'] = 0.5
df_dt.loc[(df_dt.dval<500.0) & (df_dt.origin==1.0) \
                & ((df_dt.elem=='F')), 'dval'] = 500.0
df_dt.loc[(df_dt.dval<500.0) & (df_dt.origin==1.0) \
                & ((df_dt.elem=='H')), 'dval'] = 500.0

# For historical data previous to 1800 AD, F unc 2.0 muT and 1.0 degree for D and I

df_dt.loc[(df_dt.year < 1800) & ((df_dt.dval==-999.0) | (df_dt.dval==0)) & (df_dt.origin==0.0) \
                & (df_dt.elem=='I'), 'dval'] = 1.3
df_dt.loc[(df_dt.year < 1800) & ((df_dt.dval==-999.0) | (df_dt.dval==0)) & (df_dt.origin==0.0) \
		& (df_dt.elem=='D'), 'dval'] = 1.7
df_dt.loc[(df_dt.year < 1800) & ((df_dt.dval==-999.0) | (df_dt.dval==0)) & (df_dt.origin==0.0) \
		& (df_dt.elem=='F'), 'dval'] = 2000.0
df_dt.loc[(df_dt.year < 1800) & ((df_dt.dval==-999.0) | (df_dt.dval==0)) & (df_dt.origin==0.0) \
                & (df_dt.elem=='H'), 'dval'] = 2000.0

# Lower bounds for direct data pre 1800

df_dt.loc[(df_dt.year < 1800) & (df_dt.dval<0.5) & (df_dt.origin==0.0) \
                & (df_dt.elem=='I'), 'dval'] = 0.5
df_dt.loc[(df_dt.year < 1800) & (df_dt.dval<0.5) & (df_dt.origin==0.0) \
                & (df_dt.elem=='D'), 'dval'] = 0.5
df_dt.loc[(df_dt.year < 1800) & (df_dt.dval<500.0) & (df_dt.origin==0.0) \
                & (df_dt.elem=='F'), 'dval'] = 500.0
df_dt.loc[(df_dt.year < 1800) & (df_dt.dval<500.0) & (df_dt.origin==0.0) \
                & (df_dt.elem=='H'), 'dval'] = 500.0

# For historical data after 1800 AD, F unc 0.22 muT and 0.8 degree for D and I

df_dt.loc[(df_dt.year >= 1800) & ((df_dt.dval==-999.0) | (df_dt.dval==0)) & (df_dt.origin==0.0) \
                & (df_dt.elem=='I'), 'dval'] = 1.0
df_dt.loc[(df_dt.year >= 1800) & ((df_dt.dval==-999.0) | (df_dt.dval==0)) & (df_dt.origin==0.0) \
		& (df_dt.elem=='D'), 'dval'] = 0.8
df_dt.loc[(df_dt.year >= 1800) & ((df_dt.dval==-999.0) | (df_dt.dval==0)) & (df_dt.origin==0.0) \
		& (df_dt.elem=='F'), 'dval'] = 220.0
df_dt.loc[(df_dt.year >= 1800) & ((df_dt.dval==-999.0) | (df_dt.dval==0)) & (df_dt.origin==0.0) \
                & (df_dt.elem=='H'), 'dval'] = 220.0

# Lower bounds for direct data post 1800

df_dt.loc[(df_dt.year >= 1800) & (df_dt.dval<0.3) & (df_dt.origin==0.0) \
                & (df_dt.elem=='I'), 'dval'] = 0.3
df_dt.loc[(df_dt.year >= 1800) & (df_dt.dval<0.3) & (df_dt.origin==0.0) \
                & (df_dt.elem=='D'), 'dval'] = 0.3
df_dt.loc[(df_dt.year >= 1800) & (df_dt.dval<200.0) & (df_dt.origin==0.0) \
                & (df_dt.elem=='F'), 'dval'] = 200.0
df_dt.loc[(df_dt.year >= 1800) & (df_dt.dval<200.0) & (df_dt.origin==0.0) \
                & (df_dt.elem=='H'), 'dval'] = 200.0

# Lower bounds for historical data

df_dt.loc[(df_dt.posdyear==-999.0) & (df_dt.origin==0.0), 'posdyear'] = 0.0

df_dt = df_dt.sort_values(by=['year'])

# Remove data with 0 uncertainty

# Transform angles to radians and field to micro T

delta_i = int((time_ii - time_i)/ dt)
delta_m = int((time_ff - time_ii)/ dt2)
delta_f = int((time_f - time_ff)/ dt)

bins_i = np.linspace(time_i+dt, time_ii, num = delta_i, endpoint = True)
bins_m =  np.linspace(time_ii+dt2, time_ff, num = delta_m, endpoint = True)
bins_f = np.linspace(time_ff+dt, time_f, num = delta_f, endpoint = True)

print(bins_i)
print(bins_m)
print(bins_f)

bins = np.concatenate((bins_i, bins_m, bins_f), axis=None)

print(bins)

inds = np.digitize(df_dt.year, bins, right = True)

print(inds)

print(df_dt)

# Calculate ndata

ndata = np.zeros(len(bins))
for i in range(0,len(bins)):
    ndata[i] = sum(inds == i)

print(sum(ndata))

# Print data in parody format

f = open('nobs_db_wHF.txt', 'w')
f.write("pointwise\n")
f.write("%i\n" % (len(bins)))
for i in range(0,len(bins)):
    f.write("%i %i %i\n" % (i+1, bins[i], ndata[i]))
f.close()

df_dt['indx'] = range(1, len(df_dt) + 1)

cols = ['elem', 'year', 'posdyear', 'dyearC', 'lon', 'lat', 'val', 'dval', 'origin', 'indx']
df_dt[cols].to_csv('obs_db_wHF.txt', mode = 'w', sep=' ', header=None, index=False)

cols = ['elem', 'year', 'posdyear', 'dyearC', 'lon', 'lat', 'val', 'dval', 'indx', 'source']
df_dt[cols].to_csv('full_obs_db_wHF.txt', mode = 'w', sep=' ', header=None, index=False)



