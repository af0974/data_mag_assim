import numpy as np
import pandas as pd

# Read histmag database
histmag = pd.read_csv("histmag_02_12_2021.tsv", sep='\t', \
        usecols = ['origin','year','posdyear', \
	'dyearCalc','lat','lon','decl','ddecl','inc','dinc', \
	'inten','dinten','source'])
histmag.dtypes
histmag.info()

# Unfold d, i and f data for new dataframe
header = ['origin', 'year', 'posdyear', 'dyearCalc','lat','lon', \
	'val', 'dval', 'source', 'elem']

d = histmag[(histmag['decl'].notnull())][['origin', 'year', \
	'posdyear', 'dyearCalc','lat','lon', 'decl', 'ddecl', 'source']]
d =  d.assign(elem='D')
d.columns = header
i = histmag[(histmag['inc'].notnull())][['origin', 'year', \
	'posdyear', 'dyearCalc','lat','lon', 'inc', 'dinc', 'source']]
i = i.assign(elem='I')
i.columns = header
f = histmag[(histmag['inten'].notnull())][['origin', 'year', \
	'posdyear', 'dyearCalc','lat','lon','inten', 'dinten', 'source']]
f = f.assign(elem='F')
f.columns = header
df = pd.concat([d, i, f])

# Update dyearCalc for numeric values
df['dyearC'] = 0
df.loc[df['dyearCalc'].str.contains('Estimate', na=False),'dyearC'] = 1
df.loc[df['dyearCalc'].str.contains('Deviation', na=False),'dyearC'] = 2
df = df.drop(columns = 'dyearCalc')

# Take average over the assimilation interval for time-series
dt = 1

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

# Calculate how much data, and give statistics about uncertainties

# Select data from a specific year 
time_i = 1799.5
time_f = 1800.5
df_dt = df[(df['year'] >= time_i) & (df['year'] <= time_f)]

# If no uncertainties, set one degree for now
df_dt[df_dt==""] = np.NaN
df_dt = df_dt.fillna(-999)
df_dt = df_dt.drop(df_dt[df_dt.posdyear > 100].index)
if time_f < 1800 :
   df_dt.loc[(df_dt.dval==-999.0) & ((df_dt.elem=='I') | (df_dt.elem=='D')), 'dval'] = 1.0
   df_dt.loc[(df_dt.dval==-999.0) & ((df_dt.elem=='F')), 'dval'] = 2000.0
else :
   df_dt.loc[(df_dt.dval==-999.0) & (df_dt.origin==1.0) & ((df_dt.elem=='I') | (df_dt.elem=='D')), 'dval'] = 1.0
   df_dt.loc[(df_dt.dval==-999.0) & (df_dt.origin==1.0) & ((df_dt.elem=='F')), 'dval'] = 2000.0

   df_dt.loc[(df_dt.dval==-999.0) & (df_dt.origin==0.0) & ((df_dt.elem=='I') | (df_dt.elem=='D')), 'dval'] = 0.8
   df_dt.loc[(df_dt.dval==-999.0) & (df_dt.origin==0.0) & ((df_dt.elem=='F')), 'dval'] = 200.0 

# Transform angles to radians and field to micro T

print(df_dt)
# Print data in parody format
cols = ['elem', 'posdyear', 'year', 'lon', 'lat', 'val', 'dval', 'origin']
df_dt[cols].to_csv('obs_time_1800_ns.txt', mode = 'w', sep=' ', header=None, index=False)

cols = ['elem', 'posdyear', 'year', 'lon', 'lat', 'val', 'dval', 'origin', 'source']
df_dt[cols].to_csv('obs_time_1800_ns_full.txt', mode = 'w', sep=' ', header=None, index=False)

