# data_mag_assim

Scripts and example files to manipulate paleo/archeo/geo and simulated magnetic data. The database consists of observatory data (all_obs.dat) used in gufm1 (Jackson et al., 2000) as well as paleo/archeomagnetic and historical data (results_complete_output.tsv) from the [HISTMAG](https://cobs.zamg.ac.at/data/index.php/en/models-and-databases/histmag) platform. This new version uses pandas in order to read and organize the data frame.

## Usage

Read the HISTMAG database (which contains Geomagia.v3), and produce compact and full outputs close to Parody_pdaf input format using
```bash
python read_db_time_series_nonreg_H.py
```

## Additional features 

The quality of paleomagnetic modelling can be assessed according to the methodology introduced by Sprain et al. (EPSL, 2019). 
workflow.py reads in surface fields to produce a series of diagnostics for reversing dynamo simulations, which are described in workflow_components.py. 
These diagnostics include the calculation of the Earthlikeness according to Christensen et al. (EPSL, 2010.) 


The workflow can be run in parallel (it requires the mpi4py library). An example of a submission script on the stella cluster of S-CAPAD (based on slurm) is given in submit_wf.job  
