# data_mag_assim

Scripts and example files to manipulate paleo/archeo/geo and simulated magnetic data. The database consists of observatory data (all_obs.dat) used in gufm1 (Jackson et al., 2000) as well as paleo/archeomagnetic and historical data (results_complete_output.tsv) from the [Histmag](https://cobs.zamg.ac.at/data/index.php/en/data-access/histmag) platform.

## Usage

Prepare separated bynary files for Paleo/Archeo, Historic, and Observatory data with
```bash
python read_database_orig.py
```

The preparation of parody_pdaf input files from the binary database is done with
```bash
python make_parody_pdaf_input.py
```

## Additional features 
The quality of paleomagnetic modelling can be assessed according to the methodology introduced by Sprain et al. (EPSL, 2019). 
workflow.py reads in surface fields to produce a series of diagnostics for reversing dynamo simulations. 
