# data_mag_assim

Scripts and example files to manipulate paleo/archeo/geo magnetic data. The database consists on observatory data (all_obs.dat) used in gufm1 (Jackson et al., 2000) as well as paleo/archeomagnetic and historical data (results_compact_output.tsv) from the [Histmag](https://cobs.zamg.ac.at/data/index.php/en/data-access/histmag) platform.

## Usage

Prepare separated bynary files for Paleo/Archeo, Historic, and Observatory data with
```bash
python read_database_orig.py
```

The preparation of parody_pdaf input files from the binary database is done with
```bash
`python make_parody_pdaf_input.py
```

