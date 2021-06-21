import sys
import configparser
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from workflow_components import *

#
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
if len(sys.argv) ==1: 
	config_file = "config_revproc.ini" # default name for configuration file
else:
	config_file = str(sys.argv[1])
config = configparser.ConfigParser(interpolation=None)
config.read(config_file)
ier = comm.Barrier()
rescaling_done = config['Common'].getboolean('rescaling_done')
gauss_done = config['Common'].getboolean('gauss_done')
SHB_plot_done = config['Common'].getboolean('SHB_plot_done')
outdir = config['Common']['output_directory']
if rescaling_done is not True:
    get_rescaling_factors( comm, size, rank, config_file)
ier = comm.Barrier()
if gauss_done is not True:
    make_gauss_history( comm, size, rank, config_file)
ier = comm.Barrier()
if SHB_plot_done is not True:
    prepare_SHB_plot( comm, size, rank, config_file)
ier = comm.Barrier()
config.read(config_file)
if config['Diags'].getboolean('rms_intensity') is True:
    time, F_rms, gauss_unit, time_unit = get_rms_intensity( comm, size, rank, config_file)
ier = comm.Barrier()
if config['Diags'].getboolean('QPM') is True:
    QPM_results = compute_QPM(comm, size, rank, config_file)
ier = comm.Barrier()
if config['Diags'].getboolean('chi2') is True:
	chi2_value = compute_chi2(comm, size, rank, config_file)
ier = comm.Barrier()
if config['Diags'].getboolean('pole_latitude') is True:
    time, pole_lat, time_unit = get_pole_latitude( comm, size, rank, config_file)
ier = comm.Barrier()
if config['Diags'].getboolean('eccentricity') is True:
    time, sc, zc, time_unit = get_eccentricity( comm, size, rank, config_file)
ier = comm.Barrier()
config.read(config_file)
if config['Diags'].getboolean('transitional_field') is True: 
    analyze_transitional_field( comm, size, rank, config_file)
