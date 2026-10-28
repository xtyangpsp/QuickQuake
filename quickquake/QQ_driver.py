#!/usr/bin/env python3
"""
QuickQuake: run pipeline in time chunks Config > Stations > Download > PhaseNet > GaMMA,
then run a separate merge script, then optionally run HypoXPy relocation, quality control and classification of volcanic signals 

This file holds only the settings: quickquake/QQ_run.py runs the steps with them

How to use it:
    1. Copy this file into a folder for your project (e.g. ~/research/my_region/).
    2. Set QUICKQUAKE_DIR below to the folder where QuickQuake is installed.
    3. Edit the settings in the CONFIGURATION section.
    4. Run it:   python QQ_driver.py
Results are written to a "data" folder next to this file.
"""

from pathlib import Path
import sys

# 
# CONFIGURATION
# 

QUICKQUAKE_DIR = "~/src/QuickQuake"   # folder where QuickQuake is installed (edit this)

START = "2021-07-29T08:00:00"
END   = "2021-07-30T00:00:00"
HOUR_STEP = 8

BASE_DIR  = Path(QUICKQUAKE_DIR).expanduser().resolve()   # QuickQuake code
DATA_ROOT = Path(__file__).resolve().parent / "data"      # results, next to this file
MODEL_DIR = BASE_DIR / "dependencies/PhaseNet/model/190703-214543"


RUN_CONFIG   =  True

# config options (user edits )
center   = (-161.8903, 55.4133) # central cooordinates for your region of interest 
deg      = 1.0                   # width (degrees) of the box around center where GaMMA searches for events
client   = "EARTHSCOPE_FEDERATOR"  # FDSN data center; "EARTHSCOPE_FEDERATOR" asks all data centers and adds
                                   # the backup centers below; a single center (e.g. "IRIS") also works
region   = "pavlof"

RUN_DL       = True
# download options (user edits )
# Every seismic station of every network within station_maxradius of center is used;
# the best channel set of each station is chosen from the data actually available.
station_maxradius = 2.0              # degrees around center
station_excluded_networks = ["SY"]   # networks to skip (SY = synthetic data)
station_excluded_stations = []       # stations to skip, as "NET.STA", e.g. ["DK.ILULI", "3Z.SK12G"]
station_min_sample_rate = 20.0       # ignore channels sampled below this rate (Hz)
station_target_rate = 100.0          # preferred sample rate when ranking channels (Hz)
station_min_coverage = 0.8           # minimum fraction of the chunk with data
download_max_retries = 3             # attempts after a download error
download_retry_sleep = 3             # seconds between attempts
download_fallback_clients = ["EARTHSCOPE", "GEOFON", "ETH"]  # backup data centers tried in order when the
                                     # federator has no data (GEOFON serves DK.GDH, ETH DK.KARAT)
download_resample_rate = 100.0       # save every channel at this rate (Hz), the rate PhaseNet was trained on;
                                     # None = keep the native rate of each channel

RUN_PHASENET = True
# phasenet options (user edits )
phasenet_min_p_prob = 0.30
phasenet_min_s_prob = 0.30
phasenet_mpd        = 50  # minimum peak distance
phasenet_batch_size = 20  # traces processed together (lower it if memory is short)
phasenet_highpass_filter = 0.0  # high-pass filter in Hz (0 = none)


RUN_GAMMA    = True
# gamma options (user edits )
gamma_method           = "BGMM"   # default: BGMM
gamma_oversample_factor = 30       # default: 30 for BGMM
gamma_min_picks_per_eq = 6        # default: 6
gamma_max_sigma11      = 2      # S
gamma_max_sigma22      = 1     #m/s
gamma_max_sigma12      = 1     #covariance
gamma_min_p_picks_per_eq = 0    # minimum P picks per event
gamma_min_s_picks_per_eq = 0    # minimum S picks per event
gamma_use_amplitude    = True   # use pick amplitudes as well as times
gamma_vp               = 6.0    # P velocity of GaMMA's uniform model (km/s)
gamma_vp_vs_ratio      = 1.73   # Vp/Vs of GaMMA's uniform model
gamma_depth_min_km     = 0      # shallowest event depth searched (km)
gamma_depth_max_km     = 60     # deepest event depth searched (km)
gamma_use_dbscan       = True   # False = all picks of a chunk in one job on 1 CPU (original method)
                                # True  = split picks into groups (DBSCAN) and associate them on several
                                #         CPUs; use it for many stations or chunks longer than ~1 hour
gamma_dbscan_eps       = "auto" # DBSCAN group size in seconds, or "auto" (from station spacing)
                                # "auto" can be too small for dense networks: picks of one earthquake (P and S)
                                # can span 10 s or more. Pavlof: "auto" gave 6.3 s and lost events; 15 s matched
                                # the original method.
gamma_dbscan_min_samples = 3    # minimum picks to start a group
gamma_ncpu             = None   # CPUs used with DBSCAN (None = automatic, at most all CPUs - 1)



RUN_MERGE_GAMMA = True

RUN_ARPICK = False                # Enable ar_pick repicking step

arpick_namebase = "GAMMA"         # Base name used in pipeline files
arpick_pre_p = 1.0                # Seconds before original P pick in local window
arpick_post_p = 2.0               # Seconds after original P pick in local window
arpick_pre_s = 1.5                # Seconds before original S pick in local window
arpick_post_s = 3.0               # Seconds after original S pick in local window
arpick_max_dt_p = 0.13            # Max allowed P repick shift from original pick
arpick_max_dt_s = 0.30            # Max allowed S repick shift from original pick
arpick_freqmin = 1.0              # Bandpass low cutoff before ar_pick
arpick_freqmax = 20.0             # Bandpass high cutoff before ar_pick

arpick_lta_p = 1.0                # Long-term window for P trigger
arpick_sta_p = 0.1                # Short-term window for P trigger
arpick_lta_s = 2.0                # Long-term window for S trigger
arpick_sta_s = 0.2                # Short-term window for S trigger
arpick_m_p = 2                    # AR model order for P
arpick_m_s = 8                    # AR model order for S
arpick_l_p = 0.1                  # P picker smoothing/control parameter
arpick_l_s = 0.2                  # S picker smoothing/control parameter

arpick_keep_debug_cols = True     # Save debug columns in output CSV

RUN_LOCATION      =  True
location_binpath = None   # folder with hyp1.40, hypoDD and ph2dt; None = the programs install.sh
                          # compiled into this conda environment (or ~/bin if they are not there)
location_namebase = "GAMMA"
location_extra_args = []   # e.g., ["--cleanup"]
location_p_model = "velo_p_rv_avo.cre"
location_s_model = "velo_s_rv_avo.cre"
location_ref_ele = 3.2 #reference location highest part in the topography 
location_depth_min = 0.0
location_depth_max = 20.0
location_depth_step = 0.5
location_templates_dir = BASE_DIR / "hypox_templates"  # HypoInverse, ph2dt and HypoDD templates (and models
                                                     # given by name); point it to your own copy to change them
location_min_nsta = 4         # minimum stations for a HypoInverse location
location_combine_net_sta = "yes"  # HypoDD station names: "yes" = NET.STA, "no" = STA,
                                  # "auto" = STA when NET.STA is longer than 7 characters (HypoDD limit)



RUN_QC= True
# qc options (user edits)
qc_min_total_stations = 3 #must be a positive number 
qc_min_valid_stations_per_v = None  # None = defaults to qc_min_total_stations
qc_vmin_curve   = 2.0
qc_vmax_curve   = 8.0
qc_vsteps_curve = 150
qc_winlen = 3
qc_noise_percentile  = 50.0
qc_signal_percentile = 90.0
qc_min_ratio = 5.0 
qc_energy_type = "squared_median" # other option is "envelope_median" or "squared_median"
qc_make_plot = True
qc_max_plots = 20 # number of QC figures; set to None to plot every event
qc_pre_s = 10.0       # seconds of waveform before origin time
qc_post_s = 40.0      # seconds of waveform after origin time
qc_freqmin = 1.0      # bandpass low corner (Hz)
qc_freqmax = 15.0     # bandpass high corner (Hz), clipped to Nyquist
qc_vmin_signal = 2.0  # lowest velocity of the signal band (km/s)
qc_vmax_signal = 8.0  # highest velocity of the signal band (km/s)
qc_pad_neighbors = True      # also read neighbor chunks when a window crosses a chunk boundary
qc_geom_correction = False   # multiply energy by distance^2
qc_max_events = 0            # test only the first N events (0 = all)

#
# RUN (no need to edit below: quickquake/QQ_run.py runs the steps with the settings above)
#

if __name__ == "__main__":
    if not (BASE_DIR / "quickquake" / "QQ_run.py").is_file():
        sys.exit(f"QuickQuake was not found in {BASE_DIR}. "
                 "Set QUICKQUAKE_DIR at the top of this file to the folder where QuickQuake is installed.")
    sys.path.insert(0, str(BASE_DIR / "quickquake"))
    from QQ_run import run
    run(globals())
