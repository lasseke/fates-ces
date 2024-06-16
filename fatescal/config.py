#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file to specify running options.
"""

# from typing import List
from pathlib import Path

'''
Specify site and data options.
'''

# OBS! Number of ensemble members must be divisible by 32 for now
N_ENSEMBLE_MEMBERS: int = 1
# Number of Kalman iterations. SET TO "0" FOR NON-ENKF EXPERIMENTS.
N_KALMAN_ITERATIONS: int = 0
# How the results should be aggregated and aligned, see
# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
AGGREGATION_FREQUENCY: str = '1D'

PROJECT_ROOT_PATH: Path = Path(__file__).parents[1]

CASE_NAME: str = f'HYY_default_{AGGREGATION_FREQUENCY}_gpp_et'

# Site to run the analysis for.
# Value must correspond to the folder name in '../data/sites/'
SITE_TO_ANALYZE: str = 'hyy'
OBS_DATA_FILE_NAME: str = 'HYY_EDDY233_GPP_ET_merged_allgapfilled.csv'
PATH_TO_OBS_DATASET: Path = PROJECT_ROOT_PATH / 'data' / \
    'sites' / SITE_TO_ANALYZE / 'processed'
OBS_DATE_COLUMN_NAME = 'samptime'

# Measured variables and CLM history fields to use for the data assimilation.
# Each dict adds an additional variable. Make sure unit conversions are
# implemented in './tools/unit_conversion.py'.
VARIABLES_TO_ASSIMILATE: list = [
    {  # Gross primary productivity (GPP)
        "CLM-FATES": {
            'history_var_name': 'FATES_GPP',
            'unit': 'kg C m-2 s-1'
        },
        "Observed": {  # Obs. derived from NEE -> add new history field?
            'csv_col_name': 'GPP',
            'unit': 'µmol CO2 m-2 s-1',
            'error': {  # Assumed observation error options for DA scheme
                # Univariate "normal" (Gaussian) distribution of mean 0 and variance 1.
                'distribution': 'standard_normal',
                'std': 3.78e-08  # Standard deviation, IN UNIT OF MEASUREMENT FREQUENCY
            }
        }
    },
    {  # Evaporation
        "CLM-FATES": {
            'history_var_name': 'QFLX_EVAP_TOT',  # -> Does Qle include Transp?
            'unit': 'kg H2O m-2 s-1',  # OR mm H2O/s  # 'W m-2'
        },
        "Observed": {
            'csv_col_name': 'ET_gapf',
            'unit': 'mmol H2O m-2 s-1',
            'error': {  # Assumed observation error options for DA scheme
                # Univariate "normal" (Gaussian) distribution of mean 0 and variance 1.'
                'distribution': 'standard_normal',
                'std': 1.31e-05,  # Standard deviation, IN UNIT OF MEASUREMENT FREQUENCY
            },
        }
    },
]

'''
Specify CTSM-FATES options needed within the DA workflow.
'''
# Project name to run on cluster
PROJECT_NAME: str = 'nn2806k'

# Root path for cases to create
CASES_ROOT_PATH: Path = PROJECT_ROOT_PATH / 'cases'
# Root path to forcing data (only supports default forcing for now)
INPUT_ROOT_PATH: Path = PROJECT_ROOT_PATH \
    / 'data' / 'case_input_data' / SITE_TO_ANALYZE.upper()
# Root path for storing output data
CESM_OUTPUT_ROOT_PATH: Path = PROJECT_ROOT_PATH \
    / 'data' / 'results' / 'model_output' / SITE_TO_ANALYZE.upper()
# Root path save dir for parameter jsons
DEFAULT_JSON_SAVE_DIR = PROJECT_ROOT_PATH / 'data' / 'results' / \
    'json' / CASE_NAME

# Name of model coupler/driver
MODEL_DRIVER: str = 'nuopc'
# '--res' option for './create.case'. 'CLM_USRDAT' for CLM
# 1-PT mode with input data extracted with 'subset_data.py'
CASE_RES: str = 'CLM_USRDAT'
# Model compset
CTSM_COMPSET = r'2000_DATM%GSWP3v1_CLM51%FATES_SICE_SOCN_SROF_SGLC_SWAV'

# FATES PARAMETERS - default indices and names below
''''
PFT_DICT: dict = {
    "1": True,  # broadleaf_evergreen_tropical_tree
    "2": True,  # needleleaf_evergreen_extratrop_tree
    "3": True,  # needleleaf_colddecid_extratrop_tree
    "4": True,  # broadleaf_evergreen_extratrop_tree
    "5": True,  # broadleaf_hydrodecid_tropical_tree
    "6": True,  # broadleaf_colddecid_extratrop_tree
    "7": True,  # broadleaf_evergreen_extratrop_shrub
    "8": True,  # broadleaf_hydrodecid_extratrop_shrub
    "9": True,  # broadleaf_colddecid_extratrop_shrub
    "10": True,  # arctic_c3_grass
    "11": True,  # cool_c3_grass
    "12": True,  # c4_grass
}
'''
# Total number of PFTs in default file
N_PFTS: int = 12
# Determined by hand from model output - these PFTs (indices) are prescribed in SP mode
INCLUDED_PFT_INDICES: list = [2, 6, 11]

# Specify model parameters that will be adapted to fit the data.
# See FATES default parameter file and the documentation for valid options
# OBS: only 'generalized_logit' priors are currently implemented in the full automation

FATES_PARAMS_TO_PERTURBATE: dict = {
    'fates_leaf_vcmax25top': {
        'prior': {
            'dist_name': 'generalized_logit',  # Must be 'generalized_logit' for now!
            'location': 'default_value',
            # Standard deviation AS FRACTION OF DEF VALUE. Only for 'dist_name'=='normal'
            'sd_frac': '',  # DEPRECATED!
            'scale': [0.3]*N_PFTS,  # Scale of 1.75 -> maximizes entropy, approximated uniform distribution
            # Default +/- 75%
            'upper_bound': [87.5, 108.5, 68.25, 106.75, 71.75, 101.5, 108.5, 94.5, 94.5, 136.5, 136.5, 136.5],
            'lower_bound': [12.5, 15.5, 9.75, 15.25, 10.25, 14.5, 15.5, 13.5, 13.5, 19.5, 19.5, 19.5],
        },
        'default_vals': [50, 62, 39, 61, 41, 58, 62, 54, 54, 78, 78, 78],
        'dimensions': 'n_pfts',  # Currently only implemented for 'n_pft' dimension vars
        'label': 'vcmax'  # Currently not used
    },
    'fates_leaf_stomatal_slope_ballberry': {
        'prior': {
            'dist_name': 'generalized_logit',
            'location': 'default_value',
            'sd_frac': '',
            'scale': [0.3]*N_PFTS,
            'upper_bound': [20]*N_PFTS,
            'lower_bound': [1]*N_PFTS,
        },
        'default_vals': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        'dimensions': 'n_pfts',
        'label': 'slope_bb'
    },
}

# Currently not implemented: other fixed FATES parameter changes
""" FATES_FIXED_PARAM_CHANGES: dict = {
    'fates_leaf_vcmax25top': []
} """

# 10 years are spinup period, 10 years will be analyzed
N_SPINUP_YEARS: int = 10
N_SIMULATION_YEARS: int = 10
FORCING_YR_START: int = 2004
FORCING_YR_END: int = 2013

CLM_XML_CHANGES: dict = {
    "CIME_OUTPUT_ROOT": str(CESM_OUTPUT_ROOT_PATH),
    # Path to input data from subset_data.py
    "CLM_USRDAT_DIR": str(INPUT_ROOT_PATH),
    "NTASKS": str(1),  # IMPORTANT! Only one task for single-point mode
    "DATM_MODE": "CLMGSWP3v1",  # Default GSWP3 atmospheric forcing
    "CCSM_CO2_PPMV": "389.8",  # Specifies constant ambient CO2 conc.
    "DEBUG": 'FALSE',  # Debug mode?
    "CALENDAR": 'NO_LEAP',  # GSWP3 uses a special calendar
    "STOP_N": str(N_SIMULATION_YEARS + N_SPINUP_YEARS),  # Set number of simulation years
    "STOP_OPTION": "nyears",
    # Relevant for time label in CLM
    "RUN_STARTDATE": f'{FORCING_YR_START - N_SPINUP_YEARS}-01-01',
    "DATM_YR_START": str(FORCING_YR_START),
    "DATM_YR_END": str(FORCING_YR_END),
    "DATM_YR_ALIGN": str(FORCING_YR_START),
    "LND_TUNING_MODE": 'clm5_1_GSWP3v1',  # Ensures good GSWP3 init. cond.
    "JOB_WALLCLOCK_TIME": '03:59:00',  # HPC job max. time (HH:MM:SS)
    "CLM_FORCE_COLDSTART": 'on',  # Start from bare ground!
}

CLM_NAMELIST_CHANGES: dict = {
    "use_fates": True,  # ... technically not needed with FATES compset
    "use_fates_sp": True,
    "use_fates_nocomp": True,
    "use_fates_planthydro": False,
    "use_bedrock": True,
    "hist_fincl2": [  # Add history variables with custom output frequency
        x['CLM-FATES']['history_var_name'] for x in VARIABLES_TO_ASSIMILATE
    ],
    "hist_nhtfrq": [0, -24],  # daily output in second hist tape
    "hist_mfilt": [12, 30]  # 1 history file per 30 days
}

'''
Specify more calibrate options
'''
# Number of folds for cross-validation. 1 Means no CV.
N_CV_FOLDS: int = 1

'''
Specify emulator model options.
'''
# Name of the MCMC negative log-likelihood emulator
# OBS: MUST BE 'gaussian_process_regressor' FOR NOW.
EMULATOR_MODEL_NAME: str = 'gaussian_process_regressor'
# ONLY IF EMULATOR_MODEL_NAME == 'gaussian_process_regressor':
# Inflate prediction by GPRs estimated uncertainty standard deviation?
SCALE_NLL_BY_GPR_STD: bool = True
# If yes, by which factor?
GPR_STD_PENALTY_FACTOR: float = 1.0

# Fraction of data (i.e., ensemble outputs) used for training
EMULATOR_TRAIN_FRACTION: float = 1.0

'''
Specify MCMC (Markov Chain Monte Carlo) options
'''
# Markov chain length
MCMC_CHAIN_LENGTH: int = 200_000

# Assumed observation error (TODO: improve later)
SIGMA_P: float = 0.1

# Adaptive Metropolis Hastings?
ADAPTIVE_MCMC: bool = True

# Burn-in fraction: discard first `BURN_IN_FRACTION * MCMC_CHAIN_LENGTH`
# elements in MCMC chain (which are considered not sufficiently tuned)
BURN_IN_FRACTION: float = 0.1

# Number of samples to draw from pruned chain for final eval. simulation
N_MCMC_SAMPLES_FOR_SIMULATION: int = 128
