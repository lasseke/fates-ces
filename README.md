# Inferring parameters in a land surface model by combining data assimilation and machine learning

Authors: Lasse T. Keetz, Kristoffer Aalstad, Rosie A. Fisher

Special thanks to: Matvey Vladimirovich Debolskiy, Kaveh Karimi-Asli, the [CTSM](https://github.com/ESCOMP/CTSM), [FATES](https://github.com/NGEET/fates), and [NorESM](https://github.com/NorESMhub/NorESM) development teams, [Cleary et al. (2021)](https://doi.org/10.1016/j.jcp.2020.109716), [FLUXNET](https://doi.org/10.18140/FLX/1440158), and the [SMEAR II](https://etsin.fairdata.fi/dataset/d1dd1e15-7de1-414d-b9d4-47d03bd68272) research network.

Code accompanying the research paper preliminarliy titled "Inferring parameters in a complex land surface model by combining data assimilation and machine learning" by Keetz et al. (in prep.). Adapts the approximate Bayesian inference framework termed "calibrate, emulate, sample" introduced by [Cleary et al. (2021)](https://doi.org/10.1016/j.jcp.2020.109716). Uncertainty-aware land surface model parameter estimation based on evapotranspiration and gross primary production flux observations. As a proof of concept, we assimilate single-site CLM-FATES outputs with (synthetic and real) eddy-covariance observations from a boreal forest site in southern Finland (Hyytiälä). The code currently relies on some hard-coded quick fixes and machine specific installations - use at own risk.

## 1 Installation instructions

Clone the repository.

```
git clone https://github.com/lasseke/fates-ces.git
```

### 1.1 Create and activate virtual environment

Install [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) or [Anaconda](https://docs.anaconda.com/free/anaconda/install/). Within the `fates-ces` directory, create a virtual environment with the Python dependencies via:

```
# Optional: module load [name_of_anaconda_module]
conda env create -f conda_env.yaml [-p env/installation/path]
source activate fatescal-env [or env/installation/path]
```

This will also install the code in the repo's `fatescal` package in development mode.

### 1.2 (Optional) Install model

Edit the `model_machine_config.txt` file to specify the CLM-FATES model version and other details. Then run:

```
# Assumes that git is installed and loaded!
python3 install_model.py
# On SIGMA-2 machines, execute install_model_with_modules.py for convenience
```

Note that your machine must be configured to run CLM-FATES. See https://escomp.github.io/ctsm-docs/versions/release-clm5.0/html/users_guide/overview/quickstart.html and the links therein to get started.

## 2 Running instructions

After successfully installing the software and its dependencies, several steps need to be completed before the parameter estimation can be executed. First, you need to create the CLM-FATES input and climate forcing data - refer to [this shell script](https://github.com/lasseke/fates-ces/tree/main/data/sites/hyy/forcing_data_creation) written for the [create_data.py](https://github.com/NorESMhub/ctsm-api/blob/main/data/create_data.py) tool for inspiration. Next, you need to prepare the observational data that should be assimilated. Currently, this needs to be a csv file with a [pandas.to_datetime()](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html)-compatible DateTime column (required to synchronize and resample the data) and one additional column per variable for the corresponding observations (see the [Hyytiälä example files](https://github.com/lasseke/fates-ces/tree/main/data/sites/hyy/processed)). The units and desired aggregation frequency must be specified in the config file (see below).

Many important workflow settings must be specified in the [config.py file](https://github.com/lasseke/fates-ces/blob/main/fatescal/config.py), such as the number of ensemble members, the path to the observational data file, the CTSM compset, etc. Follow the file layout specified in [this example file](https://github.com/lasseke/fates-ces/blob/main/fatescal/config.py).

### 2.1 Generate priors

Specify the prior settings (name of distribution, lower and upper bounds, default values, etc.) in `fatescal/config.py`. Then navigate to `scripts/python/cesm_ensemble_runs/` and execute:

```
python3 01_generate_fates_priors.py -n [name_stem_of_output_files]
# For example:
python3 01_generate_fates_priors.py -n vcmax_bbslope_pm50percent
```

This will randomly draw `N_ENSEMBLE_MEMBERS` parameter sets from the specified prior distributions and store them in different file formats (.csv, .json, .pkl).

### 2.2 Run CES parameter estimation with CLM-FATES ensemble runs

In `scripts/python/cesm_ensemble_runs/`, run

```
python3 04_calibrate.py -n [name_stem_of_parameter_files]
```

where `[name_stem_of_parameter_files]` must correspond to the name stem chosen in step 2.1 (e.g.: `vcmax_bbslope_pm50percent`). 

This script should then run all the required CES steps with the options defined in `config.py`. I.e.:

- Create, build, and submit a CLM-FATES multi-instance case with the specified model settings
- Wait for the ensemble simulations to finish on the HPC cluster
- Concatenate model outputs and align them with the observations
- Iteratively update the parameter sets (generated from the specified prior options) using the Iterative Ensemble-Kalman Smoother (IEnKS); within each iteration, rerun CLM-FATES with the updated parameters
- After the IEnKS updates are finished, train a machine learning emulator (Gaussian Process Regressor) on the resulting model-observation mismatches (-> negative log-likelihoods)
- Use the emulator to replace CLM-FATES as the forward model within an adaptive Markov Chain Monte Carlo algorithm -> enable systematic uncertainty quantification with the powerful MCMC method
- Create a final ensemble based on samples from the Markov chain and rerun CLM-FATES once again for evaluation

On Fram, building a multi-instance case with 128 ensemble members took ~1 h. Running a 20-year experiment (10 years for spin-up, 10 years for analysis, concatenating outputs, performing parameter updates) in Hyytiälä took ~3 hours per Kalman iteration.

### _Optional:_ Explore outputs

Refer to the `./notebooks` directory for some result analysis examples.
