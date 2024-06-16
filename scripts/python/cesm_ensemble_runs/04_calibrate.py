#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a full CES workflow using previously created parameter ensemble.
"""

import argparse
import glob
import pickle
import copy
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fatescal.ces import daalgorithms as da
from fatescal.tools import unit_conversion as uc
from fatescal.modelhandling.parameter_config import create_dummy_fates_ensemble
from fatescal.helpers import run_subprocess, setup_logging, wait_for_model
from fatescal.config import PROJECT_ROOT_PATH, CASE_NAME, \
    VARIABLES_TO_ASSIMILATE, N_SIMULATION_YEARS, FORCING_YR_START, \
    FATES_PARAMS_TO_PERTURBATE, INCLUDED_PFT_INDICES, N_PFTS, \
    N_CV_FOLDS, AGGREGATION_FREQUENCY, PATH_TO_OBS_DATASET, \
    OBS_DATE_COLUMN_NAME, OBS_DATA_FILE_NAME, \
    SITE_TO_ANALYZE, N_KALMAN_ITERATIONS, N_MCMC_SAMPLES_FOR_SIMULATION

# Tell pandas to display more float digits to explore small values
pd.options.display.float_format = '{:,.10f}'.format

# Wildcards for default naming in 'aligned_obs_mode' - should be no need to change
OBSERVATION_FNAME_GLOB = 'O_*.csv'
MODEL_FNAMES_GLOB = 'M_*.csv'

parser = argparse.ArgumentParser()

parser.add_argument(
    '-n', '--prior-file-stem-name',
    type=str,
    required=True,
    help='File stem name (wo/ file ending) of the prior csv/json/ensemble files.'
)

# Configure logger
logger = setup_logging('calibrate')


def calculate_aggregated_error(
    obs_target_df: pd.DataFrame
) -> pd.DataFrame:
    # Read initial input observation data
    obs_initial_timesteps_df = pd.read_csv(
        PATH_TO_OBS_DATASET / OBS_DATA_FILE_NAME
    )

    # Convert to datetime and UTC timezone
    obs_initial_timesteps_df[OBS_DATE_COLUMN_NAME] = pd.to_datetime(
        obs_initial_timesteps_df[OBS_DATE_COLUMN_NAME],
        utc=True
    ).dt.tz_localize(None)  # Remove timezone information

    # Subset to period of aggregated/harmonized observations
    start_date = min(obs_target_df[OBS_DATE_COLUMN_NAME])
    end_date = max(obs_target_df[OBS_DATE_COLUMN_NAME])
    date_mask = (obs_initial_timesteps_df[OBS_DATE_COLUMN_NAME] >= start_date) & \
        (obs_initial_timesteps_df[OBS_DATE_COLUMN_NAME] <= end_date)
    obs_initial_timesteps_df = obs_initial_timesteps_df.loc[date_mask]

    # Determine how many observations are in each aggregated period
    obs_count_df = obs_initial_timesteps_df.set_index(
        OBS_DATE_COLUMN_NAME
    ).resample(AGGREGATION_FREQUENCY).count()

    # Calculate scaled error according to formula:
    # for each included target variable:
    # error_std_aggregated = error_std_original / \
    # sqrt(mean(n_valid_measurements_in_aggregation_period))
    aggregated_errors_df = pd.DataFrame(
        columns=obs_count_df.columns
    )
    obs_error_dict_list = [x['Observed']['error']
                           for x in VARIABLES_TO_ASSIMILATE]
    for var_name, error_dict in zip(obs_count_df.columns, obs_error_dict_list):

        aggregated_errors_df.loc["Initial", var_name] = \
            error_dict['std']

        aggregated_errors_df.loc[AGGREGATION_FREQUENCY, var_name] = \
            error_dict['std'] / np.sqrt(np.mean(
                obs_count_df[var_name]
            ))

        # TODO: make generic
        # Convert to [g d-1]
        aggregated_errors_df.loc[:, var_name] = uc.convert_unit(
            values=aggregated_errors_df.loc[:, var_name].values,
            unit_in='kg s-1',
            unit_out='g d-1'
        )

    logger.info(f"Initial and aggregated errors: {aggregated_errors_df}")

    return aggregated_errors_df


def main() -> None:

    args = parser.parse_args()

    # File name stem of parameter ensemble (pkl, csv, json)
    file_stem_name = args.prior_file_stem_name
    aggregated_errors_df = None

    for cur_kalman_iter in range(N_KALMAN_ITERATIONS + 1):

        if cur_kalman_iter == 0:
            logger.info("Starting prior run.")
        else:
            logger.info(
                f"Starting Kalman iteration {cur_kalman_iter} of {N_KALMAN_ITERATIONS}."
            )

        # RUN MODEL

        logger.info("Creating multicase...")

        run_subprocess(
            cmd=[
                f"python3 02_create_multi_case.py -f {file_stem_name}.json -k {cur_kalman_iter}"],
            cwd=PROJECT_ROOT_PATH / 'scripts' / 'python' / 'cesm_ensemble_runs',
            shell=True,
        )

        logger.info("Submitting case...")

        run_subprocess(
            cmd=[f"./case.submit"],
            cwd=PROJECT_ROOT_PATH / 'cases' /
            f'{CASE_NAME}_k{cur_kalman_iter}',
            shell=True,
        )

        time.sleep(5)  # Sleep for 5 seconds

        print("Waiting for model to finish...", end="")

        wait_for_model()

        logger.info("Concatenating outputs...")

        # Path to default model outputs (one above hist folder)
        nc_root_path = PROJECT_ROOT_PATH / 'data' / 'results' / \
            'model_output' / SITE_TO_ANALYZE.upper() / \
            f"{CASE_NAME}_k{cur_kalman_iter}" / 'archive' / 'lnd'

        run_subprocess(
            cmd=[
                f"python3 concat_nc_output.py -d {nc_root_path / 'hist'} -o {nc_root_path} -k {cur_kalman_iter}"
            ],
            cwd=PROJECT_ROOT_PATH / 'scripts' / 'python',
            shell=True,
        )

        logger.info("Aligning model outputs with observations...")
        run_subprocess(
            cmd=[
                f"python3 03_prepare_data_assim.py -mp {nc_root_path} -ht 1 -a {AGGREGATION_FREQUENCY} -k {cur_kalman_iter}"
            ],
            cwd=PROJECT_ROOT_PATH / 'scripts' / 'python' / 'cesm_ensemble_runs',
            shell=True,
        )

        # Break loop here if last iteration, no more param update for final run
        if cur_kalman_iter == N_KALMAN_ITERATIONS:

            logger.info(f"\n\n{'*'*10} CALIBRATION OR PRIOR RUN FINISHED {'*'*10}\n\n")

            # Start emulator MCMC
            logger.info("Starting emulator MCMC...")

            run_subprocess(
                cmd=[
                    f"python3 05_emulate_sample.py -n {file_stem_name}"
                ],
                cwd=PROJECT_ROOT_PATH / 'scripts' / 'python' / 'cesm_ensemble_runs',
                shell=True,
            )

            # CREATE ENSEMBLE RUN FROM MCMC CHAIN

            # Load generated MCMC chain
            mcmc_chain_path = PROJECT_ROOT_PATH / 'data' / 'results' / \
                'mcmc_chains' / CASE_NAME / "mcmc_chain_pruned.csv"
            mcmc_chain_pruned_df = pd.read_csv(
                mcmc_chain_path, index_col=False
            )

            # Draw predefined random samples from the chain
            mcmc_chain_sample_df = mcmc_chain_pruned_df.sample(
                n=N_MCMC_SAMPLES_FOR_SIMULATION,
                random_state=11
            )

            # Delete full chain to free memory
            del mcmc_chain_pruned_df

            # Create dummy ensemble object
            mcmc_ensemble = create_dummy_fates_ensemble(
                n_ensembles=N_MCMC_SAMPLES_FOR_SIMULATION
            )

            # Replace parameter values in dummy ensemble with MCMC results
            for parameter_name in mcmc_chain_sample_df.columns:
                mcmc_ensemble.theta_df[parameter_name] = \
                    mcmc_chain_sample_df[parameter_name].values

            # Save to different formats
            mcmc_ensemble.save_as_csv(
                file_name=file_stem_name+'.csv',
                mcmc=True
            )
            mcmc_ensemble.save_as_pkl(
                file_name=file_stem_name+'.pkl',
                mcmc=True
            )
            mcmc_ensemble.write_to_json(
                file_name=file_stem_name+'.json',
                mcmc=True
            )

            # Create case from MCMC chain
            logger.info("Creating MCMC multicase...")

            run_subprocess(
                cmd=[
                    f"python3 02_create_multi_case.py -f {file_stem_name}.json -k {N_KALMAN_ITERATIONS} -i {N_MCMC_SAMPLES_FOR_SIMULATION} --mcmc"
                ],
                cwd=PROJECT_ROOT_PATH / 'scripts' / 'python' / 'cesm_ensemble_runs',
                shell=True,
            )

            logger.info("Submitting case...")

            run_subprocess(
                cmd=["./case.submit"],
                cwd=PROJECT_ROOT_PATH / 'cases' / f'{CASE_NAME}_mcmc',
                shell=True,
            )

            time.sleep(5)  # Sleep for 5 seconds

            print("Waiting for model to finish...", end="")

            wait_for_model()

            logger.info("Concatenating outputs...")

            # Path to default model outputs (one above hist folder)
            nc_root_path = PROJECT_ROOT_PATH / 'data' / 'results' / \
                'model_output' / SITE_TO_ANALYZE.upper() / \
                f"{CASE_NAME}_mcmc" / 'archive' / 'lnd'

            run_subprocess(
                cmd=[
                    f"python3 concat_nc_output.py -d {nc_root_path / 'hist'} -o {nc_root_path} -k {N_KALMAN_ITERATIONS} --mcmc"
                ],
                cwd=PROJECT_ROOT_PATH / 'scripts' / 'python',
                shell=True,
            )

            logger.info("Aligning model outputs with observations...")
            run_subprocess(
                cmd=[
                    f"python3 03_prepare_data_assim.py -mp {nc_root_path} -ht 1 -a {AGGREGATION_FREQUENCY} -k {N_KALMAN_ITERATIONS} --mcmc"
                ],
                cwd=PROJECT_ROOT_PATH / 'scripts' / 'python' / 'cesm_ensemble_runs',
                shell=True,
            )

            logger.info("\nFINISHED WITH MCMC SIMULATION. GOODBYE.\n")

            # BREAK HERE
            return

        # START DATA ASSIMILATION

        kalman_iter_dir = f'kalman_iter_{cur_kalman_iter}'

        # Update directory names
        harmonized_targets_dir = PROJECT_ROOT_PATH / 'data' / 'results' \
            / 'aligned_obs_model' / CASE_NAME / kalman_iter_dir
        param_ensemble_pkl_fpath = PROJECT_ROOT_PATH / 'data' / 'results' \
            / 'fates_param_ensembles' / CASE_NAME / 'pkl' / \
            kalman_iter_dir / (file_stem_name + '.pkl')

        # Check if harmonized_targets_dir exists
        if not (harmonized_targets_dir := Path(harmonized_targets_dir)).is_dir():
            raise ValueError(f"'{harmonized_targets_dir}' does not exist!")

        # Observed target variables
        obs_var_names = [x['Observed']['csv_col_name']
                         for x in VARIABLES_TO_ASSIMILATE]

        model_var_names = \
            [x['CLM-FATES']['history_var_name']
             for x in VARIABLES_TO_ASSIMILATE]

        n_target_variables = len(VARIABLES_TO_ASSIMILATE)

        # Read observations
        observation_csv_fpath = Path(
            glob.glob(str(harmonized_targets_dir / OBSERVATION_FNAME_GLOB))[0]
        )
        obs_target_df = pd.read_csv(observation_csv_fpath)
        # Date column to datetime
        obs_target_df[OBS_DATE_COLUMN_NAME] = pd.to_datetime(
            obs_target_df[OBS_DATE_COLUMN_NAME]
        )

        # TODO: Make non case specific
        # CONVERT TO [g C d-1]
        for var in obs_var_names:
            obs_target_df[var] = uc.convert_unit(
                values=obs_target_df[var],
                unit_in='kg s-1',
                unit_out='g d-1'
            )

        # Drop NA in observations, and store indices to also drop from modelled
        # TODO: UPDATE, DON'T NEED TO THROW AWAY OBS FOR ALL VARS IF ONLY ONE IS MISSING
        obs_na_mask = pd.isnull(obs_target_df).any(axis=1)
        obs_target_df = obs_target_df.dropna(axis='index', how='any')

        # Read model outputs
        model_out_csv_fpath_list = sorted(glob.glob(
            str(harmonized_targets_dir / MODEL_FNAMES_GLOB)
        ))

        model_out_df_list = []
        for fname in model_out_csv_fpath_list:

            cur_df = pd.read_csv(fname)
            # Convert time column to datetime. TODO: automate
            cur_df['time'] = pd.to_datetime(cur_df['time'])

            # Remove entries where observations are NA
            cur_df = cur_df.loc[~obs_na_mask]

            # CONVERT TO [g C d-1]
            for var in model_var_names:
                cur_df[var] = uc.convert_unit(
                    values=cur_df[var],
                    unit_in='kg s-1',
                    unit_out='g d-1'
                )

            model_out_df_list.append(cur_df)

        # Simple format check
        if obs_target_df.shape != model_out_df_list[0].shape:
            raise ValueError(
                "Observations and model outputs have different dimensions!"
            )

        if not all(obs_target_df[OBS_DATE_COLUMN_NAME] == model_out_df_list[0]['time']):
            raise ValueError(
                "Datetimes of observations and model outputs are different!"
            )

        print("Succesfully read observations and model data.")
        print("Shape observation DataFrame: ", obs_target_df.shape)
        print("Shape model output DataFrames: ", model_out_df_list[0].shape)

        print(f"\nCreating masks for {N_CV_FOLDS} folds...", end="")
        if N_CV_FOLDS == 1:
            # Make sure all data is included when only 1 fold
            start_date = np.datetime64(f'{FORCING_YR_START-1}-01-01')
            train_kfold_masks = [
                (obs_target_df[OBS_DATE_COLUMN_NAME] >= start_date)
            ]
        else:
            n_train_years = int(N_SIMULATION_YEARS // N_CV_FOLDS)

            train_kfold_masks = []
            cur_year = FORCING_YR_START

            for _ in range(N_CV_FOLDS):

                cur_training_start_date = np.datetime64(f'{cur_year}-01-01')
                cur_training_end_date = np.datetime64(
                    f'{cur_year+n_train_years}-01-01')

                train_kfold_masks.append(
                    (obs_target_df[OBS_DATE_COLUMN_NAME] >= cur_training_start_date) &
                    (obs_target_df[OBS_DATE_COLUMN_NAME]
                     < cur_training_end_date)
                )

                cur_year += n_train_years
        print("done!")

        # Calculate aggregated error if necessary
        if aggregated_errors_df is None:
            aggregated_errors_df = calculate_aggregated_error(
                obs_target_df=obs_target_df
            )

        # START ENKF

        # Load Ensemble object
        with (open(param_ensemble_pkl_fpath, "rb")) as pkl_file:
            ensemble = pickle.load(pkl_file)

        updated_ensembles_list = []

        for fold_idx, cur_fold_mask in enumerate(train_kfold_masks):

            cur_ensemble = copy.deepcopy(ensemble)
            cur_ensemble.current_kalman_iter = cur_ensemble.current_kalman_iter + 1

            print(
                f"Starting fold {fold_idx+1} of {len(train_kfold_masks)}... ",
                end=""
            )
            print("Generating training data... ", end="")

            # Observed Y
            Y_obs_train_df = obs_target_df.loc[cur_fold_mask]

            # Modelled Y
            Y_predicted_train_df_list = []
            for df in model_out_df_list:
                Y_predicted_train_df_list.append(df.loc[cur_fold_mask])

            n_observations = Y_obs_train_df.shape[0]
            n_ensembles = len(Y_predicted_train_df_list)

            # Instantiate observed Y in the shape needed for EnKF implementation
            Y_obs_train = np.zeros(
                shape=(n_target_variables*n_observations, 1)
            )
            # Instantiate Gaussian error matrix to add to model predictions
            obs_error_perturbations = np.zeros(
                shape=(n_target_variables*n_observations, n_ensembles)
            )

            idx = 0
            for var in obs_var_names:
                Y_obs_train[idx:idx+n_observations] = \
                    Y_obs_train_df.loc[:, var].values.reshape(
                        n_observations, 1)

                # Generate Gaussian error. TODO: make more flexible
                # Retrieve specified measurement error, scale by number of Kalman iters
                obs_error_perturbations[idx:idx+n_observations, :] = \
                    np.random.randn(n_observations, n_ensembles) \
                    * aggregated_errors_df.loc[AGGREGATION_FREQUENCY, var] \
                    * np.sqrt(N_KALMAN_ITERATIONS)
                # Increment variable index
                idx = idx + n_observations

            # Transpose for broadcasting
            Y_obs_perturbed = Y_obs_train.flatten() + obs_error_perturbations.T
            # Transpose back for Kris' EnKF code
            Y_obs_perturbed = Y_obs_perturbed.T

            # Generate modelled Y in the shape needed for EnKF implementation
            Y_predicted_train = np.zeros(
                shape=(n_target_variables*n_observations, n_ensembles)
            )

            idx = 0
            for ensemble_idx, cur_model_df in enumerate(Y_predicted_train_df_list):
                for var in model_var_names:
                    Y_predicted_train[idx:idx+n_observations, ensemble_idx] = \
                        cur_model_df[var]
                    idx = idx + n_observations
                idx = 0

            print("Generating parameter matrix theta... ", end="")
            # Copy parameter matrix theta, subset active PFT parameters
            theta = cur_ensemble.theta_df.copy()

            n_parameters = len(FATES_PARAMS_TO_PERTURBATE)
            n_included_pfts = len(INCLUDED_PFT_INDICES)

            # FATES indices 1 based to python 0 based
            included_pft_indices = [x-1 for x in INCLUDED_PFT_INDICES]

            included_parameter_indices = []
            for idx in range(n_parameters):
                included_parameter_indices.extend(
                    np.asarray(included_pft_indices)
                    + np.asarray([idx*N_PFTS] * n_included_pfts)
                )
            included_pft_colnames = theta.columns[included_parameter_indices]

            # Cast to numpy array with required shape
            theta_active_params_array = \
                theta[included_pft_colnames].to_numpy().T

            # Transform theta to DA space
            # Transform the median in the physical space (==true CLM params)
            # to transformed space (Generalized logit)

            # Theta shape: (n_parameters, n_ensemble_members)
            theta_active_params_transformed_array = \
                da.transform_included_fates_parameters(
                    # Transform to (n_ensemble_members, n_parameters)
                    parameter_matrix=theta_active_params_array.T,
                    how='to_gaussian_space'
                ).T  # Transform back to (n_parameters, n_ensembles)

            print("Updating theta with Ensemble Kalman Analysis... ", end="")

            theta_updated_transformed = da.tsubspaceEnKA(
                theta_mat=theta_active_params_transformed_array,
                y_observed_peturbed_mat=Y_obs_perturbed,
                y_predicted_mat=Y_predicted_train,
                svdt=0.9
            )

            # Theta shape: (n_parameters, n_ensemble_members)
            theta_updated = da.transform_included_fates_parameters(
                # Transform to (n_ensemble_members, n_parameters)
                parameter_matrix=theta_updated_transformed.T,
                how='to_model_space'
            ).T  # Transform back to (n_parameters, n_ensembles)

            logger.info("Saving results...")
            # Store updated theta matrix in Ensemble instance (only active PFTs)
            cur_ensemble.theta_df[included_pft_colnames] = \
                theta_updated.copy().T

            # Save results
            if N_CV_FOLDS != 1:
                save_file_stem = file_stem_name+f'_fold{fold_idx}'
            else:
                save_file_stem = file_stem_name

            cur_ensemble.save_as_csv(file_name=save_file_stem+'.csv')
            cur_ensemble.save_as_pkl(file_name=save_file_stem+'.pkl')
            cur_ensemble.write_to_json(file_name=save_file_stem+'.json')

            # Store in list for visualization
            updated_ensembles_list.append(cur_ensemble)

            # PLOT RESULTS

            save_dir = PROJECT_ROOT_PATH / 'data' / 'results' / 'plots' / \
                CASE_NAME / f'kalman_iter_{cur_kalman_iter}'

            for fold_idx, updated_ensemble in enumerate(updated_ensembles_list):

                for idx, param in enumerate(included_pft_colnames):

                    param_ensemble_old = ensemble.theta_df[param]
                    param_ensemble_new = updated_ensemble.theta_df[param]

                    fig, ax = plt.subplots(
                        figsize=(10/2.54, 10/2.54), dpi=300
                    )

                    ax.hist(
                        x=param_ensemble_old,
                        bins=20,
                        label="Prior" if cur_kalman_iter == 0 else f"K{cur_kalman_iter}",
                        alpha=0.7
                    )
                    ax.hist(
                        x=param_ensemble_new,
                        bins=20,
                        label="Posterior",
                        alpha=0.7
                    )

                    ax.set_ylabel("Frequency")
                    ax.set_xlabel(param)

                    ax.legend()

                    title_str = \
                        f"Kalman iteration {updated_ensemble.current_kalman_iter} " \
                        + f"of {updated_ensemble.n_kalman_iters}"

                    if N_CV_FOLDS == 1:
                        ax.set_title(title_str)
                    else:
                        ax.set_title(
                            title_str + f"\nFold {fold_idx} of {N_CV_FOLDS}")

                    fig.tight_layout()

                    if not save_dir.is_dir():
                        save_dir.mkdir(parents=True)
                    fig.savefig(save_dir / f"update_{param}.png")


if __name__ == '__main__':
    main()
