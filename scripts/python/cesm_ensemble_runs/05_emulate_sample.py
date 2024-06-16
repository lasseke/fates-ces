#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a negative-log-likelihood emulator and perform MCMC.
"""

import argparse
from pathlib import Path
import glob
import joblib
import pandas as pd
import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

from fatescal.visualize.emulation import plot_emulator_evaluation
from fatescal.visualize.mcmc import plot_mcmc_chain_histogram
from fatescal.ces import daalgorithms as da
from fatescal.ces import mcmc
from fatescal.tools import unit_conversion as uc
from fatescal.config import CASE_NAME, INCLUDED_PFT_INDICES, \
    FATES_PARAMS_TO_PERTURBATE, OBS_DATE_COLUMN_NAME, \
    VARIABLES_TO_ASSIMILATE, PROJECT_ROOT_PATH, \
    EMULATOR_TRAIN_FRACTION, SCALE_NLL_BY_GPR_STD, \
    GPR_STD_PENALTY_FACTOR, EMULATOR_MODEL_NAME

# Tell pandas to display more float digits to explore small values
pd.options.display.float_format = '{:,.10f}'.format

# Wildcards for default naming in 'aligned_obs_mode' - should be no need to change
OBSERVATION_FNAME_GLOB = 'O_*.csv'
MODEL_FNAMES_GLOB = 'M_*.csv'

parser = argparse.ArgumentParser()

parser.add_argument(
    '-n', '--file-stem-name',
    type=str,
    required=True,
    help='File stem name (wo/ file ending) of the prior csv/json/ensemble files.'
)

parser.add_argument(
    '-e', '--emulation-target',
    type=str,
    required=False,
    default='nll',
    help="Property to emulate. Currently only 'nll' (neg. log-likelihood) implemented."
)


def main() -> None:

    args = parser.parse_args()

    # File name stem of parameter ensemble (pkl, csv, json)
    file_stem_name = args.file_stem_name
    if any(file_stem_name.endswith(ext) for ext in ['.csv', '.json', '.pkl']):
        if file_stem_name.endswith(".json"):
            file_stem_name = file_stem_name[:-5]
        else:
            file_stem_name = file_stem_name[:-4]

    # FOR TESTING:
    # CASE_NAME = 'HYY_n128_daily_gpp_et_TEST'
    # file_stem_name = 'test_auto'

    # File name stem of parameter ensemble (pkl, csv, json)
    error_metric_code = args.emulation_target

    parameter_dirs_path = PROJECT_ROOT_PATH / 'data' / 'results' \
        / 'fates_param_ensembles' / CASE_NAME / 'csv'

    aligned_obs_model_dirs_path = PROJECT_ROOT_PATH / 'data' / 'results' \
        / 'aligned_obs_model' / CASE_NAME

    # Retrieve active parameters
    active_param_names = []
    for param in FATES_PARAMS_TO_PERTURBATE.keys():
        for pft_index in INCLUDED_PFT_INDICES:
            active_param_names.extend([f"{param}_PFT{pft_index}"])
    # Convert Fortran to Python indices for var transformation later
    included_pft_indices = [x-1 for x in INCLUDED_PFT_INDICES]

    # Instantiate feature matrix
    feature_matrix = pd.DataFrame(columns=active_param_names)

    # Loop through directories and fill feature matrix
    kalman_dirs = sorted(glob.glob(
        f"{parameter_dirs_path}/kalman_iter_*/"
    ))

    # ONLY USE LAST KALMAN ITERATION FOR EMULATOR TRAINING
    # TODO: use all data? Implications for emulator training
    # kalman_dir = kalman_dirs[-1]

    # Uncomment next line and indent loop for all data
    for kalman_dir in kalman_dirs:

        cur_param_df = pd.read_csv(
            Path(kalman_dir) / f'{file_stem_name}.csv',
            index_col=0
        )[active_param_names]  # Subset active parameters

        # Instantiate current transformed parameter matrix
        cur_feature_matrix = pd.DataFrame(columns=active_param_names)

        # Theta shape: (n_parameters, n_ensemble_members)
        cur_feature_matrix = da.transform_included_fates_parameters(
            parameter_matrix=cur_param_df,
            how='to_gaussian_space'
        )

        feature_matrix = pd.concat(
            [feature_matrix, cur_feature_matrix],
            axis=0  # Append rows in current dataframe
        )

    # Reset and drop dataframe indices
    feature_matrix = feature_matrix.reset_index(drop=True)

    # Save to file
    feat_mat_save_path = PROJECT_ROOT_PATH / 'data' / 'results' / \
        'emulation' / 'feature_matrix' / CASE_NAME
    if not feat_mat_save_path.is_dir():
        feat_mat_save_path.mkdir(parents=True)

    feature_matrix.to_csv(
        feat_mat_save_path / f"{file_stem_name}.csv"
    )

    # CREATE TARGET VECTOR
    observation_error_variances = \
        np.array(
            [x['Observed']['error']['std'] for x in VARIABLES_TO_ASSIMILATE]
        )**2  # Standard deviation to variance

    '''
    # Convert to [g d-1]
    observation_error_variances = uc.convert_unit(
            values=observation_error_variances,
            unit_in='kg s-1',
            unit_out='g d-1'
        )
    '''

    # Observed target variables
    obs_var_names = [x['Observed']['csv_col_name']
                     for x in VARIABLES_TO_ASSIMILATE]

    model_var_names = \
        [x['CLM-FATES']['history_var_name']
         for x in VARIABLES_TO_ASSIMILATE]

    # Instantiate target matrix
    targets_y = pd.DataFrame(columns=[error_metric_code])

    # Loop through directories and fill feature matrix
    kalman_dirs = sorted(glob.glob(
        f"{aligned_obs_model_dirs_path}/kalman_iter_*/"
    ))

    for kalman_dir in kalman_dirs:
        # Use line below instead of loop to only use last iteration
        # for training
        # kalman_dir = kalman_dirs[-1]

        # Read observations
        observation_csv_fpath = Path(
            glob.glob(str(Path(kalman_dir) / OBSERVATION_FNAME_GLOB))[0]
        )
        obs_target_df = pd.read_csv(observation_csv_fpath)
        # Date column to datetime
        obs_target_df[OBS_DATE_COLUMN_NAME] = \
            pd.to_datetime(obs_target_df[OBS_DATE_COLUMN_NAME])

        # Drop NA in observations, and store indices to also drop from modelled
        # TODO: UPDATE, DON'T NEED TO THROW AWAY OBS FOR ALL VARS IF ONLY ONE IS MISSING
        obs_na_mask = pd.isnull(obs_target_df).any(axis=1)
        print(
            f"Dropping {sum(obs_na_mask)} NA occurrences from observation and target data."
        )
        obs_target_df = obs_target_df.dropna(axis='index', how='any')

        '''
        for var in obs_var_names:
            obs_target_df[var] = uc.convert_unit(
                values=obs_target_df[var],
                unit_in='kg s-1',
                unit_out='g d-1'
            )
        '''

        # Drop time column
        obs_target_df = obs_target_df.drop([OBS_DATE_COLUMN_NAME], axis=1)

        # Read model outputs
        model_out_csv_fpath_list = sorted(glob.glob(
            str(Path(kalman_dir) / MODEL_FNAMES_GLOB)
        ))

        for fname in model_out_csv_fpath_list:

            cur_df = pd.read_csv(fname)

            # Drop time column
            cur_df = cur_df.drop(['time'], axis=1)

            # Remove entries where observations are NA
            cur_df = cur_df.loc[~obs_na_mask]

            # TODO: Converting to g d-1 leads to overflow due to large neg. log-lik.
            # Scale data somehow?
            '''
            # CONVERT TO [g C d-1]
            for var in model_var_names:
                cur_df[var] = uc.convert_unit(
                    values=cur_df[var],
                    unit_in='kg s-1',
                    unit_out='g d-1'
                )
            '''

            # cur_df --> model output for current ensemble member in current Kalman iteration
            if error_metric_code == 'nll':

                targets_y.loc[targets_y.shape[0], error_metric_code] = \
                    da.negative_log_likelihood(
                        actual=obs_target_df,
                        predicted=cur_df,
                        error_covariances=observation_error_variances
                )

    # Save to file
    targets_save_path = PROJECT_ROOT_PATH / 'data' / 'results' / \
        'emulation' / 'targets' / CASE_NAME
    if not targets_save_path.is_dir():
        targets_save_path.mkdir(parents=True)

    targets_y.to_csv(
        targets_save_path / f"{file_stem_name}.csv"
    )

    # EMULATOR

    # TODO: Make flexible.
    emulator_name = EMULATOR_MODEL_NAME

    if emulator_name != 'gaussian_process_regressor':

        emulator_name = 'gaussian_process_regressor'

        print(
            '''
            Warning: only 'gaussian_process_regressor' model is
            currently implemented and will be used.
            '''
        )

    # Define GP kernel and model
    # TODO: add as adjustable hyperparameters
    gp_kernel = Matern(length_scale=1.0, nu=1.5)

    emulator_model = GaussianProcessRegressor(
        kernel=gp_kernel,
        optimizer='fmin_l_bfgs_b',
        n_restarts_optimizer=9,
        normalize_y=True  # Needs 'True' to obtain unit variance!
    )

    # Scale y-data
    # TODO: Currently replaced by 'normalize_y=True' in GPR
    """ scaler = StandardScaler()
    scaler.fit(targets_y)
    normalized_y = scaler.transform(targets_y) """

    # Split train-test data
    if EMULATOR_TRAIN_FRACTION < 1.0:
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix.values,
            targets_y,  # normalized_y,
            train_size=EMULATOR_TRAIN_FRACTION,
            random_state=221,  # b
            shuffle=True,
        )

        # Fit model
        emulator_model.fit(
            X_train,
            np.asarray(y_train).astype(np.float32),
        )

        # Simple evaluation of actual-predicted mismatches
        error_metric_actual = np.asarray(y_test).astype(np.float32)
        """ scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ) """
        error_metric_predictions = emulator_model.predict(
            X_test
        ).reshape(-1, 1)
        """ scaler.inverse_transform(
            emulator_model.predict(X_test).reshape(-1, 1)
        ) """
        # Plot results
        plot_emulator_evaluation(
            actual=error_metric_actual,
            predicted=error_metric_predictions,
            metric_code=error_metric_code
        )
    else:
        # Fit model
        emulator_model.fit(
            feature_matrix.values,
            np.asarray(targets_y).astype(np.float32),
        )

    # Save model to disk
    model_save_path = PROJECT_ROOT_PATH / 'data' / 'results' / \
        'emulation' / 'models' / CASE_NAME

    if not model_save_path.is_dir():
        model_save_path.mkdir(parents=True)

    # Save trained emulator
    joblib.dump(
        emulator_model,
        model_save_path / f'{emulator_name}.joblib'
    )

    # MCMC SAMPLING

    mcmc_chain = mcmc.emulator_mcmc(
        starting_parameters=feature_matrix,
        emulator_model=emulator_model,
        scale_nll_by_gpr_std=SCALE_NLL_BY_GPR_STD,
        std_penalty_factor=GPR_STD_PENALTY_FACTOR,
        save_chain=True,
    )

    # Plot chain
    plot_mcmc_chain_histogram(
        mcmc_chain,
        bins=50
    )


if __name__ == '__main__':
    main()
