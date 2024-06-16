#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read aligned truth run model output, add random noise,
convert to HYY observation data formats to comply with
code pipeline, save to file.
"""

import pandas as pd
import numpy as np

from fatescal.tools import unit_conversion as uc
from fatescal.config import PROJECT_ROOT_PATH, \
    VARIABLES_TO_ASSIMILATE, PATH_TO_OBS_DATASET, \
    OBS_DATA_FILE_NAME, OBS_DATE_COLUMN_NAME, \
    AGGREGATION_FREQUENCY


def main() -> None:

    # Read CLM-FATES truth run model output that was already aligned
    # with daily aggregated observation data
    truth_run_file_name = \
        'M_1D_mean_HYY_synthetic_truth_1D_gpp_et.clm2.h1.2004-2013.csv'
    truth_run_csv_path = PROJECT_ROOT_PATH / 'data' / 'results' / 'aligned_obs_model' / \
        'HYY_synthetic_truth_1D_gpp_et' / 'kalman_iter_0' / truth_run_file_name

    truth_run_df = pd.read_csv(truth_run_csv_path)
    truth_run_df["time"] = pd.to_datetime(truth_run_df["time"])

    # Transform back to original units, to meet code requirements in pipeline
    transformed_df = truth_run_df.copy()

    transformed_df["FATES_GPP"] = uc.convert_unit(
        values=truth_run_df["FATES_GPP"].values,
        unit_in='kg C m-2 s-1',
        unit_out='µmol CO2 m-2 s-1',
    )

    transformed_df["QFLX_EVAP_TOT"] = uc.convert_unit(
        values=truth_run_df["QFLX_EVAP_TOT"].values,
        unit_in='kg H2O m-2 s-1',
        unit_out='mmol H2O m-2 s-1',
    )

    # Set same column names as original data csv file
    transformed_df.columns = ["samptime", "GPP", "ET_gapf"]

    # Read initial input observation dataset
    obs_initial_timesteps_df = pd.read_csv(
        PATH_TO_OBS_DATASET / OBS_DATA_FILE_NAME
    )

    # Convert to datetime and UTC timezone
    obs_initial_timesteps_df[OBS_DATE_COLUMN_NAME] = pd.to_datetime(
        obs_initial_timesteps_df[OBS_DATE_COLUMN_NAME],
        utc=True
    ).dt.tz_localize(None)  # Remove timezone information

    # Subset to period of aggregated/harmonized observations
    start_date = min(transformed_df[OBS_DATE_COLUMN_NAME])
    end_date = max(transformed_df[OBS_DATE_COLUMN_NAME])
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

    # Convert unit back to original observation units, ERRORS GIVEN IN MODEL UNITS
    aggregated_errors_df["GPP"] = uc.convert_unit(
        values=aggregated_errors_df["GPP"].values,
        unit_in='kg C m-2 s-1',
        unit_out='µmol CO2 m-2 s-1',
    )

    aggregated_errors_df["ET_gapf"] = uc.convert_unit(
        values=aggregated_errors_df["ET_gapf"].values,
        unit_in='kg H2O m-2 s-1',
        unit_out='mmol H2O m-2 s-1',
    )

    # ADD PERTURBATIONS

    perturbed_df = transformed_df.copy()

    # Add Gaussian noise, with error model adjusted to daily aggregation
    perturbed_df["GPP"] = transformed_df["GPP"] + \
        np.random.normal(
            loc=0,
            scale=aggregated_errors_df.loc["1D", "GPP"],
            size=(transformed_df.shape[0],)
    )
    perturbed_df["ET_gapf"] = transformed_df["ET_gapf"] + \
        np.random.normal(
            loc=0,
            scale=aggregated_errors_df.loc["1D", "ET_gapf"],
            size=(transformed_df.shape[0],)
    )

    # Store in original target DataFrame
    # Takes around 5 minutes
    original_df = obs_initial_timesteps_df.copy()

    for idx, fake_date in enumerate(perturbed_df["samptime"]):

        cur_date_mask = (original_df["samptime"].dt.year == fake_date.year) & \
            (original_df["samptime"].dt.month == fake_date.month) & \
            (original_df["samptime"].dt.day == fake_date.day)

        original_df.loc[cur_date_mask, "GPP"] = perturbed_df.loc[idx, "GPP"]
        original_df.loc[cur_date_mask,
                        "ET_gapf"] = perturbed_df.loc[idx, "ET_gapf"]

    # SAVE TO CSV :)
    original_df.to_csv(
        PROJECT_ROOT_PATH / 'data' / 'sites' / 'hyy' /
        'synthetic_truth' / 'HYY_synthetic_truth_gpp_et_1D_mean.csv',
        index=False,
    )


if __name__ == '__main__':
    main()
