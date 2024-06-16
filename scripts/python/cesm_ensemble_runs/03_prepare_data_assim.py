#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aligns observations with model data. See description string below.
"""

import argparse
from pathlib import Path
from fatescal.tools.obs_to_model_harmonizer import ObsToModelHarmonizer
from fatescal.config import CESM_OUTPUT_ROOT_PATH, \
    OBS_DATA_FILE_NAME, PATH_TO_OBS_DATASET, AGGREGATION_FREQUENCY


DESCRIPTION = '''
Aligns the observed data specified in 'fatescal/config.py' with
ensemble model outputs and saves them in a format consistent
with the subsequent data assimilation workflow.

Assumes that the model ensemble output data is stored in a single
directory and each ensemble's output is represented by one file
matching *<hist_tape>*.nc.
'''

parser = argparse.ArgumentParser(
    description=DESCRIPTION
)

parser.add_argument(
    '-mp', '--model-data-path',
    type=Path,
    required=False,
    default=CESM_OUTPUT_ROOT_PATH / 'archive',
    help='Path to concatenated NetCDF files.'
)

parser.add_argument(
    '-ht', '--history-tape',
    type=str,
    required=False,
    default='h1',
    help='Specifies the history tape to use.'
)

parser.add_argument(
    '-op', '--obs-data-path',
    type=Path,
    required=False,
    default=PATH_TO_OBS_DATASET,
    help='Path to observation csv file.'
)

parser.add_argument(
    '-of', '--obs-data-fname',
    type=str,
    required=False,
    default=OBS_DATA_FILE_NAME,
    help='CSV file name of observations.'
)

parser.add_argument(
    '-a', '--aggregation',
    type=str,
    required=False,
    default=AGGREGATION_FREQUENCY,
    help='How observations and model outputs should be aggregated.'
)

parser.add_argument(
    '-k', '--kalman-iter',
    type=int,
    required=True,
    help='Number of kalman iteration (due to poor design choices).'
)

parser.add_argument(
    "--mcmc",
    action="store_true",
    help="include flag to treat as mcmc run"
)


def main():

    args = parser.parse_args()

    observed_csv_file_path = args.obs_data_path / args.obs_data_fname

    # Harmonizer class to align model and obs
    harmonizer = ObsToModelHarmonizer(
        observation_csv_file_path=observed_csv_file_path,
        model_nc_output_dir_path=args.model_data_path,
        model_nc_hist_tape=args.history_tape
    )

    # Aggregate, align timesteps, etc.
    _ = harmonizer.aggregate(
        freq=args.aggregation,
        how='mean',
        save_csv=True,
        kalman_iter=int(args.kalman_iter),
        mcmc=bool(args.mcmc)
    )


if __name__ == '__main__':
    main()
