#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates prior matrices (csv, json, pkl files) for the parameters and
prior distribution options defined in 'fatescal/config.py'
"""

from fatescal.modelhandling.parameter_config import FatesDefaultParameter, \
    FatesParameterEnsemble
from fatescal.config import FATES_PARAMS_TO_PERTURBATE
import datetime
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument(
    '-n', '--file-name-stem',
    type=str,
    required=False,
    default='',
    help='Name stem of the output files (csv, json, pkl) containing the generated priors.'
)


def main() -> None:

    args = parser.parse_args()

    parameter_list = []

    for param_name, param_options in FATES_PARAMS_TO_PERTURBATE.items():

        parameter_list.append(
            FatesDefaultParameter(
                param_name=param_name,
                dimensions=param_options['dimensions'],
                label=param_options['label']
            )
        )

    # Initialize new parameter ensemble
    my_ensemble = FatesParameterEnsemble()

    # Add the generated parameters
    my_ensemble.set_default_parameters(
        parameter_list=parameter_list
    )

    # Generate initial priors
    my_ensemble.generate_priors()

    # Generate csv file name with timestamp if none provided
    if not str(args.file_name_stem):
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%y%m%d_%H%M%S")

        file_name_stem = f"FATES_priors_{formatted_datetime}.csv"
    else:
        file_name_stem = str(args.file_name_stem)

    my_ensemble.save_as_csv(
        file_name=f"{file_name_stem}.csv"
    )

    my_ensemble.save_as_pkl(
        file_name=f"{file_name_stem}.pkl"
    )

    # Save as JSON
    my_ensemble.write_to_json(
        file_name=f"{file_name_stem}.json"
    )


if __name__ == "__main__":
    main()
