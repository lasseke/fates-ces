#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a parameter json file for Latin Hyper Cube samples
using the parameter bounds defined in the settings file.
"""

import argparse
from scipy.stats import qmc
import numpy as np
import pandas as pd

from fatescal.modelhandling.parameter_config import create_dummy_fates_ensemble
from fatescal.config import INCLUDED_PFT_INDICES, N_ENSEMBLE_MEMBERS, \
    FATES_PARAMS_TO_PERTURBATE, N_PFTS

parser = argparse.ArgumentParser(
    description='Create Latin Hyper Cube parameter files.'
)

parser.add_argument(
    '-i', '--n-instances',
    type=int,
    required=False,
    default=N_ENSEMBLE_MEMBERS,
    help="Number of ensemble members."
)

parser.add_argument(
    '-f', '--file-stem',
    type=str,
    required=True,
    help="File stem for output parameter files (.json, .csv, .pkl)."
)


def main() -> None:

    args = parser.parse_args()

    # Adjust Fortran indices to Python
    included_pft_indices = np.array(INCLUDED_PFT_INDICES) - 1
    # Define small epsilon to avoid getting values exactly on bounds
    epsilon = 0.0001

    lower_bounds = np.asarray(
        [np.array(x['prior']['lower_bound'])[included_pft_indices]
            for x in FATES_PARAMS_TO_PERTURBATE.values()]
    ).flatten() + epsilon
    upper_bounds = np.asarray(
        [np.array(x['prior']['upper_bound'])[included_pft_indices]
            for x in FATES_PARAMS_TO_PERTURBATE.values()]
    ).flatten() - epsilon

    n_instances = int(args.n_instances)
    n_parameters = len(lower_bounds)

    # Generate LHC samples
    sampler = qmc.LatinHypercube(d=n_parameters)
    sample = sampler.random(n=n_instances)
    sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)

    # REPLACE PRIORS WITH LHC PARAMETERS
    n_params = len(FATES_PARAMS_TO_PERTURBATE)
    n_total_fates_parameters = N_PFTS
    active_pft_indices = []
    for idx in range(n_params):
        active_pft_indices.extend(
            [int(x) for x in np.array(included_pft_indices)+(n_total_fates_parameters*idx)]
        )

    # Generate a dummy Ensemble object
    my_ensemble = create_dummy_fates_ensemble(n_ensembles=n_instances)

    sample_scaled_df = pd.DataFrame(
        sample_scaled,
        columns=my_ensemble.theta_df.columns[active_pft_indices],
    )

    my_ensemble.theta_df[sample_scaled_df.columns] = sample_scaled_df.values

    # Save to files
    out_files_stem = str(args.file_stem)

    my_ensemble.save_as_csv(
        file_name=f"{out_files_stem}.csv",
    )
    my_ensemble.save_as_pkl(
        file_name=f"{out_files_stem}.pkl",
    )
    my_ensemble.write_to_json(
        file_name=f"{out_files_stem}.json",
    )

    print(
        f"Finished creating {n_instances} LHC samples for parameters {sample_scaled_df.columns} "
        f"with lower bounds {lower_bounds} and upper bounds {upper_bounds}!\n"
    )


if __name__ == "__main__":
    main()
