#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate the synthetic truth (ST) run used in
Keetz et al. (submitted)
"""

from fatescal.modelhandling.parameter_config import FatesDefaultParameter, \
    FatesParameterEnsemble
from fatescal.config import FATES_PARAMS_TO_PERTURBATE

# DEFINE "TRUE" PARAMETERS

# Defaults: [50. 62. 39. 61. 58. 58. 62. 54. 54. 78. 78. 78.]
# PFT indices 2, 6, 11 -> 85 % of original value
synthetic_truth_vcmax_values = [
    50, 52.7, 39, 61, 58, 49.3, 62, 54, 54, 78, 66.3, 78
]

# Defaults: [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
# PFT indices 2, 6, 11 -> 80 % of original value
synthetic_truth_bbslope_values = [
    8, 6.4, 8, 8, 8, 6.4, 8, 8, 8, 8, 6.4, 8
]


def main():

    # Generate a dummy Ensemble object
    parameter_list = []
    for param_name, param_options in FATES_PARAMS_TO_PERTURBATE.items():
        parameter_list.append(
            FatesDefaultParameter(
                param_name=param_name,
                dimensions=param_options['dimensions'],
                label=param_options['label']
            )
        )
    # Initialize new parameter ensemble with only 1 member
    my_ensemble = FatesParameterEnsemble(n_ensembles=1)
    # Add the generated parameters
    my_ensemble.set_default_parameters(
        parameter_list=parameter_list
    )

    # Generate initial dummy priors
    my_ensemble.generate_priors()

    # REPLACE PRIORS WITH TRUTH RUN PARAMETERS
    my_ensemble.theta_df.iloc[0, 0:12] = synthetic_truth_vcmax_values
    my_ensemble.theta_df.iloc[0, 12:24] = synthetic_truth_bbslope_values

    # Save to files
    my_ensemble.save_as_csv(
        file_name="synthetic_truth_run.csv",
        synth_truth_run=True,
    )
    my_ensemble.save_as_pkl(
        file_name="synthetic_truth_run.pkl",
        synth_truth_run=True,
    )
    my_ensemble.write_to_json(
        file_name="synthetic_truth_run.json",
        synth_truth_run=True,
    )


if __name__ == '__main__':
    main()
