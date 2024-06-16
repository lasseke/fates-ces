#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates a multi-instance CTSM case.
"""

import argparse
from fatescal.modelhandling.ctsm_case import Case
# import fatescal.modelhandling.helpers as hlp
from fatescal.modelhandling.model_config import ModelConfig
from fatescal.config import INPUT_ROOT_PATH, CTSM_COMPSET, \
    MODEL_DRIVER, CASE_RES, CASES_ROOT_PATH, CLM_XML_CHANGES, \
    CASE_NAME, N_ENSEMBLE_MEMBERS, CLM_NAMELIST_CHANGES, \
    CESM_OUTPUT_ROOT_PATH, PROJECT_NAME, N_KALMAN_ITERATIONS


parser = argparse.ArgumentParser()

parser.add_argument(
    '-f', '--param-json-fname',
    type=str,
    required=True,
    help='Glob-style path to NetCDF files.'
)

parser.add_argument(
    '-k', '--kalman-iter',
    type=int,
    required=True,
    help='Number of current kalman iteration, use 0 if no iterations.'
)

parser.add_argument(
    '-i', '--n-instances',
    type=int,
    required=False,
    default=N_ENSEMBLE_MEMBERS,
    help='Number of current kalman iteration, use 0 if no iterations.'
)

parser.add_argument(
    "--mcmc",
    action="store_true",
    help="Use flag to treat as mcmc run for file naming"
)

model_cfg = ModelConfig()


def main() -> None:

    args = parser.parse_args()

    # If negative kalman iteration given, treated as mcmc run
    if args.mcmc:
        kalman_iter = N_KALMAN_ITERATIONS
        folder_name = f"{CASE_NAME}_mcmc"
    else:
        kalman_iter = args.kalman_iter
        folder_name = f"{CASE_NAME}_k{args.kalman_iter}"

    my_case = Case(
        name=folder_name,
        data_root_path=INPUT_ROOT_PATH,
        data_url=None,
        compset=CTSM_COMPSET,
        model_driver=MODEL_DRIVER,
        case_res=CASE_RES,
        cases_root_path=CASES_ROOT_PATH,
        output_root_path=CESM_OUTPUT_ROOT_PATH,
        project_name=PROJECT_NAME
    )

    my_case.create_case()

    # Adjust output directory
    CLM_XML_CHANGES['DOUT_S_ROOT'] = str(
        CESM_OUTPUT_ROOT_PATH / folder_name / 'archive'
    )

    # Apply XML changes defined in config, common across all cases
    my_case.xml_change(xml_changes_dict=CLM_XML_CHANGES)

    # Create multi-instances
    my_case.case_as_multi_driver(n_model_instances=args.n_instances)

    # Create FATES param files from prior json
    my_case.create_fates_param_files(
        fates_param_json_fname=str(args.param_json_fname),
        kalman_iter=int(kalman_iter),
        mcmc=bool(args.mcmc)
    )

    # Add changes to all generated clm user namelists (one per instance)
    my_case.add_to_namelists(
        nl_changes_dict=CLM_NAMELIST_CHANGES,
        namelist='user_nl_clm',
    )

    my_case.case_build()


if __name__ == '__main__':
    main()
