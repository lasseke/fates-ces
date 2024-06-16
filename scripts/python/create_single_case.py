#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sets up a single-instance CTSM case from a single previously created
parameter .json file.
"""

import argparse
from fatescal.modelhandling.ctsm_case import Case
# import fatescal.modelhandling.helpers as hlp
from fatescal.modelhandling.model_config import ModelConfig
from fatescal.config import INPUT_ROOT_PATH, CTSM_COMPSET, \
    MODEL_DRIVER, CASE_RES, CASES_ROOT_PATH, CLM_XML_CHANGES, \
    CASE_NAME, CLM_NAMELIST_CHANGES, CESM_OUTPUT_ROOT_PATH, \
    PROJECT_NAME


parser = argparse.ArgumentParser()

parser.add_argument(
    '-f', '--param-json-fname',
    type=str,
    required=True,
    help='Name of the parameter json file.'
)

parser.add_argument(
    '-n', '--case-name',
    type=str,
    required=False,
    default=CASE_NAME,
    help='''
    Name of the case. OBS! Requires a single parameter json
    in 'data/results/json/case_name/'!
    '''
)

parser.add_argument(
    "--synth-truth-run",
    action="store_true",
    help="Use flag for synthetic truth runs."
)

model_cfg = ModelConfig()


def main():

    args = parser.parse_args()

    case_dir_name = args.case_name

    my_case = Case(
        name=case_dir_name,
        data_root_path=INPUT_ROOT_PATH,
        data_url=None,
        compset=CTSM_COMPSET,
        model_driver=MODEL_DRIVER,
        case_res=CASE_RES,
        cases_root_path=CASES_ROOT_PATH,
        output_root_path=CESM_OUTPUT_ROOT_PATH,
        project_name=PROJECT_NAME,
        is_multi_instance=False,  # New non-multi instance case bool!
    )

    my_case.create_case()

    # Adjust output directory
    CLM_XML_CHANGES['DOUT_S_ROOT'] = str(
        CESM_OUTPUT_ROOT_PATH / case_dir_name / 'archive'
    )

    # Apply XML changes defined in config, common across all cases
    my_case.xml_change(xml_changes_dict=CLM_XML_CHANGES)
    my_case.case_setup()

    # Create FATES param files from prior json
    my_case.create_fates_param_files(
        fates_param_json_fname=str(args.param_json_fname),
        synthetic_truth=args.synth_truth_run
    )

    # Add changes to all generated clm user namelists (one per instance)
    my_case.add_to_namelists(
        nl_changes_dict=CLM_NAMELIST_CHANGES,
        namelist='user_nl_clm',
    )

    my_case.case_build()

    my_case.case_submit()


if __name__ == '__main__':
    main()
