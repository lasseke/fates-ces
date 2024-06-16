#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concatenate model output into a single NetCDF file.
Requires NCO to be installed and loaded.
"""

import argparse
from pathlib import Path
from fatescal.helpers import run_subprocess
from fatescal.modelhandling.model_config import ModelConfig
from fatescal.config import CESM_OUTPUT_ROOT_PATH, CASE_NAME, \
    N_ENSEMBLE_MEMBERS, FORCING_YR_START, FORCING_YR_END


# Default path to store outputs
DEFAULT_OUT_PATH = Path(CESM_OUTPUT_ROOT_PATH) / \
    'archive' / 'lnd' / 'hist'

model_cfg = ModelConfig()

parser = argparse.ArgumentParser(description='Concatenates .nc files.')

parser.add_argument(
    '-n', '--n-instances',
    type=int,
    required=False,
    default=N_ENSEMBLE_MEMBERS,
    help="Corresponds to max. no. of clm outputs in multi instance mode for XXXX in '*clm2_XXXX*.nc'."
)

parser.add_argument(
    '-ys', '--year-start-suffix',
    type=int,
    required=False,
    default=FORCING_YR_START,
    help="First simulation year (for file naming)."
)

parser.add_argument(
    '-ye', '--year-end-suffix',
    type=int,
    required=False,
    default=FORCING_YR_END,
    help="Final simulation year (for file naming)."
)

parser.add_argument(
    '-ht', '--history-tape',
    type=str,
    required=False,
    default="h1",
    help="History tape output to concatenate."
)

parser.add_argument(
    '-d', '--hist-files-dir',
    type=Path,
    required=False,
    default=DEFAULT_OUT_PATH,
    help='Path to directory where .nc files are located.'
)

parser.add_argument(
    '-o', '--out-file-dir',
    type=Path,
    required=False,
    default="",
    help='Path to directory for storing out file.'
)

parser.add_argument(
    '-k', '--kalman-iter',
    type=int,
    required=False,
    default=0,
    help='Index of current kalman iteration, 0 if not applicable.'
)

parser.add_argument(
    "--mcmc",
    action="store_true",
    help="Use flag to treat as mcmc run for file naming"
)

parser.add_argument(
    "--single-instance",
    action="store_true",
    help="Use flag to concatenate single-instance run outputs"
)


def main() -> None:

    args = parser.parse_args()

    if not args.out_file_dir:
        args.out_file_dir = Path(args.hist_files_dir).parents[1]
    elif not Path(args.out_file_dir).is_dir():
        raise ValueError(f"'{args.out_file_dir}' does not exist!")

    if args.single_instance:
        out_file_name = \
            f"{CASE_NAME}.clm2.{args.history_tape}.{args.year_start_suffix}-{args.year_end_suffix}.nc"

        out_file_path = Path(args.out_file_dir) / out_file_name

        ncrcat_cmd = f"ncrcat {args.hist_files_dir}/*clm2.{args.history_tape}.*.nc {out_file_path}"

        _ = run_subprocess(
            [
            f"module purge;module load {model_cfg.nco_module_name};{ncrcat_cmd};module purge;"
            ],
            shell=True
        )
    
    else:
        n_instances = int(args.n_instances)

        for ensemble_idx in range(n_instances):

            if args.mcmc:
                suffix = "mcmc"
            else:
                suffix = f"k{args.kalman_iter}"

            out_file_name = \
                f"{CASE_NAME}_{suffix}.clm2_{ensemble_idx+1:04d}.{args.history_tape}.{args.year_start_suffix}-{args.year_end_suffix}.nc"

            out_file_path = Path(args.out_file_dir) / out_file_name

            ncrcat_cmd = f"ncrcat {args.hist_files_dir}/*clm2_{ensemble_idx+1:04d}.{args.history_tape}.*.nc {out_file_path}"

            _ = run_subprocess(
                [
                f"module purge;module load {model_cfg.nco_module_name};{ncrcat_cmd};module purge;"
                ],
                shell=True
            )


if __name__ == "__main__":
    main()
