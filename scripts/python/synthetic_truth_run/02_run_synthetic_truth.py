#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run model to generate synthetic truth data with known 'true' parameters.
"""

import time
from pathlib import Path

from fatescal.helpers import run_subprocess, setup_logging, wait_for_model
from fatescal.config import PROJECT_ROOT_PATH, CESM_OUTPUT_ROOT_PATH, \
    AGGREGATION_FREQUENCY, CASE_NAME

# Configure logger
logger = setup_logging('syntruth')


def main():

    # File name stem of parameter ensemble (pkl, csv, json)
    file_stem_name = 'synthetic_truth_run'

    # RUN MODEL

    logger.info("Creating truth run case...")

    run_subprocess(
        cmd=[
            f"python3 create_single_case.py -f {file_stem_name}.json --synth-truth-run"
        ],
        cwd=PROJECT_ROOT_PATH / 'scripts' / 'python',
        shell=True,
    )

    time.sleep(5)  # Sleep for 5 seconds
    print("Waiting for model to finish...", end="")
    wait_for_model()

    logger.info("Concatenating outputs...")

    # Path to default model outputs (one above hist folder)
    nc_root_path = CESM_OUTPUT_ROOT_PATH / CASE_NAME / 'archive' / 'lnd'

    run_subprocess(
        cmd=[
            f"python3 concat_nc_output.py -d {nc_root_path / 'hist'} -o {nc_root_path} --single-instance"
        ],
        cwd=PROJECT_ROOT_PATH / 'scripts' / 'python',
        shell=True,
    )

    logger.info("Aligning model outputs with observations...")
    run_subprocess(
        cmd=[
            f"python3 03_prepare_data_assim.py -mp {nc_root_path} -ht 1 -a {AGGREGATION_FREQUENCY} -k 0"
        ],
        cwd=PROJECT_ROOT_PATH / 'scripts' / 'python' / 'cesm_ensemble_runs',
        shell=True,
    )


if __name__ == '__main__':
    main()
