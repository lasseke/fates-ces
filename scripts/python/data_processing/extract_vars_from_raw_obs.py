#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subset observational data to use in DA scheme.
"""

import glob
from pathlib import Path
import pandas as pd
import numpy as np


config = {
    'site_name': 'hyy',
    'csv_root_path':
        Path(__file__).parents[3] / 'data' / 'sites' / 'hyy' / 'raw',
    # Glob style name of (multiple) dataset(s)
    'csv_glob': 'HYY_EDDY233_*.csv',
    'datetime_col': 'samptime',
    'datetime_format': None,  # Infer if None
    'drop_na': True,  # Drop rows with NA entries?
    'cols_to_extract': ['GPP', 'ET_gapf'],  # Variable column names in raw data
    'include_if': {  # Use some quality flag (QF)?
        # Keys must correspond to variables in 'cols_to_extract'
        #'GPP': {'Qc_gapf_NEE': [0]},  # Keys: QF column name
        #'ET_gapf': {'Qc_gapf_ET': [0]}  # Vals: list of acceptable values
    },
    "save_path":
    Path(__file__).parents[3] / 'data' / 'sites' / 'hyy' / \
    'processed' / 'HYY_EDDY233_GPP_ET_merged_allgapfilled.csv',
}


def main() -> None:

    # Retrieve file names specified in config
    csv_file_paths = sorted(glob.glob(
        pathname=config['csv_glob'],
        root_dir=config['csv_root_path']
    ))

    # Concatenate multiple csv files into one DataFrame (appends rows)
    if len(csv_file_paths) > 1:
        data_df = pd.concat(
            (pd.read_csv(config['csv_root_path'] / f) for f in csv_file_paths),
            ignore_index=True,
            axis=0
        )
    else:
        data_df = pd.read_csv(config['csv_root_path'] / csv_file_paths[0])

    print(f"Raw data shape: {data_df.shape}")

    # Generate list of all desired column names
    subset_col_names = [config['datetime_col']] + config['cols_to_extract']
    # Add quality flag columns, if applicable
    for key, val in config['include_if'].items():
        if key in config['cols_to_extract']:
            subset_col_names += [k for k in val.keys()]
        else:
            raise ValueError(
                f"Gave QF for '{key}' not in '{config['cols_to_extract']}'"
            )

    subset_data_df = data_df[subset_col_names]

    # Remove NA rows?
    if config['drop_na']:
        subset_data_df = subset_data_df.dropna(
            axis=0,
            how='any'
        )

    qf_masks = []

    # Mask out undesired quality flag values?
    for _, qf_dict in config['include_if'].items():
        for cur_var_qf_col, cur_qf_vals in qf_dict.items():
            for val in cur_qf_vals:
                cur_qf_mask = \
                    np.asarray(subset_data_df[cur_var_qf_col] == val)
                # print(cur_qf_mask.shape)
                qf_masks.append(cur_qf_mask)

    if qf_masks:
        final_mask = np.zeros(shape=(subset_data_df.shape[0],))
        for array in qf_masks:
            final_mask += array

        # Use only entries where all masks are True. TODO: is this desired?
        final_mask[final_mask < len(qf_masks)] = 0.0
        final_mask[final_mask == len(qf_masks)] = 1.0
        final_mask = np.array(final_mask, dtype=bool)

        # Subset data
        subset_data_df = subset_data_df.loc[final_mask]

    print(f"{subset_data_df.shape=}")

    subset_data_df.to_csv(config['save_path'], index=False)


if __name__ == '__main__':
    main()
