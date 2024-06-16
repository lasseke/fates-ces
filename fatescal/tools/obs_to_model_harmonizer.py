#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper class to bring model outputs and tabular observation datasets
into the same format for data assimilation.

Variable names are specified in 'fatescal/config.py'.
"""

import glob
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import xarray as xr
from fatescal.tools import unit_conversion as uc
from fatescal.config import VARIABLES_TO_ASSIMILATE, \
    PROJECT_ROOT_PATH, CASE_NAME, FORCING_YR_START, FORCING_YR_END


HARMONIZED_CSV_DIR = PROJECT_ROOT_PATH / 'data' / 'results' / \
    'aligned_obs_model'

if not HARMONIZED_CSV_DIR.is_dir():
    HARMONIZED_CSV_DIR.mkdir(parents=True)


class ObsToModelHarmonizer:
    '''
    Harmonizes model netCDF output and tabular observation data (.csv),
    e.g. regarding time intervals and units.

    WARNING: requires the observations to have overlapping time steps
    with the model output data.
    '''

    def __init__(
            self,
            observation_csv_file_path: str | Path,
            model_nc_output_dir_path: str | Path,
            model_nc_hist_tape: str = 'h1',
            obs_time_col_name: str = 'samptime',
            obs_time_zone: str = 'infer',
    ) -> None:

        if not (observation_csv_file_path := Path(observation_csv_file_path)).is_file():
            raise ValueError(f"'{observation_csv_file_path}' does not exist!")
        if not (model_nc_output_dir_path := Path(model_nc_output_dir_path)).is_dir():
            raise ValueError(f"'{model_nc_output_dir_path}' does not exist!")

        # Extract variable names
        self.obs_time_col_name = obs_time_col_name
        self.obs_var_names = \
            [var['Observed']['csv_col_name']
                for var in VARIABLES_TO_ASSIMILATE]
        self.model_var_names = \
            [var['CLM-FATES']['history_var_name']
                for var in VARIABLES_TO_ASSIMILATE]

        # Extract variable units
        self.obs_var_units = \
            [var['Observed']['unit'] for var in VARIABLES_TO_ASSIMILATE]
        self.model_var_units = \
            [var['CLM-FATES']['unit'] for var in VARIABLES_TO_ASSIMILATE]

        # Set model period
        self.model_time_min = np.datetime64(f"{FORCING_YR_START}-01-01")
        self.model_time_max = np.datetime64(f"{FORCING_YR_END}-12-31")

        # MODEL DATA

        self.nc_file_paths = sorted(glob.glob(
            f"{model_nc_output_dir_path}/*{model_nc_hist_tape}*.nc"
        ))

        model_ds_list = []
        for nc_path in self.nc_file_paths:
            model_ds_list.append(
                xr.open_dataset(nc_path)
            )

        self.model_ds_df_list = []
        for ds in model_ds_list:
            cur_df = pd.DataFrame(columns=['time']+self.model_var_names)
            cur_df['time'] = ds.indexes['time'].to_datetimeindex()
            for var in self.model_var_names:
                cur_df[var] = ds[var].values

            # Remove dates outside model range
            date_mask = (cur_df['time'] >= self.model_time_min) & \
                (cur_df['time'] <= self.model_time_max)
            cur_df = cur_df.loc[date_mask]

            self.model_ds_df_list.append(cur_df)

        self.example_ds_df = self.model_ds_df_list[0]

        print(
            f"Read model data from '{model_nc_output_dir_path}' and "
            f"subset columns: '{self.example_ds_df.columns}'."
        )

        # OBSERVATIONS

        self.observation_csv_file_path = Path(observation_csv_file_path)
        self.obs_df = pd.read_csv(self.observation_csv_file_path)
        col_names = self.obs_df.columns
        drop_col_names = [col for col in col_names if col not in
                          (self.obs_var_names + [obs_time_col_name])]
        self.obs_df = self.obs_df.drop(columns=drop_col_names)

        # Adjust date of observations to model dates
        self._subset_obs_time_to_gswp3_model_output(
            obs_time_col_name=obs_time_col_name,
            obs_time_zone=obs_time_zone
        )

        # Convert units
        self._convert_obs_to_model_unit()

        print(
            f"Read observations from '{observation_csv_file_path.name}', "
            f"subset columns: '{self.obs_df.columns}', adjusted date "
            f"formats, converted units."
        )

        # Instantiate aggregated dfs
        self.aggregated_obs_df = None
        self.aggregated_model_df_list = []

    def _subset_obs_time_to_gswp3_model_output(
            self,
            obs_time_col_name: str = 'samptime',
            obs_time_zone: str = 'infer'
    ) -> None:
        '''
        Adjust format of time column:
          - convert to datetime objects for easier handling
          - adjusts timezone to GSWP3's timezone (UTC)
          - removes entries from Feb 29th leap days to be consistent with GSWP3
          - removes entries outside model output's time interval
        '''

        # Convert to datetime, adjust timezone
        # TODO: add new time formats if necessary
        if obs_time_zone == 'infer':
            self.obs_df[obs_time_col_name] = \
                pd.to_datetime(  # Convert to pandas datetime
                    self.obs_df[obs_time_col_name],
                    utc=True  # Convert to UTC
                ).dt.tz_localize(None)  # Remove timezone information
        else:
            raise NotImplementedError(
                f"Timezone argument '{obs_time_zone}' not implemented!"
            )

        # Remove leap year days
        no_leap_day_mask = [False if ((date.month == 2) and (date.day == 29))
                            else True for date in self.obs_df[obs_time_col_name]]
        print(
            f"Removing {len(no_leap_day_mask) - sum(no_leap_day_mask)} "
            "leap day entries from observation DataFrame."
        )
        self.obs_df = self.obs_df.loc[no_leap_day_mask]

        # Remove dates outside model range
        date_mask = (self.obs_df[obs_time_col_name] >= self.model_time_min) & \
            (self.obs_df[obs_time_col_name] <= self.model_time_max)
        self.obs_df = self.obs_df.loc[date_mask]

    def _convert_obs_to_model_unit(self) -> None:
        '''Converts between units specified in fatescal/config.py'''

        for obs_var_name, unit_obs, unit_model in \
                zip(self.obs_var_names, self.obs_var_units, self.model_var_units):

            self.obs_df[obs_var_name] = uc.convert_unit(
                values=self.obs_df[obs_var_name],
                unit_in=unit_obs,
                unit_out=unit_model
            )

    def aggregate(
            self,
            freq: str = '1M',
            how: str = 'mean',
            save_csv: bool = False,
            save_dir: str | Path = HARMONIZED_CSV_DIR / CASE_NAME,
            kalman_iter: int = 0,
            mcmc: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''TODO'''

        self.freq = freq
        self.how = how

        # Instantiate obs df
        self.aggregated_obs_df = self.obs_df.set_index(
            self.obs_time_col_name, drop=False
        )

        # Instantiate model dfs
        for model_ds_df in self.model_ds_df_list:
            self.aggregated_model_df_list.append(
                model_ds_df.set_index(
                    'time', drop=False
                )
            )

        if how == 'mean':
            self.aggregated_obs_df = self.aggregated_obs_df.groupby(
                pd.Grouper(freq=freq)
            ).mean()

            for idx in range(len(self.aggregated_model_df_list)):
                self.aggregated_model_df_list[idx] = \
                    self.aggregated_model_df_list[idx].groupby(
                        pd.Grouper(freq=freq)
                ).mean()
        else:
            # TODO: implement more
            pass

        # Save?
        if save_csv:
            self.save_aggregated_dfs_to_csv(
                save_dir=save_dir,
                kalman_iter=kalman_iter,
                mcmc=mcmc,
            )

        return self.aggregated_obs_df, self.aggregated_model_df_list[0]

    def save_aggregated_dfs_to_csv(
            self,
            save_dir: str | Path = HARMONIZED_CSV_DIR / CASE_NAME,
            kalman_iter: int = 0,
            mcmc: bool = False,
    ) -> None:

        if mcmc:
            save_dir = Path(save_dir) / 'mcmc'
        else:
            save_dir = Path(save_dir) / f'kalman_iter_{kalman_iter}'

        if not save_dir.is_dir():
            save_dir.mkdir(parents=True)

        aggregate_info_str = f'{self.freq}_{self.how}'

        for nc_file_path, model_ds_df in zip(
            self.nc_file_paths, self.aggregated_model_df_list
        ):
            model_ds_df.to_csv(
                Path(save_dir) /
                f"M_{aggregate_info_str}_{Path(nc_file_path).stem}.csv"
            )

        print(self.aggregated_obs_df)

        self.aggregated_obs_df.to_csv(
            save_dir /
            f"O_{aggregate_info_str}_{self.observation_csv_file_path.name}"
        )
