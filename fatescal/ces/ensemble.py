#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Defines functionalities for ensemble runs testing the
"Calibrate, simulate, sample" approach for data
assimilation.

Created on Fri Mar 17, 2022

@author: Kristoffer Aalstad, Esteban Alonso González, Lasse Keetz

Needs to be improved.
'''

import pickle
from typing import List
from pathlib import Path
import numpy as np
import pandas as pd
from fatescal.config import N_ENSEMBLE_MEMBERS, \
    PROJECT_ROOT_PATH, CASE_NAME, N_KALMAN_ITERATIONS, N_PFTS
from fatescal.ces import priors

DEFAULT_CSV_SAVE_PATH: Path = PROJECT_ROOT_PATH / \
    'data' / 'results' / 'fates_param_ensembles' / CASE_NAME / 'csv'
DEFAULT_PKL_SAVE_PATH: Path = PROJECT_ROOT_PATH / \
    'data' / 'results' / 'fates_param_ensembles' / CASE_NAME / 'pkl'


class Ensemble:
    '''Helper class to create and manage ensembles'''

    def __init__(
            self,
            n_ensembles: int = N_ENSEMBLE_MEMBERS
    ) -> None:
        '''
        Initializes an ensemble with 'n_ensembles' ensemble members
        and given 'error_variance_y' for random perturbations of
        observations (to account for measurement uncertainties).
        '''

        self.n_ensembles = n_ensembles
        self.theta_df = None
        self.current_kalman_iter = 0
        self.n_kalman_iters = N_KALMAN_ITERATIONS

        # Instantiate prior meta information DataFrame
        self.prior_meta_df = pd.DataFrame(
            columns=[
                "dist_name",
                "location",
                "scale",
                "upper_bounds",
                "lower_bounds"
            ]
        )

    def create_initial_priors(
            self,
            parameter_names: str | List[str],
            prior_dist_name: str | List[str] = 'standard_normal',
            prior_dist_location: float | List[float] = 1.0,
            prior_dist_scale: float | List[float] = 0.1,
            upper_bound: float | List[float] = [],
            lower_bound: float | List[float] = []
    ) -> pd.DataFrame:
        '''
        Creates an initial parameter prior matrix for the given
        parameter names. Draws random numbers from the specified
        prior distributions for each ensemble member.

        If 'parameter_names' is a list, 'prior_dist' can either
        be a single string (-> uses same prior distribution for all
        parameters) or a list of strings. In the latter case,
        the list must specify a distribution for each element in
        'parameter_names'.

        'prior_dist_params' uses the same logic and accepts additional
        arguments for the distribution sampling functions used. The
        keys should be argument names, and values corresponding values.

        Returns a pandas DataFrame with column names corresponding
        to the parameters, and rows corresponding to the ensemble
        members.
        '''

        implemented_priors = ('standard_normal', 'normal', 'generalized_logit')

        # Put single 'prior_dist' str into a list for easier handling
        if isinstance(prior_dist_name, str):
            prior_dist_name = [prior_dist_name]
        # Put single 'parameter_names' str into a list for easier handling
        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]

        # Put single 'upper_bound' str into a list for easier handling
        if not isinstance(upper_bound, list):
            upper_bound = [upper_bound]
        # Put single 'lower_bound' str into a list for easier handling
        if not isinstance(lower_bound, list):
            lower_bound = [lower_bound]

        if not isinstance(prior_dist_location, list):
            prior_dist_location = [prior_dist_location]
        if not isinstance(prior_dist_scale, list):
            prior_dist_scale = [prior_dist_scale]

        for dist in prior_dist_name:
            if dist not in implemented_priors:
                raise ValueError(
                    f"At least one element in '{prior_dist_name}' not "
                    f"implemented! Use one of: {implemented_priors}"
                )

        # TODO: Add checks for correct list lengths
        prior_df = pd.DataFrame(columns=parameter_names)

        # TODO: Add informative summary what is done
        print(f"Generating priors for parameters:\n {parameter_names}")

        n_params = len(parameter_names)

        # Loop through parameters
        for idx, param in enumerate(parameter_names):

            # Only one prior distribution?
            if (len(prior_dist_name) == 1):

                # Store prior information (used for transformations)
                # TODO: Doesn't make sense currently, needs rework
                self.prior_meta_df.loc[(
                    N_PFTS*idx):(N_PFTS*(idx+1)), "dist_name"] = prior_dist_name[0]
                self.prior_meta_df.loc[(
                    N_PFTS*idx):(N_PFTS*(idx+1)), "location"] = prior_dist_location[0]
                self.prior_meta_df.loc[(
                    N_PFTS*idx):(N_PFTS*(idx+1)), "scale"] = prior_dist_scale[0]
                self.prior_meta_df.loc[(
                    N_PFTS*idx):(N_PFTS*(idx+1)), "upper_bounds"] = upper_bound[0]
                self.prior_meta_df.loc[(
                    N_PFTS*idx):(N_PFTS*(idx+1)), "lower_bounds"] = lower_bound[0]

                if prior_dist_name[0] == 'standard_normal':
                    # Return normal with mean 0, sd 1
                    prior_df[param] = 1*np.random.randn(
                        1, self.n_ensembles
                    ).flatten()

                if prior_dist_name[0] == 'normal':
                    # return normal with given mean and sd
                    prior_df[param] = 1*np.random.normal(
                        loc=prior_dist_location[0],
                        scale=prior_dist_scale[0],
                        size=(1, self.n_ensembles)
                    ).flatten()

                if prior_dist_name[0] == 'generalized_logit':

                    if (upper_bound[0] is None) or (lower_bound[0] is None):
                        raise ValueError(
                            "Must provide 'upper_bound' and 'lower_bound' args for 'generalized_logit'!"
                        )

                    # Transform the median in the physical space (==true CLM params) to transformed space
                    transformed_location = priors.generalized_logit(
                        x_vals=prior_dist_location[0],
                        lower_bound=lower_bound[0],
                        upper_bound=upper_bound[0]
                    )

                    transformed_samples = (
                        transformed_location +
                        np.random.randn(self.n_ensembles) * prior_dist_scale[0]
                    ).flatten()

                    # Need to store the location parameter and bounds etc.
                    prior_df[param] = priors.generalized_expit(
                        transformed_samples,
                        lower_bound=lower_bound[0],
                        upper_bound=upper_bound[0]
                    )

            # Else, are as many prior dist names as params provided?
            elif len(prior_dist_name) == n_params:

                # Store prior information (used for transformations)
                self.prior_meta_df.loc[param,
                                       "dist_name"] = prior_dist_name[idx]
                self.prior_meta_df.loc[param,
                                       "location"] = prior_dist_location[idx]
                self.prior_meta_df.loc[param, "scale"] = prior_dist_scale[idx]
                self.prior_meta_df.loc[param,
                                       "upper_bounds"] = upper_bound[idx]
                self.prior_meta_df.loc[param,
                                       "lower_bounds"] = lower_bound[idx]

                if prior_dist_name[idx] == 'standard_normal':
                    # Return normal with mean 0, sd 1
                    prior_df[param] = 1*np.random.randn(
                        1, self.n_ensembles
                    ).flatten()

                elif prior_dist_name[idx] == 'normal':
                    # return normal with given mean and sd
                    prior_df[param] = 1*np.random.normal(
                        loc=prior_dist_location[idx],
                        scale=prior_dist_scale[idx],
                        size=(1, self.n_ensembles)
                    ).flatten()

                elif prior_dist_name[idx] == 'generalized_logit':

                    if (upper_bound[idx] is None) or (lower_bound[idx] is None):
                        raise ValueError(
                            "Must provide 'upper_bound' and 'lower_bound' args for 'generalized_logit'!"
                        )

                    # Transform the median in the physical space (==true CLM params) to transformed space
                    transformed_location = priors.generalized_logit(
                        x_vals=prior_dist_location[idx],
                        lower_bound=lower_bound[idx],
                        upper_bound=upper_bound[idx]
                    )

                    transformed_samples = (
                        transformed_location +
                        np.random.randn(self.n_ensembles) *
                        prior_dist_scale[idx]
                    ).flatten()

                    # Need to store the location parameter and bounds etc.
                    prior_df[param] = priors.generalized_expit(
                        x_vals_transformed=transformed_samples,
                        lower_bound=lower_bound[idx],
                        upper_bound=upper_bound[idx]
                    )

                else:
                    print(prior_dist_name[idx])
                    raise ValueError(
                        f'''
                        Wrong number of arguments! See documentation!
                        {len(prior_dist_name)=}
                        {len(prior_dist_location)=}
                        {len(prior_dist_scale)=}
                        {len(lower_bound)=}
                        '''
                    )

            else:
                raise ValueError(
                    f'''
                    Wrong number of arguments! See documentation!
                    {len(prior_dist_name)=}
                    {len(prior_dist_location)=}
                    {len(prior_dist_scale)=}
                    {len(lower_bound)=}
                    '''
                )

        self.theta_df = prior_df

        print(
            f"Created parameter priors for {self.n_ensembles} ensemble members with the "
            f"following properties:\n{self.prior_meta_df}"
        )

        return prior_df

    def save_as_csv(
            self,
            file_name: str,
            save_dir: str | Path = DEFAULT_CSV_SAVE_PATH,
            mcmc: bool = False,
            latin_hyper_cube: bool = False,
            synth_truth_run: bool = False,
    ) -> None:
        '''Save parameter matrix in a csv file.'''

        if not (save_dir := Path(save_dir)).is_dir():
            print(f"Creating '{save_dir}'...")
            save_dir.mkdir(parents=True)

        if mcmc:
            save_dir_path = Path(save_dir) / \
                "mcmc"
        elif latin_hyper_cube:
            save_dir_path = Path(save_dir) / \
                "latin_hyper_cube"
        elif synth_truth_run:
            save_dir_path = Path(save_dir) / 'synth_truth_run'
        else:
            save_dir_path = Path(save_dir) / \
                f"kalman_iter_{self.current_kalman_iter}"

        if not save_dir_path.is_dir():
            print(f"Creating '{save_dir_path}'...")
            save_dir_path.mkdir()

        if not file_name.endswith(".csv"):
            file_name += ".csv"

        if self.theta_df is not None:
            self.theta_df.to_csv(
                save_dir_path / file_name,
                index=True,
                encoding='utf-8'
            )
        else:
            raise ValueError("No ensemble matrix created yet!")

        print(
            f"Succesfully saved parameter table in '{save_dir_path / file_name}'."
        )

    def save_as_pkl(
            self,
            file_name: str,
            save_dir: str | Path = DEFAULT_PKL_SAVE_PATH,
            mcmc: bool = False,
            latin_hyper_cube: bool = False,
            synth_truth_run: bool = False,
    ) -> None:
        '''Save Ensemble object as a pickle to reuse later.'''

        if not (save_dir := Path(save_dir)).is_dir():
            print(f"Creating '{save_dir}'...")
            save_dir.mkdir(parents=True)

        if mcmc:
            save_dir_path = Path(save_dir) / \
                "mcmc"
        elif latin_hyper_cube:
            save_dir_path = Path(save_dir) / \
                "latin_hyper_cube"
        elif synth_truth_run:
            save_dir_path = Path(save_dir) / 'synth_truth_run'
        else:
            save_dir_path = Path(save_dir) / \
                f"kalman_iter_{self.current_kalman_iter}"

        if not save_dir_path.is_dir():
            save_dir_path.mkdir()

        if not file_name.endswith(".pkl"):
            file_name += ".pkl"

        with open(save_dir_path / file_name, "wb") as pkl_file:
            pickle.dump(self, pkl_file, pickle.HIGHEST_PROTOCOL)

        print(
            f"Succesfully saved Ensemble object in '{save_dir_path / file_name}'."
        )
