#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 13:52:54 2022

@author: kristaal
"""

from typing import Any, List, Optional
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from fatescal.ces import priors
from fatescal.ces import daalgorithms as da

from fatescal.config import SIGMA_P, MCMC_CHAIN_LENGTH, \
    ADAPTIVE_MCMC, FATES_PARAMS_TO_PERTURBATE, INCLUDED_PFT_INDICES, \
    PROJECT_ROOT_PATH, CASE_NAME, BURN_IN_FRACTION

MCMC_CHAIN_SAVE_DIR = PROJECT_ROOT_PATH / 'data' / 'results' / \
    'mcmc_chains' / CASE_NAME


def emulator_mcmc(
    starting_parameters: pd.DataFrame,
    emulator_model: Any,
    scale_nll_by_gpr_std: bool = False,
    std_penalty_factor : float = 1.0,
    scaler: Optional[Any] = None,
    sigma_p=SIGMA_P,
    chain_length: int = MCMC_CHAIN_LENGTH,
    burn_in_fraction: float = BURN_IN_FRACTION,
    adaptive: bool = ADAPTIVE_MCMC,
    params_to_perturbate: List[str] = FATES_PARAMS_TO_PERTURBATE,
    included_pft_indices: List[int] = INCLUDED_PFT_INDICES,
    save_chain: bool = True,
    mcmc_chain_save_stem: str = "mcmc_chain",
    mcmc_chain_save_dir: Path | str = MCMC_CHAIN_SAVE_DIR,
) -> pd.DataFrame:
    '''
    MCMC implementation where the negative log-likelihood is predicted
    with a pre-trained emulator.

    'starting_parameters': pd.DataFrame with shape (1, n_parameters) and
        column names representing the parameter names.
    '''

    n_parameters = starting_parameters.shape[1]

    if isinstance(starting_parameters, pd.DataFrame):
        param_names = starting_parameters.columns
        starting_parameters = starting_parameters.copy().to_numpy()
    else:
        param_names = [str(idx+1) for idx in range(n_parameters)]

    # Prior scale factor
    prior_stds = np.array(
        [np.array(x['prior']['scale'])[included_pft_indices]
            for x in params_to_perturbate.values()]
    ).flatten()
    # Prior bounds
    lower_bounds = np.asarray(
        [np.array(x['prior']['lower_bound'])[included_pft_indices]
            for x in params_to_perturbate.values()]
    ).flatten()
    upper_bounds = np.asarray(
        [np.array(x['prior']['upper_bound'])[included_pft_indices]
            for x in params_to_perturbate.values()]
    ).flatten()

    # Transforming prior means
    model_space_prior_means = []
    for param_dict_value in params_to_perturbate.values():
        if param_dict_value['prior']['location'] == 'default_value':
            model_space_prior_means.append(
                np.array(param_dict_value['default_vals'])[included_pft_indices]
            )
        elif param_dict_value['prior']['location'] == 'center_bounds':
            model_space_prior_means.append(
                np.mean(
                    [
                        np.array(param_dict_value['prior']['upper_bound'])[included_pft_indices],
                        np.array(param_dict_value['prior']['lower_bound'])[included_pft_indices],
                    ],
                    axis=0
                )
            )
        else:
            raise NotImplementedError(
                f"Prior location {param_dict_value['prior']['location']} not implemented!"
            )
    # Combine into 1d array
    model_space_prior_means = np.asarray(model_space_prior_means).flatten()

    # Transform
    transformed_prior_means = priors.generalized_logit(
        x_vals=model_space_prior_means,
        lower_bound=lower_bounds,
        upper_bound=upper_bounds
    )

    # START MCMC
    start_time = datetime.now()

    # MCMC parameters
    C0 = (sigma_p**2)*np.eye(n_parameters)
    Sc = np.linalg.cholesky(C0)
    n_param_identity_mat = np.eye(n_parameters)

    # Initialize Markov chain filled with zeros
    # len(vars_to_perturbate)))
    mcmc_storage = np.zeros(shape=(chain_length, n_parameters))
    mcmc_storage[:] = np.nan

    # Init chain
    phic = np.reshape(
        starting_parameters[0, :],
        newshape=(1, starting_parameters.shape[1])
    )

    if scaler is None:
        # Penalize GPR predictions at uncertain regions
        # by inflating neg. log-lik. prediction with
        # prediction's standard deviation
        if scale_nll_by_gpr_std:

            nll, nll_std = emulator_model.predict(
                phic,
                return_std=True,
            )

            nll += nll_std * std_penalty_factor

            # return # FOR TESTING
        else:
            nll = emulator_model.predict(phic)
    else:
        nll = scaler.inverse_transform(
            emulator_model.predict(phic).reshape(-1, 1)
        )

    Uc = da.negative_log_posterior(
        negative_log_likelihood=nll,
        transformed_theta=phic,
        SD0=prior_stds,
        gaussian_prior_means=transformed_prior_means
    )

    n_accepted_samples: int = 0

    for chain_idx in range(chain_length):

        std_normal_r = np.random.randn(n_parameters)

        prop = Sc @ std_normal_r
        phip = phic + prop

        if scaler is None:
            # Penalize GPR predictions at uncertain regions
            # by inflating neg. log-lik. prediction with
            # prediction's standard deviation
            if scale_nll_by_gpr_std:

                nll, nll_std = emulator_model.predict(
                    phip,
                    return_std=True,
                )
                nll += nll_std * std_penalty_factor
            else:
                nll = emulator_model.predict(phip)
        else:
            nll = scaler.inverse_transform(
                emulator_model.predict(phip).reshape(-1, 1)
            )

        Up = da.negative_log_posterior(
            negative_log_likelihood=nll,
            transformed_theta=phip,
            SD0=prior_stds,
            gaussian_prior_means=transformed_prior_means
        )

        mh = min(1, np.exp(-Up+Uc))
        u = np.random.rand(1)
        accept = (mh > u)
        if accept:
            phic = phip
            Uc = Up
            n_accepted_samples += 1

        mcmc_storage[chain_idx, :] = phic

        # If adaptive, update proposal covariance for next step.
        # RAM algorithm by Vihola (https://doi.org/10.1007/s11222-011-9269-5)
        if adaptive:
            mhopt = 0.234  # Hard coded hyper-parameters for RAM
            gam = 2.0/3.0
            stepc = chain_idx + 1  # Step counter with 1-based indexing.
            eta = min(1, n_parameters*stepc**(-gam))
            rinner = std_normal_r @ std_normal_r
            router = np.outer(std_normal_r, std_normal_r)
            roi = router / rinner
            Cp = Sc @ (n_param_identity_mat + eta * (mh-mhopt) * roi) @ (Sc.T)
            Sc = np.linalg.cholesky(Cp)

        if ((chain_idx + 1) % 10000) == 0:
            print(
                f"Finished MCMC iterations: {chain_idx+1} of {chain_length} - Time elapsed: {datetime.now() - start_time}"
            )

    print(
        f"\nFinished MCMC. adaptive={adaptive}, acceptance rate={n_accepted_samples / chain_length}."
    )

    # Transform back to model space
    transformed_mcmc_chain = da.transform_included_fates_parameters(
        parameter_matrix=mcmc_storage,
        how='to_model_space'
    )

    transformed_mcmc_chain = pd.DataFrame(
        transformed_mcmc_chain,
        columns=param_names
    )

    if not mcmc_chain_save_dir.is_dir():
        mcmc_chain_save_dir.mkdir(parents=True)

    if save_chain:
        transformed_mcmc_chain.to_csv(
            mcmc_chain_save_dir / f"{mcmc_chain_save_stem}_full.csv",
            index=None
        )

    # Remove burn-in
    n_burned_samples = int(mcmc_storage.shape[0] * burn_in_fraction)

    mcmc_chain_pruned_df = pd.DataFrame(
        transformed_mcmc_chain.loc[n_burned_samples:, :],
        columns=param_names
    )

    if save_chain:
        mcmc_chain_pruned_df.to_csv(
            mcmc_chain_save_dir / f"{mcmc_chain_save_stem}_pruned.csv",
            index=None
        )

    return mcmc_chain_pruned_df
