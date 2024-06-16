#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data assimilation algorithms.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from fatescal.config import INCLUDED_PFT_INDICES, FATES_PARAMS_TO_PERTURBATE
from fatescal.ces import priors


def transform_included_fates_parameters(
    parameter_matrix: np.ndarray | pd.DataFrame,
    how: str,
) -> pd.DataFrame:
    '''
    Assumed shape 'parameter_matrix': (n_ensemble_members, n_active_parameters),
    where 'n_active_parameters' must be ordered in the same way as would be inferred
    from the settings file with ascending active PFT indices.
    '''

    implemented_transformations = (
        'to_gaussian_space',
        'to_model_space',
    )

    if how not in implemented_transformations:
        raise ValueError(
            f"'{how=}' not implemented! Use one of: {implemented_transformations}"
        )

    # Convert Fortran to Python indices for var transformation later
    included_pft_indices = [x-1 for x in INCLUDED_PFT_INDICES]

    # Instantiate transformed matrix
    transformed_parameter_matrix = np.zeros(shape=parameter_matrix.shape)

    column_names = []
    if isinstance(parameter_matrix, pd.DataFrame):
        column_names = parameter_matrix.columns
        parameter_matrix = parameter_matrix.to_numpy()

    # Start transformation
    param_idx = 0

    # Transform active parameter matrix to Gaussian space
    for fates_param_option_dict in FATES_PARAMS_TO_PERTURBATE.values():

        cur_active_param_lower_bounds = \
            [fates_param_option_dict['prior']['lower_bound'][idx]
                for idx in included_pft_indices]
        cur_active_param_upper_bounds = \
            [fates_param_option_dict['prior']['upper_bound'][idx]
                for idx in included_pft_indices]

        for cur_lower_bound, cur_upper_bound in zip(
            cur_active_param_lower_bounds,
            cur_active_param_upper_bounds
        ):

            if how == 'to_gaussian_space':
                transformed_parameter_matrix[:, param_idx] = \
                    priors.generalized_logit(
                        x_vals=parameter_matrix[:, param_idx],
                        lower_bound=cur_lower_bound,
                        upper_bound=cur_upper_bound
                )
            elif how == 'to_model_space':
                transformed_parameter_matrix[:, param_idx] = \
                    priors.generalized_expit(
                        x_vals_transformed=parameter_matrix[:, param_idx],
                        lower_bound=cur_lower_bound,
                        upper_bound=cur_upper_bound
                )

            param_idx += 1

    if len(column_names) != 0:
        transformed_parameter_matrix = pd.DataFrame(
            transformed_parameter_matrix,
            columns=column_names
        )

    return transformed_parameter_matrix


def tsubspaceEnKA(
        theta_mat, y_observed_peturbed_mat, y_predicted_mat,
        svdt: float = 0.9
) -> np.ndarray:
    """
    tsubspaceEnKA: Implmentation of the Ensemble Kalman Analysis in the
    ensemble subspace. This scheme is more robust in the regime where you have
    a larger number of observations and/or states and/or parameters than
    ensemble members.

    Inputs:
        theta_mat: Parameter ensemble matrix (n x n_ensemble_members array)
        y_observed_peturbed_mat: Perturbed observation ensemble matrix
        (m x n_ensemble_members array)
        y_predicted_mat: Predicted observation ensemble matrix
        (m x n_ensemble_members array)
        svdt: Level of truncation of singular values for pseudoinversion,
        recommended=0.9 (90%)
    Outputs:
        post: Posterior ensemble matrix (n x n_ensemble_members array)
    Dimensions:
        n_ensemble_members is the number of ensemble members, n is the number
        of state variables and/or parameters, and m is the number of
        observations.

    The implementation follows that described in Algorithm 6 in the book of
    Evensen et al. (2022), while adopting the truncated SVD procedure described
    in Emerick (2016) which also adopts the ensemlbe supspace method to the
    ES-MDA.

    Note that the observation error covariance R is defined implicitly here
    through the perturbed observations.
    This matrix (Y) should be perturbed in such a way that it is consistent
    with R in the case of single data
    assimilation (no iterations) or alpha*R in the case of multiple data
    assimilation (iterations). Moreover,
    although we should strictly be perturbing the predicted observations this
    does not make any difference in
    practice (see van Leeuwen, 2020) and simplifies the implmentation of the
    ensemble subspace approach.

    References:
        Evensen et al. 2022: https://doi.org/10.1007/978-3-030-96709-3
        Emerick 2016: https://doi.org/10.1016/j.petrol.2016.01.029
        van Leeuwen 2020: https://doi.org/10.1002/qj.3819

    Code by K. Aalstad (last revised December 2022)
    """

    n_ensemble_members = np.shape(theta_mat)[1]  # Number of ensemble members
    n_observations = np.shape(y_observed_peturbed_mat)[0]
    identity_mat_n_ens_memb = np.eye(n_ensemble_members)

    # Anomaly operator (subtracts ensemble mean)
    anomaly_operator = (
        identity_mat_n_ens_memb
        - np.ones([n_ensemble_members, n_ensemble_members])
        / n_ensemble_members
    ) / np.sqrt(n_ensemble_members - 1)

    # Observation anomalies
    y_obs_anomalies = y_observed_peturbed_mat@anomaly_operator
    # Predicted observation anomalies
    y_predicted_anomalies = y_predicted_mat@anomaly_operator

    S = y_predicted_anomalies
    # Singular value decomposition
    [U, E, _] = np.linalg.svd(S, full_matrices=False)
    Evr = np.cumsum(E)/np.sum(E)
    N = min(n_ensemble_members, n_observations)
    these = np.arange(N)
    try:
        Nr = min(these[Evr > svdt])  # Truncate small singular values
    except Exception as exception:
        raise exception

    these = np.arange(Nr+1)  # Exclusive python indexing
    E = E[these]
    U = U[:, these]
    Ei = np.diag(1/E)
    P = Ei@(U.T)@y_obs_anomalies@(y_obs_anomalies.T)@U@(Ei.T)
    [Q, L, _] = np.linalg.svd(P)
    LpI = L+1
    LpIi = np.diag(1/LpI)
    UEQ = U@(Ei.T)@Q
    # Pseudo-inversion of C=(C_YY+alpha*R) in the ensemble subspace
    Cpinv = UEQ@LpIi@UEQ.T
    innovation_mat = y_observed_peturbed_mat-y_predicted_mat  # Innovation
    W = (S.T)@Cpinv@innovation_mat
    T = (identity_mat_n_ens_memb+W/np.sqrt(n_ensemble_members-1))
    Xu = theta_mat@T  # Update

    return Xu


def negative_log_posterior(
    negative_log_likelihood: np.ndarray,
    transformed_theta: np.ndarray,
    SD0,
    gaussian_prior_means
) -> np.ndarray:
    '''
    Code by K. Aalstad.
    '''

    ndev = (transformed_theta - gaussian_prior_means) / SD0

    negative_log_prior = 0.5 * np.sum(ndev**2, axis=1)

    return negative_log_likelihood + negative_log_prior


def negative_log_likelihood(
    actual: ArrayLike,
    predicted: ArrayLike,
    error_covariances: ArrayLike
) -> np.ndarray:
    '''
    Calculates the negative log likelihood for arrays of observed and
    modelled target variables.

    The column indices in 'actual' and 'predicted' must match, i.e.
    column [:, X] in both arrays must contain corresponding entries
    (same point in time) of the same target variable.

    'error_covariances' must contain variances, i.e., sqrt(error_std)!
    '''

    if actual.shape != predicted.shape:
        raise ValueError(
            "Must provide the same number of actual and predicted "
            + "target variables and observations!"
        )

    if len(error_covariances) != actual.shape[1]:
        raise ValueError(
            "Must provide the same number of error covariances "
            + "as number of target variables (one each)!"
        )

    # Cast inputs to required data types, if needed
    error_covariances = np.array(error_covariances)
    actual = pd.DataFrame(actual)
    predicted = pd.DataFrame(predicted)

    scaled_residual_sum: float = 0.0

    for variable_idx, var_error_cov in enumerate(error_covariances):
        scaled_residual_sum += sum(
            (actual.iloc[:, variable_idx] -
             predicted.iloc[:, variable_idx])**2 / var_error_cov
        )

    return 0.5 * scaled_residual_sum
