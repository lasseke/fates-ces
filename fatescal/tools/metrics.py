#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions to calculate metrics such as error statistics.
"""

from typing import Any, Tuple, List
import pandas as pd
import numpy as np
from scipy.stats import linregress

def rmse(
    actual: np.ndarray,
    predicted: np.ndarray
) -> float | np.ndarray:
    '''Calculates Root Mean Square Error between two arrays'''

    return np.sqrt(np.nanmean((np.asarray(actual) - np.asarray(predicted))**2, axis=0))

def rmspe(
    actual: np.ndarray,
    predicted: np.ndarray
) -> np.ndarray:
    '''Calculates RMSE as a percentage of actual values.'''

    try:
        result = (
            np.sqrt(np.mean(np.square((actual - predicted) / actual)))
        ) * 100
    except ZeroDivisionError as zero_error:
        raise zero_error(
            "There is a zero in 'actual', consider adding a small constant!"
        )

    return result

def bias(
    actual: np.ndarray, predicted: np.ndarray
) -> float | np.ndarray:
    '''Calculates bias between two arrays'''

    return np.nanmean(actual - predicted)

def linear_regression(
    actual: np.ndarray, predicted: np.ndarray
) -> Any:
    '''
    Returns scipy.stats.linregress object with e.g.
    slope, intercept, r_value, p_value, std_err.

    Discards nan values in both x and y.
    '''

    actual = np.array(actual).reshape(actual.shape[0], 1)
    predicted = np.array(predicted).reshape(predicted.shape[0], 1)
    nan_mask = (np.isnan(actual) | np.isnan(predicted))

    return linregress(actual[~nan_mask], predicted[~nan_mask])


def ensemble_mean_max_min(
    ensemble_df_list: List[pd.DataFrame],
    column_name: str,
) -> Tuple[np.array, np.array, np.array]:
    '''
    'ensemble_df_list': assumes list of pd.DataFrames, each representing one
    ensemble member with shape (n_obs, n_variables).

    Returns:
        mean, maximum, minimum values
    '''
    # Calculate ensemble means and range
    n_obs = ensemble_df_list[0].shape[0]
    n_ensemble_members = len(ensemble_df_list)

    # Priors
    ensemble_sum = np.zeros(shape=(n_obs,))
    ensemble_max = ensemble_df_list[0][column_name]
    ensemble_min = ensemble_df_list[0][column_name]

    for ensemble_df in ensemble_df_list:
        ensemble_sum = ensemble_sum + ensemble_df[column_name]
        ensemble_max = np.maximum(ensemble_max, ensemble_df[column_name])
        ensemble_min = np.minimum(ensemble_min, ensemble_df[column_name])

    ensemble_mean = ensemble_sum / n_ensemble_members

    return ensemble_mean, ensemble_max, ensemble_min


def get_ensemble_error_metrics(
    ensemble_df_list: List[pd.DataFrame],
    col_name: str,
    actual: np.ndarray,
) -> Tuple[float, float, float, float, float, float]:
    '''Coming.'''

    stats = ["RMSE", "r2", "bias"]

    stats_df = pd.DataFrame(
        data=np.zeros(shape=(
            len(ensemble_df_list),
            len(stats)
        )),
        columns=stats,
    )

    for ens_idx, ensemble_df in enumerate(ensemble_df_list):

        # RMSE
        stats_df.loc[ens_idx, "RMSE"] = rmse(
            actual=actual,
            predicted=ensemble_df[col_name].values,
        )

        # r2
        _, _, r_value, _, _ = linear_regression(
                actual=actual,
                predicted=ensemble_df[col_name].values,
        )
        stats_df.loc[ens_idx, "r2"] = r_value**2

        # bias
        stats_df.loc[ens_idx, "bias"] = bias(
            actual=actual,
            predicted=ensemble_df[col_name].values,
        )

    return np.mean(stats_df["RMSE"]), np.std(stats_df["RMSE"]), \
        np.mean(stats_df["r2"]), np.std(stats_df["r2"]), \
        np.mean(stats_df["bias"]), np.std(stats_df["bias"])
