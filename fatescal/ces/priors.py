#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 13:52:54 2022

@author: kristaal
"""

import numpy as np
from scipy.special import expit, logit


def generalized_logit(
        x_vals: np.ndarray | float, lower_bound: float | int, upper_bound: float | int
) -> np.ndarray | float:
    r'''
    Generalized logit that transforms a logit-normal rv x_vals\in(a,b)
    to a normal r x_vals_transformed\in(-inf,\inf)
    '''

    # Scale x\in(lower_bound, upper_bound) to be between (0, 1) instead
    x_vals_scaled = (x_vals - lower_bound) / (upper_bound - lower_bound)
    x_vals_transformed = logit(x_vals_scaled)

    return x_vals_transformed


def generalized_expit(
        x_vals_transformed: np.ndarray, lower_bound: float | int, upper_bound: float | int
) -> np.ndarray | float:
    r'''
    Generalized expit that transforms a normal rv xt\in(-inf, inf) to a logit-normal rv x\in(a, b)
    '''

    x_vals_scaled = expit(x_vals_transformed)
    x_vals = (upper_bound-lower_bound) * x_vals_scaled+lower_bound

    return x_vals
