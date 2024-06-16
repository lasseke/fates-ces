#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emulator analysis visualization.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fatescal.config import PROJECT_ROOT_PATH, CASE_NAME
from fatescal.tools import metrics

DEFAULT_SAVE_PATH: Path = PROJECT_ROOT_PATH / 'data' / \
    'results' / 'plots' / 'emulation' / CASE_NAME
FIG_SIZE = (8/2.54, 8/2.54)


def plot_emulator_evaluation(
    actual: np.ndarray,
    predicted: np.ndarray,
    metric_code: str = "nll",
    file_name: str = "emulator_predicted_vs_actual_testset.png",
    save_path: Path | str = DEFAULT_SAVE_PATH,
    save_fig: bool = True,
    ax: plt.Axes | None = None,
) -> None:
    '''Plot emulator results'''

    if not save_path.is_dir():
        save_path.mkdir(parents=True)

    if actual.ndim == 1:
        actual.reshape(actual.shape[0], 1)
    if predicted.ndim == 1:
        predicted.reshape(predicted.shape[0], 1)

    rmse = metrics.rmse(
        actual=actual,
        predicted=predicted
    )

    lin_reg = metrics.linear_regression(
        actual=actual,
        predicted=predicted
    )

    # Plot
    if ax is None:
        fig, ax = plt.subplots(
            figsize=FIG_SIZE,
            dpi=300
        )

    ax.scatter(
        actual,
        predicted,
    )

    # Quick fix
    if isinstance(rmse, float):
        rmse = [rmse]

    metric_str = f"y = {round(lin_reg.slope, 2)}x{' + ' if lin_reg.intercept >= 0 else ' - '}{np.abs(round(lin_reg.intercept, 2))}\n" \
        + f"r² = {round(lin_reg.rvalue**2, 4)}; p {'< 0.001' if lin_reg.pvalue < 0.001 else f'= {round(lin_reg.pvalue, 3)}'}" \
        + f"\nRMSE = {np.round(rmse[0], 2)}"

    ax.text(
        .01, .99,
        metric_str,
        ha='left',
        va='top',
        transform=ax.transAxes
    )

    # Add trend line
    x_vals = np.array(ax.get_xlim())
    y_vals = lin_reg.intercept + lin_reg.slope * x_vals
    ax.plot(x_vals, y_vals, '--', color='red', zorder=999)

    if metric_code == 'nll':
        ax.set_xlabel("Actual negative log-likelihood")
        ax.set_ylabel("Predicted negative log-likelihood")

    ax.axline((0, 0), slope=1, color="black")

    ax.set_xlim(
        left=min(min(actual), min(predicted)),
        right=max(max(actual), max(predicted))
    )
    ax.set_ylim(
        bottom=min(min(actual), min(predicted)),
        top=max(max(actual), max(predicted))
    )

    if ax is None:
        fig.tight_layout()

    if (save_fig) and (ax is None):
        fig.savefig(save_path / file_name)

    return ax
