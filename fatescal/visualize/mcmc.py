#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCMC analysis visualization.
"""

import pandas as pd
import matplotlib.pyplot as plt

from fatescal.config import PROJECT_ROOT_PATH, CASE_NAME
#from fatescal.tools import metrics

def plot_mcmc_chain_histogram(
    mcmc_chain_df: pd.DataFrame,
    bins: int = 50,
    case_name : str = CASE_NAME,
    file_name: str = "mcmc_chain_histogram.png",
) -> None:
    '''Plot histogram from mcmc chain'''

    plot_save_path = PROJECT_ROOT_PATH / 'data' / 'results' \
        / 'plots' / 'mcmc' / case_name

    if not plot_save_path.is_dir():
        plot_save_path.mkdir(parents=True)

    n_rows = mcmc_chain_df.shape[1] // 2
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=2,
        sharey=True,
        figsize=(12/2.54, 20/2.54),
        dpi=300
    )

    mcmc_chain_df.hist(
        bins=bins,
        ax=axes
    )

    for idx, ax in enumerate(axes.flatten()):
        ax.set_xlabel(mcmc_chain_df.columns[idx], fontsize=6)
        ax.set_title("")

    fig.tight_layout()
    fig.savefig(
        plot_save_path / file_name
    )