#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Additional helper functionalities for handling CTSM-FATES.
"""

from typing import List
from pathlib import Path
import xarray as xr

from fatescal.modelhandling.model_config import ModelConfig

model_cfg = ModelConfig()


def get_default_param_values(
        param_name: str,
        fates_param_file: str | Path,
) -> List[float | int]:
    '''
    Returns the default values for a FATES parameter.
    Used in generating parameter priors.
    '''

    # Input validation for parameter file
    if (not Path(fates_param_file).is_file()) or \
            (not str(fates_param_file).endswith('.nc')):
        raise ValueError(
            f"'{fates_param_file}' is not a '.nc' file or does not exist!"
        )

    param_nc_file = xr.open_dataset(fates_param_file)

    nc_var = param_nc_file[param_name]

    return list(nc_var.values.flatten())
