"""Helpers to handle CLM-FATES parameters in Python"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import json
import re
import pandas as pd
import numpy as np
from fatescal.modelhandling.model_config import ModelConfig
from fatescal.ces.ensemble import Ensemble
from fatescal.config import N_ENSEMBLE_MEMBERS, FATES_PARAMS_TO_PERTURBATE, \
    DEFAULT_JSON_SAVE_DIR
from fatescal.modelhandling.helpers import get_default_param_values

model_config = ModelConfig()

DEFAULT_FATES_PARAM_FILE_PATH = Path(
    model_config.model_root /
    model_config.default_fates_param_nc_file_path
)


class FatesDefaultParameter:
    """Retrieves and handles default FATES parameter attributes"""

    def __init__(
            self,
            param_name: str,
            fates_param_file: Path | str = DEFAULT_FATES_PARAM_FILE_PATH,
            dimensions: str = 'n_pfts',
            label: Optional[str] = None
    ) -> None:
        self.param_name = param_name
        self.dimensions = dimensions

        self.default_values: List[float] = get_default_param_values(
            param_name=self.param_name,
            fates_param_file=fates_param_file
        )

        self.attribute_dict = {
            "values": self.default_values,
            "label": label
        }

    def __str__(self) -> str:
        return str(f"Name: {self.param_name}\nAttrs: {self.attribute_dict}")


class FatesParameterEnsemble(Ensemble):
    '''
    Class to create and handle ensemble matrices based on default FATES
    parameter values.
    '''

    def __init__(self, n_ensembles=N_ENSEMBLE_MEMBERS):

        super().__init__(n_ensembles=n_ensembles)

        self.default_param_list = None
        self.ensemble_dict = None

    def set_default_parameters(
            self,
            parameter_list: List[FatesDefaultParameter]
    ) -> None:
        """Defines default FATES parameters to include in ensemble"""

        self.default_param_list = parameter_list

    def generate_priors(self) -> pd.DataFrame:
        '''Generates priors based on default parameter values'''

        if not self.default_param_list:
            raise ValueError("You must add default parameters first!")

        param_names = []
        prior_dist_names = []
        param_location_list = []
        param_scale_list = []
        param_upper_bound_list = []
        param_lower_bound_list = []

        for parameter in self.default_param_list:

            # TODO: add more cases later if needed
            if parameter.dimensions == 'n_pfts':

                cur_name = parameter.param_name
                n_param_values = len(parameter.default_values)

                # Append PFT number to names for file column naming convention
                param_names += \
                    [f"{cur_name}_PFT{idx+1}" for idx in range(n_param_values)]

                # Read current parameter prior settings
                param_cfg_dict = FATES_PARAMS_TO_PERTURBATE[cur_name]

                # Create lists of means and corresponding st. deviations
                if param_cfg_dict['prior']['location'] == 'default_value':
                    param_location_list += parameter.default_values
                elif param_cfg_dict['prior']['location'] == 'center_bounds':
                    param_location_list += list(
                        np.mean(
                            [param_cfg_dict['prior']['upper_bound'],
                                param_cfg_dict['prior']['lower_bound']],
                            axis=0
                        )
                    )
                else:
                    param_location_list += param_cfg_dict['prior']['location']

                # Add names for prior distributions from config
                for _ in range(n_param_values):
                    prior_dist_names.append(
                        str(param_cfg_dict['prior']['dist_name'])
                    )

                # TODO: add additional implementations
                if param_cfg_dict['prior']['sd_frac']:
                    # Calculates fraction of default value to use as st. dev.
                    param_scale_list.extend(
                        np.asarray(parameter.default_values) *
                        float(param_cfg_dict['prior']['sd_frac'])
                    )

                if param_cfg_dict['prior']['scale']:
                    if not isinstance(param_cfg_dict['prior']['scale'], list):
                        param_scale_list.extend(
                            [param_cfg_dict['prior']['scale']]*n_param_values
                        )
                    else:
                        param_scale_list.extend(
                            param_cfg_dict['prior']['scale']
                        )

                if param_cfg_dict['prior']['upper_bound']:
                    if not isinstance(param_cfg_dict['prior']['upper_bound'], list):
                        param_upper_bound_list.extend(
                            [param_cfg_dict['prior']['upper_bound']]*n_param_values
                        )
                    else:
                        param_upper_bound_list.extend(
                            param_cfg_dict['prior']['upper_bound']
                        )
                else:
                    param_upper_bound_list.extend(
                        [None]*n_param_values
                    )

                if param_cfg_dict['prior']['lower_bound']:
                    if not isinstance(param_cfg_dict['prior']['lower_bound'], list):
                        param_lower_bound_list.extend(
                            [param_cfg_dict['prior']['lower_bound']]*n_param_values
                        )
                    else:
                        param_lower_bound_list.extend(
                            param_cfg_dict['prior']['lower_bound']
                        )
                else:
                    param_lower_bound_list.extend(
                        [None]*n_param_values
                    )

        # Create prior matrix from Ensemble parent class
        prior_df = super().create_initial_priors(
            parameter_names=param_names,
            prior_dist_name=prior_dist_names,
            prior_dist_location=param_location_list,
            prior_dist_scale=param_scale_list,
            upper_bound=param_upper_bound_list,
            lower_bound=param_lower_bound_list
        )

        return prior_df

    def write_to_json(
            self,
            file_name: str | Path,
            save_dir_path: Optional[str | Path] = DEFAULT_JSON_SAVE_DIR,
            mcmc: bool = False,
            latin_hyper_cube: bool = False,
            synth_truth_run: bool = False,
    ) -> None:
        """Writes current parameter dataframe to a json file"""

        if mcmc:
            save_dir_path = Path(save_dir_path) / \
                "mcmc"
        elif latin_hyper_cube:
            save_dir_path = Path(save_dir_path) / \
                "latin_hyper_cube"
        elif synth_truth_run:
            save_dir_path = Path(save_dir_path) / \
                "synthetic_truth_run"
        else:
            save_dir_path = Path(save_dir_path) / \
                f"kalman_iter_{self.current_kalman_iter}"

        if not save_dir_path.is_dir():
            save_dir_path.mkdir(parents=True)

        if not file_name.endswith('.json'):
            file_name = file_name + '.json'

        file_path = save_dir_path / file_name

        # Create json from theta
        param_col_names = self.theta_df.columns
        param_default_names = [
            param.param_name for param in self.default_param_list
        ]

        # Store results in a dictionary
        ensemble_dict = {}

        for ensemble_idx in range(self.n_ensembles):

            # Instantiate dict for storing parameters and associated values
            cur_ens_dict = {}

            for param in param_default_names:

                # Colums matching the current parameter, regular expression
                param_re = r".*" + re.escape(param) + r".*"
                cur_col_names = [
                    x for x in param_col_names if re.match(param_re, x)
                ]

                ensemble_param_vals = self.theta_df.loc[ensemble_idx,
                                                        cur_col_names]

                cur_ens_dict[param] = list(ensemble_param_vals)

            ensemble_dict[str(ensemble_idx + 1)] = cur_ens_dict

        self.ensemble_dict = ensemble_dict

        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(ensemble_dict, json_file, indent=4)

        print(f"Parameter json created in: {file_path}")


def create_dummy_fates_ensemble(
    n_ensembles: int,
) -> FatesParameterEnsemble:
    '''
    Creates a dummy FatesParameterEnsemble for the specified number of
    'n_ensembles', intended to be manipulated for custom parametrizations.
    '''

    # Generate a dummy Ensemble object
    parameter_list = []
    for param_name, param_options in FATES_PARAMS_TO_PERTURBATE.items():
        parameter_list.append(
            FatesDefaultParameter(
                param_name=param_name,
                dimensions=param_options['dimensions'],
                label=param_options['label']
            )
        )

    # Initialize new parameter ensemble
    my_ensemble = FatesParameterEnsemble(n_ensembles=n_ensembles)
    # Add the generated parameters
    my_ensemble.set_default_parameters(
        parameter_list=parameter_list
    )

    # Generate initial dummy priors
    my_ensemble.generate_priors()

    print("\nCreated dummy ensemble!\n")

    return my_ensemble
