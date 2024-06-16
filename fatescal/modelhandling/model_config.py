"""Shared code for model configuration"""

import configparser
from pathlib import Path
from fatescal.helpers import run_subprocess

CONFIG_FILE_NAME = 'model_machine_config.txt'
CONFIG_FILE_PATH = Path(__file__).parents[2] / CONFIG_FILE_NAME

if not CONFIG_FILE_PATH.is_file():
    raise FileNotFoundError(
        f"No '{CONFIG_FILE_NAME}' file found in {CONFIG_FILE_PATH.parent}!"
    )


class ModelConfig:

    def __init__(self) -> None:

        self.read_config()

    def read_config(self) -> None:

        config = configparser.ConfigParser()
        config.read(CONFIG_FILE_PATH)

        self.model_repo = str(config['install_options']['MODEL_URL'])
        self.model_root = Path(
            str(config['install_options']['MODEL_INSTALL_PATH'])
        ).expanduser()
        self.model_version = str(config['install_options']['MODEL_VERSION'])
        self.machine_name = str(config['install_options']['MACHINE_NAME'])

        # FATES settings
        self.default_fates_param_cdl_file_path = \
            str(config['fates_options']['DEFAULT_PARAM_CDL_FILE'])
        self.default_fates_param_nc_file_path = \
            str(config['fates_options']['DEFAULT_PARAM_NC_FILE'])

        self.modify_fates_params_tool_path = \
            str(config['fates_options']['MODIFY_PARAMS_TOOL'])

        # NetCDF module, needed to create FATES parameter files
        self.netcdf_module_name = \
            str(config['hpc_module_names']['NETCDF'])
        # NCO module, needed to concatenate .nc files
        self.nco_module_name = \
            str(config['hpc_module_names']['NCO'])

        # Git module
        self.git_module_name = \
            str(config['hpc_module_names']['GIT'])
        # Anaconda module
        self.anaconda_module_name = \
            str(config['hpc_module_names']['ANACONDA'])

    def make_files(self) -> None:
        """Create required files (only parameter nc from cdl for now)"""

        if not Path(self.default_fates_param_nc_file_path).is_file():
            self.make_nc_from_cdl(
                cdl_file_path=self.model_root / self.default_fates_param_cdl_file_path,
                nc_out_path=self.model_root / self.default_fates_param_nc_file_path
            )

    def make_nc_from_cdl(
        self,
        cdl_file_path: str | Path,
        nc_out_path: str | Path
    ) -> None:
        """
        Create parameter nc from cdl file. Requires NetCDF (defined in model machine config file)!
        """

        cdl_file_path = Path(cdl_file_path)
        nc_out_path = Path(nc_out_path)

        if nc_out_path.is_file():
            print(f"'{nc_out_path}' already exists!")
            return

        if (str(cdl_file_path).endswith('.cdl')) and (cdl_file_path.is_file()):

            if not str(nc_out_path).endswith('.nc'):
                nc_out_path = Path(str(nc_out_path)+'.nc')

            # Load NetCDF module (defined in .model_config)
            if self.netcdf_module_name:
                cmd = f'module purge --quiet;module load {self.netcdf_module_name};' \
                    + f'ncgen {cdl_file_path} -o {nc_out_path};' \
                    + 'module purge --quiet;'
            else:
                cmd = f'ncgen {cdl_file_path} -o {nc_out_path};'

            _ = run_subprocess(
                [cmd],
                shell=True
            )

            print(f"'{nc_out_path}' succesfully created!")

        else:
            raise ValueError(
                f"""
                '{cdl_file_path}' must be a path to an existing .cdl file!
                """
            )
