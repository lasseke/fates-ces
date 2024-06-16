"""Helper to create CTSM cases with desired settings"""

import shutil
import urllib.request
import zipfile
import time
import pickle
import json
from pathlib import Path
from typing import Optional
from fatescal.helpers import run_subprocess
from fatescal.modelhandling.model_config import ModelConfig
from fatescal.config import DEFAULT_JSON_SAVE_DIR

model_cfg = ModelConfig()

DEFAULT_FATES_PARAM_FILE_PATH = Path(
    model_cfg.default_fates_param_nc_file_path
)
MODIFY_FATES_PARAMS_TOOL_PATH = Path(
    model_cfg.modify_fates_params_tool_path
)

CASES_ROOT_PATH = Path(__file__).parents[2] / 'cases'
OUTPUT_ROOT_PATH = Path(__file__).parents[2] / 'data' \
    / 'results' / 'model_output'
DEF_MODEL_DRIVER_STR = 'nuopc'
DEF_CASE_RES_STR = 'CLM_USRDAT'
DEF_COMPSET_STR = r'2000_DATM%GSWP3v1_CLM51%FATES_SICE_SOCN_MOSART_SGLC_SWAV'


class Case:
    """CTSM case objects for convenience"""

    def __init__(
            self,
            name: str,
            data_root_path: str | Path,
            compset: str = DEF_COMPSET_STR,
            model_driver: str = DEF_MODEL_DRIVER_STR,
            case_res: str = DEF_CASE_RES_STR,
            cases_root_path: str | Path = CASES_ROOT_PATH,
            output_root_path: str | Path = OUTPUT_ROOT_PATH,
            data_url: Optional[str] = None,
            project_name: Optional[str] = None,
            is_multi_instance: bool = True,
    ):

        self.name = name
        self.data_root_path = Path(data_root_path)
        self.output_root_path = Path(output_root_path)
        if not output_root_path.is_dir():
            output_root_path.mkdir(parents=True, exist_ok=False)
        self.compset = compset
        self.data_url = data_url
        self.model_driver = model_driver
        self.case_res = case_res
        self.project_name = project_name

        self.case_path = cases_root_path / self.name

        self.is_multi_instance = is_multi_instance

        # Stays at one until multi case is created
        self.n_model_instances = 1
        self.case_is_set_up = False
        self.fates_param_ensemble_json = None
        self.fates_param_files_list = []

    def download_input_data(self) -> None:
        """Download single-point model input data if necessary"""

        if not self.data_root_path.is_dir():

            if self.data_url is None:
                raise RuntimeError("No 'data_url' argument provided for "
                                   + "this case instance! Create a new Case "
                                   + "object and include it.")

            print(f'Downloading input data zip into {self.data_root_path}...')

            self.data_root_path.mkdir(exist_ok=True)

            tmp_zip_file_path = self.data_root_path / 'inputdata.zip'

            # Download file from `url` and save it locally under `file_name`:
            with urllib.request.urlopen(self.data_url) as response, \
                    open(tmp_zip_file_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

            print('Extracting input data...')

            # Unzip folder
            with zipfile.ZipFile(tmp_zip_file_path, 'r') as zip_file:
                zip_file.extractall(self.data_root_path)

            print('Deleting zip file...')
            tmp_zip_file_path.unlink()

            print('Finished downloading input data!')

        else:
            print(
                f'{self.data_root_path} already in place, skipping download.\n'
            )

    def create_case(self) -> None:
        """Create a new case with chosen settings"""

        print(
            f"Creating case {self.name} with the following attributes:\n"
            f"Path: {self.case_path}\n"
            f"Compset: {self.compset}\n"
        )

        shutil.rmtree(self.case_path, ignore_errors=True)

        if self.data_url is not None:
            self.download_input_data()

        create_new_case_cmd = [
            str(model_cfg.model_root / "cime" /
                "scripts" / "create_newcase"),
            "--case",
            str(self.case_path),
            "--compset",
            self.compset,
            "--driver",
            self.model_driver,
            "--res",
            self.case_res,
            "--machine",
            model_cfg.machine_name,
            "--run-unsupported",
            "--handle-preexisting-dirs",
            "r",
        ]

        if (self.data_root_path / "user_mods").exists():
            create_new_case_cmd.extend(
                [
                    "--user-mods-dirs",
                    str(self.data_root_path / "user_mods"),
                ]
            )

        # Add project string if given
        if self.project_name is not None:
            create_new_case_cmd.extend(
                [
                    "--project",
                    self.project_name,
                ]
            )

        print('Starting to create the case...')

        start = time.time()

        _ = run_subprocess(
            create_new_case_cmd
        )

        print(f'Finished creating case in {time.time() - start} seconds.')

    def xml_change(self, xml_changes_dict: dict) -> None:
        """
        Call CESM './xmlchange' tool with function's **kwargs arguments where:
        parameter='value'
        """

        xml_change_cmd = "./xmlchange "

        xml_change_cmd += \
            ",".join([f"{param}={val}" for param,
                     val in xml_changes_dict.items()])

        _ = run_subprocess(
            xml_change_cmd,
            cwd=self.case_path,
            shell=True
        )

        print('Finished custom XML changes.')

    def case_as_multi_driver(
            self,
            n_model_instances: int
    ) -> None:
        """
        Enable multi driver model, see
        http://esmci.github.io/cime/versions/master/html/users_guide/multi-instance.html
        """

        if (n_model_instances := int(n_model_instances)) <= 1:
            raise ValueError(
                "Multi-case must have more than one instance!"
            )

        case_setup_run_cmd = [
            "./xmlchange",
            f"MULTI_DRIVER=TRUE,NINST={n_model_instances}"
        ]

        _ = run_subprocess(
            case_setup_run_cmd,
            cwd=self.case_path
        )

        self.n_model_instances = n_model_instances

        # Setup the case
        self.case_setup()

    def add_to_namelists(
            self,
            nl_changes_dict: dict,
            namelist: str = 'user_nl_clm'
    ) -> None:
        """
        Adds namelist arguments to multiple namelists created by multi-driver
        mode where: parameter=value

        Fortran booleans (e.g., .true.) must be passed as Python booleans,
        and numbers as numbers as the code adds quotation marks around strings.

        Must be called AFTER case setup.
        """

        if not self.case_is_set_up:
            raise RuntimeError(
                "'./case.setup' was not called yet!"
            )

        nl_param_values_str = ''

        for param, val in nl_changes_dict.items():

            if isinstance(val, str):
                val = f"'{val}'"

            if isinstance(val, bool):
                if val:
                    val = '.true.'
                else:
                    val = '.false.'

            if isinstance(val, list):
                val = ','.join(
                    [f"'{x}'" if isinstance(x, str) else f"{x}" for x in val]
                )

            nl_param_values_str += f"{param}={val}\n"

        # Add to multi-driver namelists
        for idx in range(self.n_model_instances):

            file_idx = f'{idx+1:04d}'

            if self.is_multi_instance:
                with open(self.case_path / f"{namelist}_{file_idx}", "a") as file:
                    file.write(
                        nl_param_values_str
                    )
            else:
                with open(self.case_path / namelist, "a") as file:
                    file.write(
                        nl_param_values_str
                    )

    def _create_fates_param_file(
            self,
            file_out_path: str | Path,
            param_config: dict,
            file_in_path: str | Path = DEFAULT_FATES_PARAM_FILE_PATH,
    ):
        """
        Creates a FATES parameter file for each ensemble defined in
        'self.param_dict'.
        """

        # Generate .nc from .cdl param file if necessary!
        file_in_path = Path(file_in_path)

        if str(file_in_path).endswith('.cdl'):

            nc_param_file_path = file_in_path.parent / \
                (file_in_path.stem + '.nc')

            if not nc_param_file_path.is_file():
                raise ValueError(
                    "FATES parameter file (.nc) not found! Create it!"
                )
        elif str(file_in_path).endswith('.nc'):
            nc_param_file_path = file_in_path
        else:
            raise ValueError("Must provide a .nc or .cdl file!")

        for idx, (parameter, values) in enumerate(param_config.items()):

            if idx == 0:
                _ = run_subprocess(
                    [
                        'python',
                        f'{MODIFY_FATES_PARAMS_TOOL_PATH}',
                        '--var', parameter,
                        '--val',
                        f"{','.join([str(x) for x in values])}",
                        '--allPFTs',
                        '--fin', f'{nc_param_file_path}',
                        '--fout', f'{file_out_path}'
                    ]
                )

            else:
                _ = run_subprocess(
                    [
                        'python',
                        f'{MODIFY_FATES_PARAMS_TOOL_PATH}',
                        '--var', parameter,
                        '--val',
                        f"{','.join([str(x) for x in values])}",
                        '--allPFTs',
                        '--fin', f'{file_out_path}',
                        '--fout', f'{file_out_path}', '--overwrite'
                    ]
                )

    def create_fates_param_files(
            self,
            fates_param_json_fname: str,
            kalman_iter: int = 0,
            mcmc: bool = False,
            synthetic_truth: bool = False,
            json_dir_path: Path | str = DEFAULT_JSON_SAVE_DIR,
    ):
        """
        Create FATES parameter files from given 'FatesParameterEnsemble'
        '.json' files and link them to the Case's model instances.
        """

        if mcmc:
            param_json_path = Path(json_dir_path) / \
                "mcmc" / fates_param_json_fname
        elif synthetic_truth:
            param_json_path = Path(json_dir_path) / \
                "synthetic_truth_run" / fates_param_json_fname
        else:
            param_json_path = Path(json_dir_path) / \
                f"kalman_iter_{kalman_iter}" / fates_param_json_fname

        if (not param_json_path.is_file()) or \
                (not str(param_json_path).endswith(".json")):
            raise ValueError(
                f"'{param_json_path}' is not a valid .json file!"
            )

        with open(param_json_path, encoding="utf-8") as json_file:
            ensemble_dict: dict = json.load(json_file)

        self.fates_param_ensemble_json = ensemble_dict

        if len(ensemble_dict) != self.n_model_instances:
            raise ValueError(
                f'''
                Parameter json contains {len(ensemble_dict)} ensembles,
                but this multi-case has {self.n_model_instances}
                instances. Must be identical!
                '''
            )

        for idx, param_config in enumerate(ensemble_dict.values()):

            file_idx = f'{idx+1:04d}'
            cur_param_file = self.case_path / f'fates_params_{file_idx}.nc'

            self._create_fates_param_file(
                file_out_path=cur_param_file,
                param_config=param_config
            )

            self.fates_param_files_list.append(cur_param_file)

            # Add to CLM namelist
            if self.is_multi_instance:
                with open(self.case_path / f"user_nl_clm_{file_idx}", "a") as file:
                    file.write(
                        f"fates_paramfile='{cur_param_file}'\n"
                    )
            else:
                with open(self.case_path / f"user_nl_clm", "a") as file:
                    file.write(
                        f"fates_paramfile='{cur_param_file}'\n"
                    )

    def case_setup(self) -> None:
        """Invoke ./case.setup"""

        _ = run_subprocess(
            ["./case.setup"],
            cwd=self.case_path
        )

        self.case_is_set_up = True

    def case_build(self) -> None:
        """Invoke ./case.build"""

        if not self.case_is_set_up:
            raise RuntimeError("Invoke case setup first!")

        _ = run_subprocess(
            ["./case.build"],
            cwd=self.case_path
        )

        self.case_is_built = True

    def case_submit(self) -> None:
        """Invoke ./case.submit"""

        if not self.case_is_built:
            raise RuntimeError("Build case first!")

        _ = run_subprocess(
            ["./case.submit"],
            cwd=self.case_path
        )

    def save_as_pkl(self, save_path: str | Path) -> None:
        '''Saves a case object as a pickle for persistence'''

        with open(save_path, 'wb', encoding='utf-8') as out_pkl:
            pickle.dump(self, out_pkl, pickle.HIGHEST_PROTOCOL)
