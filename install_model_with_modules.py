#!/usr/bin/env python3

import subprocess
import configparser
from pathlib import Path

if __name__ == "main":

    CONFIG_FILE_PATH = Path(__file__).parent / '.model_config'

    config = configparser.ConfigParser()
    config.read(CONFIG_FILE_PATH)

    git_module_name = str(config['hpc_module_names']['GIT'])
    python_module_name = str(config['hpc_module_names']['PYTHON'])

    subprocess.run(['module', '--quiet', 'purge'])
    subprocess.run(['module', 'load', git_module_name])
    subprocess.run(['module', 'load', python_module_name])
    subprocess.run(
        [
            'python3',
            str(Path(__file__).parent / 'install_model.py')
        ]
    )
    subprocess.run(['module', '--quiet', 'purge'])
