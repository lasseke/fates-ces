#!/usr/bin/env python3

'''
Script to install the host model. Make sure you have a recent "Git"
module loaded/version installed!
'''

import subprocess
from pathlib import Path
from fatescal.modelhandling.model_config import ModelConfig

model_cfg = ModelConfig()

MODEL_REPO = model_cfg.model_repo
MODEL_ROOT = model_cfg.model_root
MODEL_VERSION = model_cfg.model_version


def setup_model(
    model_root: Path = MODEL_ROOT,
    model_repo: str = MODEL_REPO,
    model_version: str = MODEL_VERSION,
) -> None:
    """
    Clone the model and switch to the correct tag as specified in the settings.
    """

    try:
        proc = subprocess.run(["git", "--version"], capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError("Could not find a working git installation.")
    except FileNotFoundError:
        raise RuntimeError("Could not find a working git installation.")

    if not model_root.exists():
        subprocess.run(
            [
                "git",
                "clone",
                model_repo,
                model_root,
            ]
        )
        subprocess.run(["git", "checkout", model_version], cwd=model_root)
    else:
        print(f"Warning! {model_root} already exists. "
              + "Make sure it is configured correctly.")

    proc = subprocess.run(
        ["git", "describe", "--tags"], cwd=model_root, capture_output=True
    )
    if not (
        (proc.returncode == 0) and
            (proc.stdout.strip().decode("utf8") == model_version)
    ):
        subprocess.run(["git", "fetch", "--all"], cwd=model_root)

        subprocess.run(["git", "restore", "."], cwd=model_root)

        subprocess.run(["git", "checkout", model_version], cwd=model_root)

    # Checkout model externals
    subprocess.run(["manage_externals/checkout_externals"], cwd=model_root)


def main():

    print(
        f'''
        Trying to install model from '{MODEL_REPO}'
        with version tag '{MODEL_VERSION}'
        into: '{MODEL_ROOT}'...
        '''
    )

    setup_model()

    model_cfg.make_files()

    print("\nDone!\n")


if __name__ == "__main__":
    main()
