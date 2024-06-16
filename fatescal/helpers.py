#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General helper functionalities.
"""

import subprocess
import os
import logging
import sys
import io
import time
from pathlib import Path
from datetime import datetime as dt
from typing import List, Union

ENVIRONMENT = {**os.environ}
LOG_SAVE_PATH = Path(__file__).parents[1]
LOG_LEVEL_FILE = logging.DEBUG
LOG_LEVEL_CONSOLE = logging.DEBUG


class LoggerWriter(io.TextIOBase):
    '''Custom class to redirect output to logger'''

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.rstrip() != '':
            self.logger.log(self.level, message.rstrip())


def setup_logging(logger_name: str) -> logging.Logger:
    '''
    Logger that simultaneously writes to a log file
    and the standard console output.
    '''

    # Create the logger object
    logger = logging.getLogger(logger_name)
    logger.setLevel(LOG_LEVEL_CONSOLE)

    timestamp = dt.now().strftime('%Y%m%d-%H%M%S')

    # Create a file handler that writes to a file
    file_handler = logging.FileHandler(
        LOG_SAVE_PATH / f'{logger_name}_{timestamp}.log'
    )
    file_handler.setLevel(LOG_LEVEL_FILE)

    # Create a stream handler that prints to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL_CONSOLE)

    # Create a formatter for the log messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the formatter for both handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Redirect stdout and stderr to the logger
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)

    return logger


def run_subprocess(cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
    '''
    Executes a list of commands with the Python subprocess library.
    'kwargs' are additional keyword arguments for 'subprocess.run()'.
    '''

    if isinstance(cmd, str):
        cmd = [cmd]

    print(f"\nRunning command: {' '.join([str(x) for x in cmd])}\n")

    process = subprocess.run(
        [str(x) for x in cmd],
        capture_output=True,
        env=ENVIRONMENT,
        **kwargs
    )

    if process.returncode != 0:
        print(process.stdout.decode("utf-8").strip())
        raise Exception(process.stderr.decode("utf-8").strip())
    else:
        print(process.stdout.decode("utf-8").strip())

    print("\nFinished running subprocess.\n")

    return process.stdout.decode("utf-8").strip()


def wait_for_model(wait_minutes: Union[float, int] = 5) -> None:
    '''
    Sigma2 specific function to check if CLM-FATES is still running,
    in a silly way. Assumes that no other jobs are running in your
    queue.
    '''

    model_running = True

    while model_running:

        process_response = run_subprocess(
            cmd=["squeue --me"],
            shell=True,
        )

        # Hacky way to check if job is still in the queue
        # (REASON) is the last part of the string when job list is empty
        if not process_response.endswith("(REASON)"):
            print(
                f"Model still running, trying again in {wait_minutes} minutes...", end=""
            )
            time.sleep(wait_minutes*60)
        else:
            print("Model finished!")
            return
