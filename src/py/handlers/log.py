"""
The module contains functions to configuration logs
"""
import logging
import os.path
import sys


def set_dir(path='../output/logs') -> str:
    """
    The function checks whether log path exist and otherwise, creates one.

    :param path: Log path.
    :return: Path created.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def set_log(file_name: str, path='../output/logs', level=logging.INFO, format='%(asctime)s\t[%(levelname)s]\t %(message)s'):
    """
    The function designed to aid in log configuration.

    :param file_name: Log name file.
    :param path: Log file path.
    :param level: Log level.
    :param format: Log format.
    :return: None.
    """

    logPath = os.path.join(set_dir(path), file_name)
    logFormatter = logging.Formatter(format)
    rootLogger = logging.getLogger()

    rootLogger.setLevel(level)

    fileHandler = logging.FileHandler(logPath)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
