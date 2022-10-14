"""
The module contains functions to configuration logs
"""
import logging
import os.path


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
    file_path = os.path.join(set_dir(path), file_name)
    logging.basicConfig(filename=file_path, level=level, format=format)
