from typing import Final
from typing import List

import logging

SEPARATOR: Final = '\n------------------------------------\n'


def setup_logger(name, file_path, level=logging.INFO):
    handler = logging.FileHandler(file_path)
    handler.setFormatter(logging.Formatter('%(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())

    return logger


def close_loggers(loggers: List[logging.Logger]):
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def log_params(logger, params, title="Parameters"):
    logger.info(f"{title}:")
    for key, value in params.items():
        logger.info(f"{key}: {value}")


# Funzione per loggare le metriche di CodeCarbon
def log_codecarbon_metrics(logger, emissions):
    logger.info("CodeCarbon Metrics:")
    logger.info(f"Energy consumed (kWh): {emissions['energy_consumed']:.4f}")
    logger.info(f"CO2 emissions (kg): {emissions['emissions']:.4f}")
    logger.info(f"Energy cost (USD): {emissions['energy_cost']:.4f}")


# Funzione per scrivere il separatore nel log
def write_separator(logger):
    separator = '\n------------------------------------\n'
    logger.info(separator)
