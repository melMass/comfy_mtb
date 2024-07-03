import logging
import os
import re

base_log_level = logging.DEBUG if os.environ.get("MTB_DEBUG") else logging.INFO


# Custom object that discards the output
class NullWriter:
    def write(self, text):
        pass


class Formatter(logging.Formatter):
    grey = "\x1b[38;20m"
    cyan = "\x1b[36;20m"
    purple = "\x1b[35;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # format = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format = "[%(name)s] | %(levelname)s -> %(message)s"

    FORMATS = {
        logging.DEBUG: purple + format + reset,
        logging.INFO: cyan + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def mklog(name: str, level: int = base_log_level):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    for handler in logger.handlers:
        logger.removeHandler(handler)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(Formatter())
    logger.addHandler(ch)

    # Disable log propagation
    logger.propagate = False

    return logger


# - The main app logger
log = mklog(__package__, base_log_level)


def log_user(arg: str):
    print(f"\033[34mComfy MTB Utils:\033[0m {arg}")


def get_summary(docstring: str):
    return docstring.strip().split("\n\n", 1)[0]


def blue_text(text: str):
    return f"\033[94m{text}\033[0m"


def cyan_text(text: str):
    return f"\033[96m{text}\033[0m"


def get_label(label: str):
    if label.startswith("MTB_"):
        label = label[4:]

    words = re.findall(
        r"(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z]))",
        label,
    )
    reformatted_label = re.sub(r"([A-Z]+)", r" \1", label).strip()
    words = reformatted_label.split()
    return " ".join(words).strip()
