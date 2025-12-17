import logging
import sys
import os

def setup_logger(
    log_dir: str | None = None,
    log_file: str = "app.log",
    for_child_process: bool = False,
) -> logging.Logger:
    # если путь не задан — берём директорию этого файла (cat_app/logs)
    if log_dir is None:
        log_dir = os.path.dirname(__file__)

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - PID:%(process)d - %(levelname)s - "
        "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if not for_child_process:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger
