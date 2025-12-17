import argparse
import asyncio
import sys
import os
from cat_app import setup_logger
from cat_app import CatImageProcessor


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Обработка изображений: локально (один файл) или через TheCatAPI."
    )
    p.add_argument(
        "inputs",
        nargs="?",
        default="1",
        help="Количество изображений (целое число, по умолчанию 1)."
    )
    # новый аргумент: директория для логов
    p.add_argument(
        "--log-dir",
        default=".",
        help="Папка, куда писать логи (по умолчанию текущая директория)."
    )
    # новый аргумент: имя файла логов
    p.add_argument(
        "--log-file",
        default="app.log",
        help="Имя файла логов (по умолчанию app.log)."
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # используем значения из аргументов
    setup_logger(log_dir=None, log_file="app.log")

    try:
        limit_cnt = int(args.inputs)
    except ValueError:
        print("Ошибка: Количество изображений должно быть целым числом.")
        return 1

    proc = CatImageProcessor(output_dir="saved_images")
    asyncio.run(proc.run_async(limit=limit_cnt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
