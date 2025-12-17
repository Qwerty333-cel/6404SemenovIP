"""
main.py — запуск обработки изображений локально или через TheCatAPI.

Режимы:
  * API-режим: получает изображения из TheCatAPI и обрабатывает их,
    используя асинхронный/параллельный CatImageProcessor.
    (ВНИМАНИЕ: Текущая реализация CatImageProcessor поддерживает только метод 'edges').
  * local-режим: читает локальные файлы и обрабатывает их.
    (ВНИМАНИЕ: Локальный режим оставлен синхронным и использует заглушки
    для отсутствующих классов ImageProcessing и методов CatImageAbstract.)

Примеры:
  - API-режим (5 картинок, метод контуров, сохранять в папку ./results)
  python main.py edges API 5 -o ./results

  - API-режим (градации серого, одна картинка, папка по умолчанию)
  python main.py edges API
"""

import argparse
import asyncio
import os
import sys
import numpy as np
import cv2
from dotenv import load_dotenv

from cat_app import CatImageProcessor, setup_logger

try:
    from cat_image_extra.cat_image_interface import CatImageRGB
except ImportError:
    class CatImageRGB:

        def __init__(self, image, url, breed): self.image = image

        def adding_with_correction(self, other, use_cv2=False, correction=False): raise NotImplementedError(
            "Adding not implemented in stub.")

        def subtract_with_correction(self, other, use_cv2=False, correction=False): raise NotImplementedError(
            "Subtracting not implemented in stub.")


class ImageProcessing:

    def __init__(self):
        print("Внимание: Используется заглушка ImageProcessing.")

    def rgb_to_grayscale(self, img, use_cv2=False): return img

    def edge_detection(self, img, use_cv2=False): return img

# --- Простые константы ---
DEFAULT_KERNEL = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]], dtype=np.float32)
DEFAULT_GAMMA = 3.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Обработка изображений: локально (один файл) или через TheCatAPI (несколько файлов)."
    )
    p.add_argument(
        # Оставим все методы для совместимости, но API-режим поддерживает только 'edges'
        "method",
        choices=["edges", "corners", "circles", "conv", "gamma", "gray", "add", "sub"],
        help="Метод обработки.",
    )
    p.add_argument(
        "source",
        choices=["API", "local"],
        help="Источник изображений: API (TheCatAPI) или local (локальный файл).",
    )
    p.add_argument(
        "inputs",
        nargs="*",
        help=("Если source=API — первое значение это количество изображений (целое, по умолчанию 1). "
              "Если source=local — пути к входным файлам: для add/sub нужно минимум 2 файла; "
              "для остальных — 1 файл.")
    )
    p.add_argument(
        "-o", "--output",
        default=None,
        help="Путь/каталог для сохранения. Для API — каталог, для local — файл результата.",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help=f"Значение гамма для метода 'gamma' (по умолчанию {DEFAULT_GAMMA}).",
    )
    p.add_argument(
        "--as-gray",
        action="store_true",
        help="В API-режиме: создавать объекты ч/б (GrayCatImage) вместо цветных. (Проигнорировано в текущей асинхронной реализации)."
    )
    return p


def ensure_env_for_api() -> None:
    """Проверяет наличие CAT_API_KEY. Падает с сообщением, если его нет."""
    load_dotenv()
    api_key = os.getenv("CAT_API_KEY")
    if not api_key:
        print("Ошибка: переменная окружения CAT_API_KEY не задана. "
              "Создайте .env с CAT_API_KEY или выставьте переменную окружения.")
        sys.exit(1)


async def run_api_flow_async(method: str, count_str: str | None, outdir: str | None, as_gray: bool) -> None:
    """Обработка через API: грузим N изображений, обрабатываем и сохраняем результаты асинхронно/параллельно."""
    ensure_env_for_api()

    # Внимание: CatImageProcessor.run_async() выполняет только обнаружение границ ('edges').
    if method != "edges":
        print(
            f"[API] Предупреждение: Текущая реализация CatImageProcessor.run_async() выполняет только 'edges'. Метод '{method}' будет проигнорирован.")
    if as_gray:
        print(
            "[API] Предупреждение: Флаг '--as-gray' игнорируется, так как CatImageProcessor использует CatImageRGB по умолчанию.")

    # 1. Получение количества изображений
    count = 1
    if count_str:
        try:
            count = int(count_str)
        except ValueError:
            print("[API] Ошибка: количество изображений должно быть целым числом.")
            sys.exit(1)
    if count <= 0:
        print("[API] Ошибка: количество изображений должно быть > 0.")
        sys.exit(1)

    # 2. Инициализация процессора
    output_dir = outdir if outdir else "results"
    processor = CatImageProcessor(output_dir=output_dir)

    print(f"[API] Запуск асинхронной обработки {count} изображений (метод: 'edges')...")

    # 3. Вызов асинхронного пайплайна
    try:
        await processor.run_async(limit=count)
        print("-" * 50)
        print(f"[API] Асинхронная/параллельная обработка завершена. Результаты сохранены в: {output_dir}")
        print("Проверьте логи для подтверждения параллельного/асинхронного выполнения и замеров времени.")
        print("-" * 50)
    except Exception as e:
        print(f"[API] КРИТИЧЕСКАЯ ОШИБКА в асинхронном выполнении: {e}")
        sys.exit(1)


def run_local_flow(method: str, input_paths: list[str], out_path: str | None, gamma: float) -> None:
    """
    Обработка локальных файлов. (Оставлена синхронной с заглушками)
    """
    print("-" * 50)
    print(f"[LOCAL] ВНИМАНИЕ: Локальный режим остался СИНХРОННЫМ и использует ЗАГЛУШКИ.")
    print(f"[LOCAL] Метод: {method}, Файлы: {input_paths}")

    if not input_paths:
        print("[LOCAL] Ошибка: для local-режима нужно указать путь(и) к входному файлу(ам).")
        return

    # Заглушка для логики add/sub
    if method in ("add", "sub"):
        if len(input_paths) < 2:
            print("[LOCAL] Ошибка: для методов 'add' и 'sub' укажите минимум два входных файла.")
            return

        try:
            img1 = cv2.imread(input_paths[0])
            img2 = cv2.imread(input_paths[1])
            if img1 is None or img2 is None:
                raise Exception("Не удалось загрузить изображение.")

            cat1 = CatImageRGB(img1, input_paths[0], "file1")

            if method == "add":
                cat1.adding_with_correction(CatImageRGB(img2, input_paths[1], "file2"))
            else:
                cat1.subtract_with_correction(CatImageRGB(img2, input_paths[1], "file2"))

            print(f"[LOCAL] Выполнен вызов метода {method} (через заглушку). Результат не сохранен.")

        except NotImplementedError:
            print(f"[LOCAL] Ошибка: Метод {method} не реализован в CatImageAbstract/CatImageRGB (заглушка).")
        except Exception as e:
            print(f"[LOCAL] Ошибка в add/sub: {e}")

    # Заглушка для логики остальных методов (один файл)
    else:
        img = cv2.imread(input_paths[0])
        if img is None:
            print("[LOCAL] Ошибка при загрузке изображения.")
            return

        proc = ImageProcessing()
        # Вызов заглушки метода
        getattr(proc, {
            "edges": "edge_detection",
            "gray": "rgb_to_grayscale",
            "corners": "corner_detection",
            "circles": "circle_detection",
            "conv": "convolution",
            "gamma": "gamma_correction"
        }.get(method, "edge_detection"))(img.copy())

        print(f"[LOCAL] Выполнен вызов метода '{method}' (через заглушку). Результат не сохранен.")

    print("-" * 50)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.source == "API":
        count_str = args.inputs[0] if args.inputs else None
        # Запуск асинхронной функции через asyncio.run()
        try:
            asyncio.run(run_api_flow_async(
                method=args.method,
                count_str=count_str,
                outdir=args.output,
                as_gray=args.as_gray
            ))
        except SystemExit:
            # Ловим sys.exit(1) из вспомогательных функций
            pass
    else:
        # Синхронный запуск локального режима
        run_local_flow(method=args.method, input_paths=args.inputs, out_path=args.output, gamma=args.gamma)


if __name__ == "__main__":
    main()