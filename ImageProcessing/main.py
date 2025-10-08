"""
main.py — запуск обработки изображений локально или через TheCatAPI.

Режимы:
  * API-режим: получает изображения из TheCatAPI и обрабатывает их классами
    CatImageProcessor / CatImage (поддерживаются: gray, edges, corners, circles, conv, gamma, add, sub).
  * local-режим: читает локальные файлы и обрабатывает их классом ImageProcessing
    (поддерживаются: gray, edges, corners, circles, conv, gamma),
    а для add/sub — складывает/вычитает два локальных изображения через CatImage.


Примеры:
  - API-режим (5 картинок, метод контуров, сохранять в папку ./results)
  python main.py edges API 5 -o ./results

  - API-режим (градации серого, одна картинка, папка по умолчанию)
  python main.py gray API

  - local-режим (один файл)
  python main.py corners local ./cat.jpg -o corners_result.png

  - local-режим (gray для одного файла)
  python main.py gray local ./cat.jpg -o gray_cat.png

  - local-режим (add/sub для двух файлов)
  python main.py add local ./a.jpg ./b.jpg -o sum.png
  python main.py sub local ./a.jpg ./b.jpg

  - local-режим (add/sub для двух файлов)
  открой файл test.ipynb, там есть пример сложения двух картинок через CatImage

"""

import argparse
import os
import sys
import numpy as np
import cv2
from dotenv import load_dotenv

# Локальные модули
from implementation.image_processing import ImageProcessing
import api_testing  # содержит CatImageProcessor из твоего модуля

# --- Простые константы ---
DEFAULT_KERNEL = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]], dtype=np.float32)
DEFAULT_GAMMA = 3.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Обработка изображений: локально (один файл) или через TheCatAPI (несколько файлов)."
    )
    p.add_argument(
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
    return p


def ensure_env_for_api() -> tuple[str, str]:
    """Загружает .env и вытаскивает API_KEY/BASE_URL. Падает с сообщением, если чего-то нет."""
    load_dotenv()
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    if not api_key or not base_url:
        print("Ошибка: переменные окружения API_KEY/BASE_URL не заданы. "
              "Создай .env или выставь переменные окружения.")
        sys.exit(1)
    return api_key, base_url


def run_api_flow(method: str, count_str: str | None, outdir: str | None, gamma: float) -> None:
    """Обработка через API: грузим N изображений и сохраняем результаты пачкой."""
    api_key, base_url = ensure_env_for_api()

    # Количество изображений
    if count_str is None:
        count = 1
    else:
        try:
            count = int(count_str)
        except ValueError:
            print("Ошибка: количество изображений должно быть целым числом.")
            sys.exit(1)
    if count <= 0:
        print("Ошибка: количество изображений должно быть > 0.")
        sys.exit(1)

    # Инициализация процессора
    processor = api_testing.CatImageProcessor(api_key=api_key, url=base_url)

    try:
        print(f"[API] Запрос {count} изображений из TheCatAPI...")
        cats = processor.fetch_cats(limit=count)  # has_breeds=1, mime_types='jpg,png' внутри по умолчанию
        if cats.size == 0:
            print("[API] Не удалось получить изображения.")
            return

        print(f"[API] Обработка методом '{method}'...")
        processor.process_cat_image(
            cats=cats,
            method=method,
            path=outdir,
            gamma=gamma,
            kernel=DEFAULT_KERNEL
        )
        print("[API] Готово.")

    except Exception as e:
        print(f"[API] Ошибка: {e}")


def run_local_flow(method: str, input_paths: list[str], out_path: str | None, gamma: float) -> None:
    """Обработка локальных файлов.
    Для add/sub требуется минимум 2 файла, для остальных — 1 файл.
    """
    if not input_paths:
        print("Ошибка: для local-режима нужно указать путь(и) к входному файлу(ам).")
        return

    need_two = method in ("add", "sub")
    if need_two and len(input_paths) < 2:
        print("Ошибка: для методов 'add' и 'sub' укажите минимум два входных файла.")
        return

    # Первый файл
    first_path = input_paths[0]
    if not os.path.exists(first_path):
        print(f"Ошибка: файл не найден: {first_path}")
        return
    img1 = cv2.imread(first_path)
    if img1 is None:
        print(f"Ошибка: не удалось загрузить изображение: {first_path}")
        return

    # --- add/sub через CatImage с коррекцией размеров ---
    if method in ("add", "sub"):
        second_path = input_paths[1]
        if not os.path.exists(second_path):
            print(f"Ошибка: файл не найден: {second_path}")
            return
        img2 = cv2.imread(second_path)
        if img2 is None:
            print(f"Ошибка: не удалось загрузить изображение: {second_path}")
            return

        def stem(p: str) -> str:
            return os.path.splitext(os.path.basename(p))[0]

        cat1 = api_testing.CatImage(
            id="local1", url=first_path, breed=stem(first_path),
            width=img1.shape[1], height=img1.shape[0], image=img1
        )
        cat2 = api_testing.CatImage(
            id="local2", url=second_path, breed=stem(second_path),
            width=img2.shape[1], height=img2.shape[0], image=img2
        )

        if method == "add":
            result = cat1.adding_with_correction(cat2, variant=True, correction=True)
            suffix = "add"
        else:
            result = cat1.subtract_with_correction(cat2, variant=True, correction=True)
            suffix = "sub"

        save_path = out_path if out_path else f"{os.path.splitext(first_path)[0]}_{suffix}_result{os.path.splitext(first_path)[1] or '.png'}"
        ok = cv2.imwrite(save_path, result)
        print(f"[LOCAL] Результат сохранён в: {save_path}" if ok else "[LOCAL] Не удалось сохранить результат.")
        return

    # --- остальные методы (один файл) через ImageProcessing ---
    if len(input_paths) > 1:
        print("[LOCAL] Предупреждение: передано несколько файлов, будет обработан только первый.")

    proc = ImageProcessing()
    result = None

    print(f"[LOCAL] Применение '{method}' к '{first_path}'...")
    if method == "edges":
        result = proc.edge_detection(img1)
    elif method == "corners":
        result = proc.corner_detection(img1)
    elif method == "circles":
        result = proc.circle_detection(img1)
    elif method == "conv":
        result = proc.convolution(img1, DEFAULT_KERNEL)
    elif method == "gamma":
        result = proc.gamma_correction(img1, gamma)
    elif method == "gray":
        result = proc.rgb_to_grayscale(img1)
    else:
        print(f"[LOCAL] Метод '{method}' не поддержан в локальном режиме.")
        return

    if result is None:
        print(f"[LOCAL] Метод '{method}' не вернул результат.")
        return

    save_path = out_path if out_path else f"{os.path.splitext(first_path)[0]}_{method}_result{os.path.splitext(first_path)[1] or '.png'}"
    ok = cv2.imwrite(save_path, result)
    print(f"[LOCAL] Результат сохранён в: {save_path}" if ok else "[LOCAL] Не удалось сохранить результат.")



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.source == "API":
        count_str = args.inputs[0] if args.inputs else None
        run_api_flow(method=args.method, count_str=count_str, outdir=args.output, gamma=args.gamma)
    else:
        run_local_flow(method=args.method, input_paths=args.inputs, out_path=args.output, gamma=args.gamma)





if __name__ == "__main__":
    main()
