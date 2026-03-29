"""
main.py

Пример лабораторной работы по курсу "Технологии программирования на Python".

Модуль предназначен для демонстрации работы с обработкой изображений с помощью библиотеки OpenCV.
Реализован консольный интерфейс для применения различных методов обработки к изображению:
- обнаружение границ (edges)
- обнаружение углов (corners)
- обнаружение окружностей (circles)

Запуск:
    python main.py <метод> <путь_к_изображению> [-o путь_для_сохранения]
    python main.py <путь_к_изображению> --process-all

Аргументы:
    метод: edges | corners | circles | conv | gray | gamma
    путь_к_изображению: путь к входному изображению
    -o, --output: путь для сохранения результата (по умолчанию: <имя_входного_файла>_result.png)
    -p, --process-all: применить все доступные методы к одному изображению.

Пример:
    python main.py edges input.jpg
    python main.py corners input.jpg -o corners_result.png
    python main.py input.jpg --process-all

Автор: [Ваше имя]
"""

import argparse
import os
import numpy as np
import cv2

# Предполагается, что класс ImageProcessing находится в файле implementation.py
from implementation import ImageProcessing

def printer(input_path: str) -> None:
    """
    Применяет все доступные методы обработки к одному изображению и сохраняет результаты.

    Для каждого метода создается два файла:
    1. Обычный результат (например, 'image_conv.jpg').
    2. Результат с вариантом 'old' (например, 'image_conv_old.jpg').
    """
    print(f"Запуск пакетной обработки для изображения: {input_path}")
    image = cv2.imread(input_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {input_path}")
        return

    processor = ImageProcessing()
    base, ext = os.path.splitext(input_path)

    # Словарь, где ключ - короткое имя для файла, а значение - кортеж из:
    # (имя_метода_в_классе, словарь_дополнительных_аргументов)
    methods_to_process = {
        "conv": ("convolution", {"kernel": np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])}),
        "gray": ("rgb_to_grayscale", {}),
        "gamma": ("gamma_correction", {"gamma": 2.2}),
        "edges": ("edge_detection", {}),
        "corners": ("corner_detection", {}),
    }

    for name, (method_name, kwargs) in methods_to_process.items():
        print(f"--- Применение метода: {method_name} ---")
        # Цикл по вариантам: обычный и 'old'
        for variant in [None, "old"]:
            try:
                # Копируем аргументы, чтобы не изменять их в словаре
                current_kwargs = kwargs.copy()
                
                # Получаем метод из экземпляра класса по его имени
                process_function = getattr(processor, method_name)
                
                # Устанавливаем выходной путь
                if variant:
                    output_path = f"{base}_{name}_{variant}.jpg"
                    current_kwargs['variant'] = variant
                else:
                    output_path = f"{base}_{name}.jpg"

                # Вызываем метод с нужными аргументами
                result = process_function(image, **current_kwargs)

                # Сохраняем результат
                cv2.imwrite(output_path, result)
                print(f"Результат сохранён в {output_path}")

            except TypeError as e:
                # Эта ошибка возникнет, если метод не принимает аргумент 'variant'
                if variant == "old":
                    print(f"Предупреждение: Метод '{method_name}' не поддерживает 'variant=old'. Пропускаем.")
                else:
                    print(f"Ошибка при вызове метода '{method_name}': {e}")
            except Exception as e:
                print(f"Неожиданная ошибка при обработке методом '{method_name}': {e}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обработка изображения с помощью методов ImageProcessing (OpenCV).",
    )
    # Аргумент method теперь необязательный
    parser.add_argument(
        "method",
        nargs='?', # Делает аргумент необязательным
        default=None,
        choices=[
            "edges",
            "corners",
            "circles",
            "conv",
            "gray",
            "gamma"
        ],
        help="Метод обработки: edges, corners, circles, conv, gray, gamma",
    )
    parser.add_argument(
        "input",
        help="Путь к входному изображению",
    )
    parser.add_argument(
        "-o", "--output",
        help="Путь для сохранения результата (по умолчанию: <input>_result.png)",
    )
    # Заменяем -p на флаг --process-all
    parser.add_argument(
        "-p", "--process-all",
        action="store_true", # Теперь это флаг, он не требует значения
        help="Если указано, применить все методы обработки к изображению."
    )

    args = parser.parse_args()

    # Если указан флаг --process-all, вызываем printer и выходим
    if args.process_all:
        printer(args.input)
        return

    # Если флаг не указан, но и метод не выбран, выводим ошибку
    if not args.method:
        print("Ошибка: необходимо указать метод обработки или использовать флаг --process-all.")
        parser.print_help()
        return

    # --- Старая логика для обработки одного метода ---
    image =cv2.imread(args.input)  #вычесть из 255 для инвертирования
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {args.input}")
        return

    processor = ImageProcessing()

    # Выбор метода
    if args.method == "edges":
        result = processor.edge_detection(image)
    elif args.method == "corners":
        result = processor.corner_detection(image)
    elif args.method == "circles":
        # Ваше примечание: circle_detection пока не работает
        result = processor.circle_detection(image)
    elif args.method == "conv":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) # Ядро для повышения резкости
        result = processor.convolution(image, kernel)
    elif args.method == "gray":
        result = processor.rgb_to_grayscale(image, variant="old")
    elif args.method == "gamma":
        gamma_value = 2.2
        result = processor.gamma_correction(image, gamma_value)
    else:
        # Эта проверка избыточна из-за 'choices' в парсере, но не помешает
        print("Ошибка: неизвестный метод")
        return

    # Определение пути для сохранения
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_{args.method}_result.png"

    # Сохранение результата
    cv2.imwrite(output_path, result)
    print(f"Результат сохранён в {output_path}")


if __name__ == "__main__":
    main()