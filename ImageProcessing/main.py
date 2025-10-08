"""
main.py

Пример лабораторной работы по курсу "Технологии программирования на Python".

Модуль предназначен для демонстрации работы с обработкой изображений.
- Для методов edges, corners, circles, conv, gamma используется локальная обработка
  с помощью класса ImageProcessing.
- Для метода gray используется загрузка изображения из TheCatAPI с помощью
  классов CatImageProcessor и CatImage.

Запуск:
    # Для локальных файлов
    python main.py <метод> <путь_к_изображению> [-o путь_для_сохранения]
    
    # Для методов, работающих с API (входной файл не нужен)
    python main.py gray [-o путь_для_сохранения]

Аргументы:
    метод: edges | corners | circles | conv | gamma | gray
    путь_к_изображению: путь к входному изображению (обязателен для всех методов, кроме 'gray')
    -o, --output: путь для сохранения результата

Пример:
    python main.py edges cat.jpg
    python main.py corners dog.png -o corners_result.png
    python main.py gray
"""

import argparse
import os
import numpy as np
import cv2

# Импортируем оба необходимых компонента
from implementation.image_processing import ImageProcessing
import api_testing

from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обработка изображения: локально или через TheCatAPI.",
    )
    parser.add_argument(
        "method",
        choices=[
            "edges", "corners", "circles", "conv", "gamma", "gray"
        ],
        help="Метод обработки.",
    )
    parser.add_argument(
        "numberOfCats",
        nargs='?', # Аргумент необязательный
        default=1,
        help="Количество изображений котов для получения из API.",
    )
    parser.add_argument(
        "source",
        nargs='?', # Аргумент необязательный
        choices = ["API", "local"],
        default="API",
        help="Работа с изображениями загруженными через API или локально. ",
    )
    parser.add_argument(
        "input",
        nargs='?', # Аргумент необязательный
        default=None,
        help="Путь к входному изображению. ",
    )
    parser.add_argument(
        "-o", "--output",
        help="Путь для сохранения результата.",
    )

    args = parser.parse_args()

    # Работа с API  
    if args.source == "API":
        if args.input:
            print("Предупреждение: для работы с API входной файл не используется, он будет проигнорирован.")
        try:

            print("Запрос изображения с сервера TheCatAPI...")
            # Используем процессор для работы с API
            api_processor = api_testing.CatImageProcessor(api_key=API_KEY)
            
            cats = api_processor.fetch_cats(limit=int(args.numberOfCats))
            if not cats.size:
                print("Ошибка: не удалось получить изображение из API.")
                return
            print(cats.size)
            api_processor.process_cat_image(cats=cats, path=args.output, method=args.method)

        except Exception as e:
            print(f"Произошла ошибка при работе с API: {e}")
        return # Завершаем выполнение

    # Работа с локальным файлом
    else:
        # Проверяем, был ли предоставлен входной файл
        if not args.input:
            print(f"Ошибка: для метода '{args.method}' необходимо указать путь к входному файлу.")
            parser.print_help()
            return

        image = cv2.imread(args.input)
        if image is None:
            print(f"Ошибка: не удалось загрузить изображение по пути: {args.input}")
            return

        # Используем класс для локальной обработки изображений
        local_processor = ImageProcessing()
        result = None
        print(f"Применение локального метода '{args.method}' к файлу '{args.input}'...")

        # Выбор метода из ImageProcessing
        if args.method == "edges":
            result = local_processor.edge_detection(image)
        elif args.method == "corners":
            result = local_processor.corner_detection(image)
        elif args.method == "circles":
            # Убедитесь, что этот метод реализован в вашем классе ImageProcessing
            result = local_processor.circle_detection(image)
        elif args.method == "conv":
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) # Ядро для повышения резкости
            result = local_processor.convolution(image, kernel)
        elif args.method == "gamma":
            gamma_value = 2.2
            result = local_processor.gamma_correction(image, gamma_value)
        
        # Определение пути для сохранения
        if args.output:
            output_path = args.output
        else:
            base, ext = os.path.splitext(args.input)
            output_path = f"{base}_{args.method}_result{ext}"

        # Сохранение результата
        if result is not None:
            cv2.imwrite(output_path, result)
            print(f"Результат сохранён в {output_path}")
        else:
            print(f"Метод '{args.method}' не вернул результат для сохранения.")

if __name__ == "__main__":
    main()