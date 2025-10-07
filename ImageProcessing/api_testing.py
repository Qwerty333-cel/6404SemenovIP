"""
TODO:
(сделано) - Реализовать методы __add__ и __sub__ для класса CatImage
- Добавить работу с запросами к API
- Добавить обработку ошибок при запросах
"""
import os
import requests
import json
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import cv2
from implementation.image_processing import ImageProcessing as img_proc
from requests.exceptions import HTTPError, Timeout, ConnectionError, RequestException
from dotenv import load_dotenv
import pathlib
from numpy.typing import NDArray
load_dotenv()  # Загружает переменные из .env файла

image_processor_instance = img_proc()

DEFAULT_TIMEOUT = (5, 20)
HEADERS = {"Accept": "application/json"} # мы готовы и ожидаем получить JSON-ответ от сервера

"""
перенести это в main

from dotenv import load_dotenv
load_dotenv()  # Загружает переменные из .env файла
API_KEY = os.getenv("API_KEY")
и добавить api key в headers
"""

class CatImage:
    """Класс для хранения данных изображения кошки и выполнения операций над ним"""

    def __init__(self, id: str, url: str, breed: str, width: int, height: int, image: np.ndarray):
        self.id = id
        self.url = url
        self.breed = breed
        self.width = width
        self.height = height
        self.image = image  # numpy array



    def convolution(self, kernel: np.ndarray) -> np.ndarray:
        """
        Применяет свёртку с указанным ядром к изображению
        
        Args:
            kernel (np.ndarray): Матрица-ядро для свёртки 
        
        Returns:
            np.ndarray: Изображение с наложенным фильтром свёртки
        """

        return image_processor_instance.convolution(self.image, kernel, variant = "old")
    


    def to_grayscale(self) -> np.ndarray:
        """
        Преобразует изображение в градации серого
        
        Args:
        Returns:
            np.ndarray: Изображение в градациях серого (2D)
        """

        return image_processor_instance.rgb_to_grayscale(self.image, variant = "old")
    


    def gamma_correction(self, gamma: float) -> np.ndarray:
        """
        Применяет гамма коррекцию к изображению

        Args:
            gamma (float): Коэффициент гамма-преобразования
        
        Returns:
            np.ndarray: Изображение с наложенным гамма-преобразованем
        """

        return image_processor_instance.gamma_correction(self.image, gamma, variant = "old")
    


    def edge_detection(self, variant: str = "new") -> np.ndarray:
        """
        Применяет детекцию границ к изображению

        Args:
        Returns:
            np.ndarray: 
        """

        return image_processor_instance.edge_detection(self.image, variant)
    


    def corner_detection(self) -> np.ndarray:
        """Применяет детекцию углов к изображению"""

        return image_processor_instance.corner_detection(self.image, variant = "old")
    


    def circle_detection(self) -> np.ndarray:
        """Применяет детекцию кругов к изображению"""

        return image_processor_instance.circle_detection(self.image)
    


    def __add__(self, other: 'CatImage', variant: bool = True) -> np.ndarray:
        """Сложение двух изображений"""

        if self.image.shape != other.image.shape:
            raise ValueError("Изображения должны быть одинакового размера для сложения.")
        
        else:
            if variant:
                    return cv2.add(self.image, other.image)
            
            else:
                # Альтернативный вариант сложения через numpy с обрезкой значений
                added_image = self.image.astype(np.int16) + other.image.astype(np.int16)
                np.clip(added_image, 0, 255, out=added_image)
                return added_image.astype(np.uint8)
    


    def __sub__(self, other: 'CatImage', variant: bool = True) -> np.ndarray:
        """Вычитание двух изображений"""

        if self.image.shape != other.image.shape:
            raise ValueError("Изображения должны быть одинакового размера для вычитания.")
        
        else:
            if variant:
                return cv2.subtract(self.image, other.image)
            
            else:
                # Альтернативный вариант вычитания через numpy с обрезкой значений
                subtracted_image = self.image.astype(np.int16) - other.image.astype(np.int16)
                np.clip(subtracted_image, 0, 255, out=subtracted_image)
                return subtracted_image.astype(np.uint8)



class CatImageProcessor:
    """Класс для работы с API, загрузки, обработки и сохранения изображений кошек"""

    def __init__(self, api_key: Optional[str] = os.getenv("API_KEY"), url: Optional[str] = os.getenv("BASE_URL")):
        self.headers = HEADERS.copy()
        self.api_key = api_key
        self.base_url = url
        if not self.api_key:
            raise ValueError("API_KEY не указан ни в параметрах, ни в переменной окружения!")
        if not self.base_url:
            raise ValueError("BASE_URL не указан ни в параметрах, ни в переменной окружения!")
        self.headers["x-api-key"] = self.api_key

    #в качестве параметров лучше указать "has_breeds": 1,"mime_types": "jpg,png" для получения изображений с породами и в нужных форматах
    def request_json(
                     self,
                     path: str,     # Часть URL, которая идет после базового адреса

                     *,     # Звёздочка означает, что все последующие аргументы должны быть переданы по имени (например, api_key="my_key"), а не по позиции
                     params: Optional[Dict[str, Any]]=None,     #Параметры для GET-запроса (то, что идет в URL после ?)(например limit). Optional означает, что можно передать словарь (Dict) или ничего (None).
                     timeout: Tuple[float, float]=DEFAULT_TIMEOUT,      #Таймауты для подключения и чтения ответа (в секундах). Tuple означает, что это кортеж из двух чисел с плавающей точкой.
                    ) -> Dict[str, Any]:
        """Aункция для выполнения GET-запроса к API и обработки ответов"""

        url = f"{self.base_url}{path if path.startswith('/') else '/'+path}"
        try:
            r = requests.request(
                                "GET",
                                 url,
                                 params=params,
                                 headers=self.headers,
                                 timeout=timeout
                                 )
            
            # Отлов ограничений / ошибок
            try:
                r.raise_for_status()
            except HTTPError as e:
                # Пробуем вывести тело ошибки (часто JSON)
                try:
                    err = r.json()
                except Exception:
                    err = r.text
                raise HTTPError(f"HTTP {r.status_code} for {url}: {err}") from e

            # Попытка распарсить JSON
            try:
                payload = r.json()
            except ValueError:
                payload = {"raw_text": r.text}
        

            return {"ok": True, "status": r.status_code, "url": r.url, "data": payload}
        except Timeout:
            return {"ok": False, "error": "Timeout", "status": None, "url": url}
        except ConnectionError:
            return {"ok": False, "error": "ConnectionError", "status": None, "url": url}
        except RequestException as e:
            return {"ok": False, "error": f"RequestException: {e}", "status": None, "url": url}



    def load_image_from_url(
                            self, 
                            url: str, 
                            timeout: Tuple[float, float]=DEFAULT_TIMEOUT
                            ) -> np.ndarray:
        """Функция для загрузки изображения через ссылку"""

        try:
            resp = requests.get(url, headers=self.headers, timeout=timeout)
            resp.raise_for_status()
            image_array = np.frombuffer(resp.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Не удалось декодировать изображение.")
            return image
        
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке изображения: {e}")
        

    # NDArray — это специальный тип для аннотаций, а np.object_ уточняет, что dtype этого массива — object.
    def fetch_cats(self, limit:int = 1, has_breeds: int = 1, mime_types: str = "jpg, png") -> NDArray[np.object_]:
        """Функция для получения списка данных кошек с API и создания объектов CatImage"""

        data = self.request_json("/images/search", params = {"limit": limit, "has_breeds": has_breeds,"mime_types": mime_types})
        
        if data["ok"]:

            len_data = len(data["data"])
            if len_data != limit:
                print(f"Внимание: API вернуло {len_data} записей вместо запрошенных {limit}.")
                limit = min(limit, len_data)  # На случай, если API вернул меньше данных, чем запрошено

            cats: List[CatImage] = []

            for i in range(limit):
                item = data["data"][i]
                breeds_list = item.get("breeds")
                breed_name = "Unknown Breed"    # Значение по умолчанию
                if breeds_list:     # Проверяем, что список не None и не пустой
                    breed_info = breeds_list[0]     # Берем первый словарь
                    breed_name = breed_info.get("name", "Unknown Breed")    # Безопасно получаем имя

                try:
                    cat = CatImage(     #Создаём экземпляр класса CatImage
                        id = item.get("id", ""),
                        breed = breed_name,
                        url = item.get("url", ""),
                        width = item.get("width", 0),
                        height = item.get("height", 0),
                        image = self.load_image_from_url(item.get("url", ""))
                    )
                    cats.append(cat)  # Добавляем объект в массив
                    
                except Exception as e:
                    print(f"Ошибка при обработке данных: {e}")

            return np.array(cats, dtype = object)
        
        else:
            raise RuntimeError(f"Ошибка при запросе данных: {data.get('error', 'Unknown error')}")
        

    def save_image(self, image: np.ndarray, filename: str, path: str = None) -> None:
        """Функция для сохранения изображения на диск"""

        if path is None:
            path = pathlib.Path(__file__).resolve().parent.parent / "saved_images"
        path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path/filename), image)
        print(f"Изображение {filename} сохранено в папку: {path}")

    
    def process_cat_image(self, cats: np.ndarray[CatImage], method: str, path: str = None) -> None:
        """Обработка изображения кошки и сохранение исходника и результата"""
        if method == "gray":
            for cat in cats:
                self.save_image(cat.image, f"{cat.id}_{cat.breed}_orig.png", path=path)
                self.save_image(cat.to_grayscale(), f"{cat.id}_{cat.breed}_{method}.png", path=path)
        elif (method == "conv"):
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) # Ядро для повышения резкости
            for cat in cats:
                self.save_image(cat.image, f"{cat.id}_{cat.breed}_orig.png", path=path)
                self.save_image(cat.convolution(kernel), f"{cat.id}_{cat.breed}_{method}.png", path=path)
        elif (method == "gamma"):
            for cat in cats:
                self.save_image(cat.image, f"{cat.id}_{cat.breed}_orig.png", path=path)
                self.save_image(cat.gamma_correction(gamma = 3), f"{cat.id}_{cat.breed}_{method}.png", path=path)
        elif (method == "edges"):
            for cat in cats:
                self.save_image(cat.image, f"{cat.id}_{cat.breed}_orig.png", path=path)
                self.save_image(cat.edge_detection(), f"{cat.id}_{cat.breed}_{method}.png", path=path)
                #self.save_image(cat.edge_detection(variant = "old"), f"{cat.id}_{cat.breed}_{method}_cv2.png", path=path)
        elif (method == "corners"):
            for cat in cats:
                self.save_image(cat.image, f"{cat.id}_{cat.breed}_orig.png", path=path)
                self.save_image(cat.corner_detection(), f"{cat.id}_{cat.breed}_{method}.png", path=path)
        elif (method == "circles"):
            for cat in cats:
                self.save_image(cat.image, f"{cat.id}_{cat.breed}_orig.png", path=path)
                self.save_image(cat.circle_detection(), f"{cat.id}_{cat.breed}_{method}.png", path=path)
        # elif (method == "add"):
        #     for cat in cats:

    