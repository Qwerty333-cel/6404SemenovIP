import os
import requests
from typing import Any, Dict, Optional, Tuple, List, Sequence
import numpy as np
import cv2
from implementation.image_processing import ImageProcessing as img_proc
from implementation.image_processing import log_execution_time
from requests.exceptions import HTTPError, Timeout, ConnectionError, RequestException
from numpy.typing import NDArray
import time
from abc import ABC, abstractmethod


image_processor_instance = img_proc()

TIME_NOW = time.strftime("%Y%m%d_%H%M%S")
HEADERS = {"Accept": "application/json"}
DEFAULT_TIMEOUT = (5, 20)  # (connect, read)

class CatImage:
    """Класс для хранения данных изображения кошки и выполнения операций над ним"""

    def __init__(self, id: str, url: str, breed: str, width: int, height: int, image: np.ndarray):
        self._id = id
        self._url = url
        self._breed = breed
        self._width = width
        self._height = height
        self._image = image

    @property
    def id(self) -> str:
        return self._id
    @property
    def url(self) -> str:
        return self._url
    @property
    def breed(self) -> str:
        return self._breed
    @property
    def width(self) -> int:
        return self._width
    @property
    def height(self) -> int:
        return self._height
    @property
    def image(self) -> np.ndarray:
        return self._image



    def convolution_self(self, kernel: np.ndarray) -> np.ndarray:
        """
        Применяет свёртку с указанным ядром к изображению
        
        Args:
            kernel (np.ndarray): Матрица-ядро для свёртки 
        
        Returns:
            np.ndarray: Изображение с наложенным фильтром свёртки
        """

        return image_processor_instance.convolution(self._image.copy(), kernel, variant = "new")



    def convolution_cv2(self, kernel: np.ndarray) -> np.ndarray:
        """
        Применяет свёртку с указанным ядром к изображению
        
        Args:
            kernel (np.ndarray): Матрица-ядро для свёртки 
        
        Returns:
            np.ndarray: Изображение с наложенным фильтром свёртки
        """

        return image_processor_instance.convolution(self._image.copy(), kernel, variant = "old")
    


    def to_grayscale_self(self) -> np.ndarray:
        """
        Преобразует изображение в градации серого
        
        Args:
        Returns:
            np.ndarray: Изображение в градациях серого (2D)
        """

        return image_processor_instance.rgb_to_grayscale(self._image.copy(), variant = "new")



    def to_grayscale_cv2(self) -> np.ndarray:
        """
        Преобразует изображение в градации серого
        
        Args:
        Returns:
            np.ndarray: Изображение в градациях серого (2D)
        """

        return image_processor_instance.rgb_to_grayscale(self._image.copy(), variant = "old")
    


    def gamma_correction_self(self, gamma: float) -> np.ndarray:
        """
        Применяет гамма коррекцию к изображению (самописный метод)

        Args:
            gamma (float): Коэффициент гамма-преобразования
        
        Returns:
            np.ndarray: Изображение с наложенным гамма-преобразованем
        """

        return image_processor_instance.gamma_correction(self._image.copy(), gamma, variant = "new")

    

    def gamma_correction_cv2(self, gamma: float) -> np.ndarray:
        """
        Применяет гамма коррекцию к изображению с помощью cv2

        Args:
            gamma (float): Коэффициент гамма-преобразования
        
        Returns:
            np.ndarray: Изображение с наложенным гамма-преобразованем
        """

        return image_processor_instance.gamma_correction(self._image.copy(), gamma, variant = "old")
    


    def edge_detection_self(self) -> np.ndarray:
        """
        Применяет поиск границ к изображению

        Args:
        Returns:
            np.ndarray: Ч/б изображение состоящее из границ изображения
        """

        return image_processor_instance.edge_detection(self._image.copy(), variant = "new")
    
    def edge_detection_cv2(self) -> np.ndarray:
        """
        Применяет поиск границ к изображению

        Args:
        Returns:
            np.ndarray: Ч/б изображение состоящее из границ изображения
        """

        return image_processor_instance.edge_detection(self._image.copy(), variant = "old")
    
    

    def corner_detection_cv2(self) -> np.ndarray:
        """
        Применяет поиск углов к изображению с помощью cv2
        
        Args:
        Returns:
            np.ndarray: Изображение с наложенным фильтром поиска углов
        
        """

        return image_processor_instance.corner_detection(self._image.copy(), variant = "old")



    def corner_detection_self(self) -> np.ndarray:
        """
        Применяет поиск углов к изображению (самописный метод)
        
        Args:
        Returns:
            np.ndarray: Изображение с наложенным фильтром поиска углов
        
        """
        return image_processor_instance.corner_detection(self._image.copy(), variant = "new")
    


    def circle_detection(self) -> np.ndarray:
        """
        Применяет поиск кругов к изображению
        
        Args:
        Returns:
            np.ndarray: Изображение с наложенным фильтром поиска кругов
        """

        return image_processor_instance.circle_detection(self._image.copy())
    

    def __add__(self, other: 'CatImage') -> np.ndarray:
        variant = False
        """
        Сложение двух изображений
        
        Args:
            other (CatImage): Второе изображение для сложения
        Returns:
            np.ndarray: Результат сложения двух изображений
        """

        if (self._image.shape != other._image.shape):
            raise ValueError("Изображения должны быть одинакового размера для сложения.")
        
        else:
            copy_self = self.image.copy()
            copy_other = other.image.copy()

            if variant:
                    return cv2.add(copy_self, copy_other)
            
            else:
                # Альтернативный вариант сложения через numpy с обрезкой значений
                added_image = copy_self.astype(np.int16) + copy_other.astype(np.int16)
                np.clip(added_image, 0, 255, out=added_image)
                return added_image.astype(np.uint8)


    def adding_with_correction(self, other: 'CatImage', variant: bool = True, correction: bool = True) -> np.ndarray:
        """
        Сложение двух изображений
        
        Args:
            other (CatImage): Второе изображение для сложения
            variant (bool): Если True, использовать cv2.add, иначе использовать свою реализацию через numpy
            correction (bool): При True подгоняет размер второго изобраения под первое при необходимости
        Returns:
            np.ndarray: Результат сложения двух изображений
        """

        if (self._image.shape != other._image.shape) and not correction:
            raise ValueError("Изображения должны быть одинакового размера для сложения.")
        
        else:
            copy_self = self.image.copy()
            copy_other = other.image.copy()
            if correction:
                if copy_self.shape < copy_other.shape:
                    copy_other = cv2.resize(copy_other, (copy_self.shape[1], copy_self.shape[0]), cv2.INTER_AREA)
                elif copy_self.shape > copy_other.shape:
                    copy_other = cv2.resize(copy_other, (copy_self.shape[1], copy_self.shape[0]), cv2.INTER_CUBIC)
                else: pass

            if variant:
                    return cv2.add(copy_self, copy_other)
            
            else:
                # Альтернативный вариант сложения через numpy с обрезкой значений
                added_image = copy_self.astype(np.int16) + copy_other.astype(np.int16)
                np.clip(added_image, 0, 255, out=added_image)
                return added_image.astype(np.uint8)


    def __sub__(self, other: 'CatImage') -> np.ndarray:
        variant = False
        """
        Вычитание одного изображения из другого
        
        Args:
            other (CatImage): Вычитаемое изображение
            
        Returns:
            np.ndarray: Результат вычитания одного изображения из другого
        """
        copy_self = self.image.copy()
        copy_other = other.image.copy()
        if copy_self.shape != copy_other.shape:
            raise ValueError("Изображения должны быть одинакового размера для вычитания.")
        
        else:
            if variant:
                return cv2.subtract(copy_self, copy_other)
            
            else:
                # Альтернативный вариант вычитания через numpy с обрезкой значений
                subtracted_image = copy_self.astype(np.int16) - copy_other.astype(np.int16)
                np.clip(subtracted_image, 0, 255, out=subtracted_image)
                return subtracted_image.astype(np.uint8)
            

    def subtract_with_correction(self, other: 'CatImage', variant: bool = True, correction: bool = True) -> np.ndarray:
        """
        Вычитание одного изображения из другого с возможностью подгонки размеров
        
        Args:
            other (CatImage): Вычитаемое изображение
            variant (bool): Если True, использовать cv2.subtract, иначе использовать свою реализацию через numpy
            correction (bool): При True подгоняет размер второго изобраения под первое при необходимости
            
        Returns:
            np.ndarray: Результат вычитания одного изображения из другого
        """

        if (self._image.shape != other._image.shape) and not correction:
            raise ValueError("Изображения должны быть одинакового размера для вычитания.")
        
        else:
            copy_self = self.image.copy()
            copy_other = other.image.copy()
            if correction:
                if copy_self.shape < copy_other.shape:
                    copy_other = cv2.resize(copy_other, (copy_self.shape[1], copy_self.shape[0]), cv2.INTER_AREA)
                elif copy_self.shape > copy_other.shape:
                    copy_other = cv2.resize(copy_other, (copy_self.shape[1], copy_self.shape[0]), cv2.INTER_CUBIC)
                else: pass

            if variant:
                return cv2.subtract(copy_self, copy_other)
            
            else:
                # Альтернативный вариант вычитания через numpy с обрезкой значений
                subtracted_image = copy_self.astype(np.int16) - copy_other.astype(np.int16)
                np.clip(subtracted_image, 0, 255, out=subtracted_image)
                return subtracted_image.astype(np.uint8)



    @classmethod
    def from_api_payload(cls, item: Dict[str, Any], image: np.ndarray) -> "CatImage":
        """
        Безопасно строит CatImage из JSON-объекта и уже загруженного изображения.
        Внутри нормализует поля и подставляет дефолты.
        """
        # Название породы (если есть) — берём первую
        breed_name = "Unknown Breed"
        breeds = item.get("breeds")
        if isinstance(breeds, list) and breeds:
            first = breeds[0]
            if isinstance(first, dict):
                breed_name = str(first.get("name", "Unknown Breed"))

        # Остальные поля с базовыми значениями
        return cls(
            id=str(item.get("id", "")),
            url=str(item.get("url", "")),
            breed=breed_name,
            width=int(item.get("width", 0) or 0),
            height=int(item.get("height", 0) or 0),
            image=image,
        )



    def __str__(self) -> str:
        """Возвращает поля класса в виде строки"""
        return f"CatImage(id={self.id}, breed={self.breed}, url={self.url}, width={self.width}, height={self.height})"



class CatImageProcessor:
    """Класс для работы с API, загрузки, обработки и сохранения изображений кошек"""

    def __init__(self, api_key: Optional[str] = os.getenv("API_KEY"), url: Optional[str] = os.getenv("BASE_URL"), headers: Optional[Dict[str, str]] = HEADERS, timeout: Tuple[float, float] = DEFAULT_TIMEOUT):
        self.headers = headers.copy()
        self.api_key = api_key
        self.base_url = url
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("API_KEY не указан ни в параметрах, ни в переменной окружения!")
        if not self.base_url:
            raise ValueError("BASE_URL не указан ни в параметрах, ни в переменной окружения!")
        self.headers["x-api-key"] = self.api_key


    @staticmethod
    def _sanitize(text: str) -> str:
        # безопасное имя файла: только буквы/цифры/подчёркивания/дефисы
        return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text.strip())


    @log_execution_time
    def make_filename(self, index: int, breed: str, suffix: str, ext: str = "png") -> str:
        b = self._sanitize(breed.lower())
        return f"{index}_{b}_{suffix}.{ext}"



    #в качестве параметров лучше указать "has_breeds": 1,"mime_types": "jpg,png" для получения изображений с породами и в нужных форматах
    @log_execution_time
    def request_json(
                     self,
                     path: str,     # Часть URL, которая идет после базового адреса

                     *,     # Звёздочка означает, что все последующие аргументы должны быть переданы по имени (например, api_key="my_key"), а не по позиции
                     params: Optional[Dict[str, Any]]=None,     #Параметры для GET-запроса (то, что идет в URL после ?)(например limit). Optional означает, что можно передать словарь (Dict) или ничего (None).
                     timeout: Optional[Tuple[float, float]]=None,      #Таймауты для подключения и чтения ответа (в секундах). Tuple означает, что это кортеж из двух чисел с плавающей точкой.
                    ) -> Dict[str, Any]:
        """Функция для выполнения GET-запроса к API и обработки ответов"""

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



    @log_execution_time
    def load_image_from_url(
                            self, 
                            url: str, 
                            ) -> np.ndarray:
        """Функция для загрузки изображения через ссылку"""

        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            image_array = np.frombuffer(resp.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Не удалось декодировать изображение.")
            return image
        
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке изображения: {e}")
        


    @log_execution_time
    # NDArray — это специальный тип для аннотаций, а np.object_ уточняет, что dtype этого массива — object.
    def fetch_cats(self, limit:int = 1, has_breeds: int = 1, mime_types: str = "jpg,png") -> NDArray[np.object_]:
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
                try:
                    img = self.load_image_from_url(item.get("url", ""))
                    cat = CatImage.from_api_payload(item, img) # Делегируем разбор JSON-объекта в класс CatImage
                    cats.append(cat)
                except Exception as e:
                    print(f"Ошибка при обработке данных: {e}")

            return np.array(cats, dtype = object)
        
        else:
            raise RuntimeError(f"Ошибка при запросе данных: {data.get('error', 'Unknown error')}")
        
    @log_execution_time
    def save_image(self, 
                   image: np.ndarray, 
                   filename: str, 
                   path: Optional[str] = None
                   ) -> None:
        """
        Сохраняет изображение, гарантируя существование поддиректории.
        
        """
        if path is None:
            # поддиректория по умолчанию: ./saved_images/run_YYYYmmdd_HHMM
            ts = TIME_NOW
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_images", f"run_{ts}")
        os.makedirs(path, exist_ok=True)
        fullpath = os.path.join(path, filename)
        cv2.imwrite(fullpath, image)
        print(f"Saved: {fullpath}")


    @log_execution_time
    def process_cat_image(self, 
                          cats: Sequence[CatImage], 
                          method: str, 
                          path: Optional[str] = None, 
                          gamma: float = 3, 
                          kernel: np.ndarray = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                          ) -> None:
        """
        Функция обработки изображения кошки и сохранения исходников и результов.
        
        Args:
            cats (Sequence[CatImage]): Список объектов CatImage для обработки
            method (str): Метод обработки ("gray", "conv", "gamma", "edges", "corners", "circles", "add", "sub")
            path (Optional[str]): Путь для сохранения изображений. Если None, используется путь по умолчанию.
            gamma (float): Коэффициент гамма-преобразования (используется, если method="gamma")
            kernel (np.ndarray): Ядро свёртки (используется, если method="conv")
        Returns:   
        """
        if method == "gray":
            for idx, cat in enumerate(cats, start=1):
                self.save_image(cat.image, self.make_filename(idx, cat.breed, "original"), path=path)
                self.save_image(cat.to_grayscale_cv2(), self.make_filename(idx, cat.breed, f"{method}_cv2"), path=path)
                self.save_image(cat.to_grayscale_self(), self.make_filename(idx, cat.breed, f"{method}_self"), path=path)

        elif (method == "conv"):
            for idx, cat in enumerate(cats, start=1):
                self.save_image(cat.image, self.make_filename(idx, cat.breed, "original"), path=path)
                self.save_image(cat.convolution_cv2(kernel=kernel), self.make_filename(idx, cat.breed, f"{method}_cv2"), path=path)
                self.save_image(cat.convolution_self(kernel=kernel), self.make_filename(idx, cat.breed, f"{method}_self"), path=path)
        
        elif (method == "gamma"):
            for idx, cat in enumerate(cats, start=1):
                self.save_image(cat.image, self.make_filename(idx, cat.breed, "original"), path=path)
                self.save_image(cat.gamma_correction_cv2(gamma = gamma), self.make_filename(idx, cat.breed, f"{method}_cv2"), path=path)
                self.save_image(cat.gamma_correction_self(gamma = gamma), self.make_filename(idx, cat.breed, f"{method}_self"), path=path)
       
        elif (method == "edges"):
            for idx, cat in enumerate(cats, start=1):
                self.save_image(cat.image, self.make_filename(idx, cat.breed, "original"), path=path)
                self.save_image(cat.edge_detection_cv2(), self.make_filename(idx, cat.breed, f"{method}_cv2"), path=path)
                self.save_image(cat.edge_detection_self(), self.make_filename(idx, cat.breed, f"{method}_self"), path=path)
       
        elif (method == "corners"):
            for idx, cat in enumerate(cats, start=1):
                self.save_image(cat.image, self.make_filename(idx, cat.breed, "original"), path=path)
                self.save_image(cat.corner_detection_cv2(), self.make_filename(idx, cat.breed, f"{method}_cv2"), path=path)
                self.save_image(cat.corner_detection_self(), self.make_filename(idx, cat.breed, f"{method}_self"), path=path)
        
        elif (method == "circles"):
            for idx, cat in enumerate(cats, start=1):
                self.save_image(cat.image, self.make_filename(idx, cat.breed, "original"), path=path)
                self.save_image(cat.circle_detection(), self.make_filename(idx, cat.breed, f"{method}"), path=path)
       
        elif (method == "add"):
            for idx, cat in enumerate(cats, start=1):
                self.save_image(cat.image, self.make_filename(idx, cat.breed, "original"), path=path)
                try:
                    other = next(c for c in cats if c is not cat)  # берём следующее изображение
                    added_image = cat + other
                    self.save_image(added_image, self.make_filename(idx, cat.breed, "add"), path=path)
                except StopIteration:
                    print("Нужно минимум 2 изображения для сложения.")

        elif (method == "sub"):
            for idx, cat in enumerate(cats, start=1):
                self.save_image(cat.image, self.make_filename(idx, cat.breed, "original"), path=path)
                try:
                    other = next(c for c in cats if c is not cat)
                    subtracted_image = cat - other
                    self.save_image(subtracted_image, self.make_filename(idx, cat.breed, "sub"), path=path)
                except StopIteration:
                    print("Нужно минимум 2 изображения для вычитания.")


    