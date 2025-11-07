import os
import cv2
import time
import requests
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List, Sequence
from requests.exceptions import HTTPError, Timeout, ConnectionError, RequestException


from implementation.image_processing import log_execution_time
from implementation.image_processing import ImageProcessing as img_proc


image_processor = img_proc()

TIME_NOW = time.strftime("%Y%m%d_%H%M%S")
HEADERS = {"Accept": "application/json"}
DEFAULT_TIMEOUT = (5, 20)  # (connect, read)



class CatImage(ABC):
    """Класс для хранения данных изображения кошки и выполнения операций над ним"""

    def __init__(self, id: str, url: str, breed: str, width: int, height: int, image: np.ndarray, index: int):
        self._id = str(id)
        self._url = str(url)
        self._breed = str(breed)
        self._image = image
        self._index = index

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
    def image(self) -> np.ndarray:
        return self._image.copy()
    # Поля width и height получаем из размеров изображения
    # (да, можно считывать с ответа сервера, но так надёжнее, я думаю)
    @property
    def width(self) -> int:
        # W — второй элемент (shape = H, W[, C])
        return int(self._image.shape[1])
    @property
    def height(self) -> int:
        # H — первый элемент
        return int(self._image.shape[0])
    
    @property
    def index(self) -> int:
        return self._index
    
        
    @property #Благодаря этому дескриптору, можно вызывать эту функцию как атрибут, то есть через .is_color
    @abstractmethod
    def is_color(self) -> bool:
        """ True для цветных изображений, False для 2D (ч/б)."""
        raise NotImplementedError


    def convolution_self(self, kernel: np.ndarray) -> np.ndarray:
        """
        Применяет свёртку с указанным ядром к изображению
        
        Args:
            kernel (np.ndarray): Матрица-ядро для свёртки 
        
        Returns:
            np.ndarray: Изображение с наложенным фильтром свёртки
        """

        return image_processor.convolution(self.image, kernel, use_cv2 = False)


    def convolution_cv2(self, kernel: np.ndarray) -> np.ndarray:
        """
        Применяет свёртку с указанным ядром к изображению
        
        Args:
            kernel (np.ndarray): Матрица-ядро для свёртки 
        
        Returns:
            np.ndarray: Изображение с наложенным фильтром свёртки
        """

        return image_processor.convolution(self.image, kernel, use_cv2 = True)


    @abstractmethod
    def to_grayscale_self(self) -> np.ndarray:
        """
        Преобразует изображение в градации серого
        
        Args:
        Returns:
            np.ndarray: Изображение в градациях серого (2D)
        """

        raise NotImplementedError


    @abstractmethod
    def to_grayscale_cv2(self) -> np.ndarray:
        """
        Преобразует изображение в градации серого
        
        Args:
        Returns:
            np.ndarray: Изображение в градациях серого (2D)
        """

        raise NotImplementedError
    

    def gamma_correction_self(self, gamma: float) -> np.ndarray:
        """
        Применяет гамма коррекцию к изображению (самописный метод)

        Args:
            gamma (float): Коэффициент гамма-преобразования
        
        Returns:
            np.ndarray: Изображение с наложенным гамма-преобразованем
        """

        return image_processor.gamma_correction(self.image, gamma, use_cv2 = False)


    def gamma_correction_cv2(self, gamma: float) -> np.ndarray:
        """
        Применяет гамма коррекцию к изображению с помощью cv2

        Args:
            gamma (float): Коэффициент гамма-преобразования
        
        Returns:
            np.ndarray: Изображение с наложенным гамма-преобразованем
        """

        return image_processor.gamma_correction(self.image, gamma, use_cv2 = True)


    def edge_detection_self(self) -> np.ndarray:
        """
        Применяет поиск границ к изображению

        Args:
        Returns:
            np.ndarray: Ч/б изображение состоящее из границ изображения
        """

        return image_processor.edge_detection(self.image, use_cv2 = False)
    

    def edge_detection_cv2(self) -> np.ndarray:
        """
        Применяет поиск границ к изображению

        Args:
        Returns:
            np.ndarray: Ч/б изображение состоящее из границ изображения
        """

        return image_processor.edge_detection(self.image, use_cv2 = True)
    

    def corner_detection_cv2(self) -> np.ndarray:
        """
        Применяет поиск углов к изображению с помощью cv2
        
        Args:
        Returns:
            np.ndarray: Изображение с наложенным фильтром поиска углов
        
        """

        return image_processor.corner_detection(self.image, use_cv2 = True)


    def corner_detection_self(self) -> np.ndarray:
        """
        Применяет поиск углов к изображению (самописный метод)
        
        Args:
        Returns:
            np.ndarray: Изображение с наложенным фильтром поиска углов
        
        """
        return image_processor.corner_detection(self.image, use_cv2 = False)
    

    def circle_detection(self) -> np.ndarray:
        """
        Применяет поиск кругов к изображению
        
        Args:
        Returns:
            np.ndarray: Изображение с наложенным фильтром поиска кругов
        """

        return image_processor.circle_detection(self.image)
    

    def __add__(self, other: "CatImage") -> np.ndarray:
        use_cv2 = False
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
            copy_self = self.image
            copy_other = other.image

            if use_cv2:
                    return cv2.add(copy_self, copy_other)
            
            else:
                # Альтернативный вариант сложения через numpy с обрезкой значений
                addedimage = copy_self.astype(np.int16) + copy_other.astype(np.int16)
                np.clip(addedimage, 0, 255, out=addedimage)
                return addedimage.astype(np.uint8)


    def adding_with_correction(self, 
                               other: "CatImage", 
                               use_cv2: bool = False, 
                               correction: bool = True
                               ) -> np.ndarray:
        """
        Сложение двух изображений
        
        Args:
            other (CatImage): Второе изображение для сложения
            use_cv2 (bool): Если True, использовать cv2.add, иначе использовать свою реализацию через numpy
            correction (bool): При True подгоняет размер второго изобраения под первое при необходимости
        Returns:
            np.ndarray: Результат сложения двух изображений
        """
        a = self.image
        b = other.image

        # Проверим число каналов (если есть 3 оси — сравним последнюю)
        if a.ndim != b.ndim or (a.ndim == 3 and a.shape[2] != b.shape[2]):
            raise ValueError("Число каналов должно совпадать для сложения.")

        if (a.shape[:2] != b.shape[:2]) and not correction:
            raise ValueError("Размеры (H, W) должны совпадать, либо разрешите correction=True.")

        if correction and a.shape[:2] != b.shape[:2]:
            if a.shape < b.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]), cv2.INTER_AREA)
            elif a.shape > b.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]), cv2.INTER_CUBIC)
            else: pass

        if use_cv2:
            return cv2.add(a, b)
        else:
            added = a.astype(np.int16) + b.astype(np.int16)
            np.clip(added, 0, 255, out=added)
            return added.astype(np.uint8)


    def __sub__(self, other: "CatImage") -> np.ndarray:
        use_cv2 = False
        """
        Вычитание одного изображения из другого
        
        Args:
            other (CatImage): Вычитаемое изображение
            
        Returns:
            np.ndarray: Результат вычитания одного изображения из другого
        """
        copy_self = self.image
        copy_other = other.image
        if copy_self.shape != copy_other.shape:
            raise ValueError("Изображения должны быть одинакового размера для вычитания.")
        
        else:
            if use_cv2:
                return cv2.subtract(copy_self, copy_other)
            
            else:
                # Альтернативный вариант вычитания через numpy с обрезкой значений
                subtractedimage = copy_self.astype(np.int16) - copy_other.astype(np.int16)
                np.clip(subtractedimage, 0, 255, out=subtractedimage)
                return subtractedimage.astype(np.uint8)


    def subtract_with_correction(self, 
                                 other: "CatImage", 
                                 use_cv2: bool = True, 
                                 correction: bool = True
                                 ) -> np.ndarray:
            """Вычитание с подгонкой HxW (каналы должны совпадать)."""
            a = self.image
            b = other.image
            if a.ndim != b.ndim or (a.ndim == 3 and a.shape[2] != b.shape[2]):
                raise ValueError("Число каналов должно совпадать для вычитания.")

            if correction and a.shape[:2] != b.shape[:2]:
                if a.shape < b.shape:
                    b = cv2.resize(b, (a.shape[1], a.shape[0]), cv2.INTER_AREA)
                elif a.shape > b.shape:
                    b = cv2.resize(b, (a.shape[1], a.shape[0]), cv2.INTER_CUBIC)
                else: pass

            if use_cv2:
                return cv2.subtract(a, b)
            else:
                sub = a.astype(np.int16) - b.astype(np.int16)
                np.clip(sub, 0, 255, out=sub)
                return sub.astype(np.uint8)


    # Фабрика из нагрузки(JSON), возвращённой API после GET-запроса (возвращает подходящий подкласс)
    @classmethod
    def from_api_payload(cls, 
                         item: Dict[str, Any], 
                         image: np.ndarray, 
                         *, 
                         as_gray: bool = False,
                         idx: int
                         ) -> "CatImage":
        """
        Безопасно строит класс-наследник от CatImage из JSON-объекта и уже загруженного изображения.

        Args:
            cls: Класс для создания его объектов
            item (Dict[str, Any]): один из объектов JSON-массива данных из ответа
            image (np.ndarray): Изображение, которое помещаем в объект класса
            as_gray (bool): В каком формате сохранять (чб/цветном)
        Returns:
            "CatImage": Объект класса CatImage \n
                        (В зависимости от аргументов и параметров изображения 
                            или объект от CatImageGray или от CatImageColor)
        """

        # Название породы (берём первую, если есть)
        breed_name = "Unknown Breed"
        breeds = item.get("breeds")
        if isinstance(breeds, list) and breeds:
            first = breeds[0]
            if isinstance(first, dict):
                breed_name = str(first.get("name", "Unknown Breed"))

        cid = str(item.get("id", ""))
        url = str(item.get("url", ""))
        width = int(item.get("width", 0) or 0)
        height = int(item.get("height", 0) or 0)
        if as_gray or image.ndim == 2:
            # превращаем в ч/б (2D)
            gray = image_processor.rgb_to_grayscale(image.copy(), use_cv2=True)
            return CatImageGray(cid, url, breed_name, width, height, gray, index = idx)
        else:
            # возвращаем цветое изображение (3D)
            color = image
            return CatImageColor(cid, url, breed_name, width, height, color, index = idx)

    def __str__(self) -> str:
        """
        Возвращает информационные поля класса в виде строки.

        Args:
        Returns:
            str: Строка вида "CatImage<{color}>(id={self.id}, breed={self.breed}, size={w}x{h}, url={self.url})"
        """

        color = "color" if self.is_color else "gray"
        h, w = self._image.shape[:2]
        return f"CatImage<{color}>(id={self.id}, breed={self.breed}, size={w}x{h}, url={self.url})"



class CatImageGray(CatImage):
    """Класс для хранения данных ч/б(2D) изображений и операций над ними"""

    @property
    def is_color(self) -> bool:
        """ True для цветных изображений (3D), False для ч/б (2D)."""
        return False

    def to_grayscale_self(self) -> np.ndarray:
        """
        Преобразует изображение в градации серого
        
        Args:
        Returns:
            np.ndarray: Изображение в градациях серого (2D)
        """
        return self._image  # Уже ч/б, возвращаем как есть
    
    def to_grayscale_cv2(self):
        """
        Преобразует изображение в градации серого
        
        Args:
        Returns:
            np.ndarray: Изображение в градациях серого (2D)
        """
        return self._image  # Уже ч/б, возвращаем как есть



class CatImageColor(CatImage):
    """Класс для хранения данных цветных(3D) изображений и операций над ними"""

    @property
    def is_color(self) -> bool:
        """ True для цветных изображений (3D), False для ч/б (2D)."""
        return True


    def to_grayscale_self(self) -> np.ndarray:
        """
        Преобразует изображение в градации серого
        
        Args:
        Returns:
            np.ndarray: Изображение в градациях серого (2D)
        """

        return image_processor.rgb_to_grayscale(self._image.copy(), use_cv2 = False)


    def to_grayscale_cv2(self) -> np.ndarray:
        """
        Преобразует изображение в градации серого
        
        Args:
        Returns:
            np.ndarray: Изображение в градациях серого (2D)
        """

        return image_processor.rgb_to_grayscale(self._image.copy(), use_cv2 = True)

    

class CatImageProcessor:
    """Класс для работы с API, загрузки, обработки и сохранения изображений кошек"""

    def __init__(self, 
                 api_key: Optional[str] = os.getenv("API_KEY"), 
                 url: Optional[str] = os.getenv("BASE_URL"), 
                 headers: Optional[Dict[str, str]] = HEADERS, 
                 timeout: Tuple[float, float] = DEFAULT_TIMEOUT):
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
    def make_filename(self, 
                      index: int, 
                      breed: str, 
                      suffix: str, 
                      ext: str = "png") -> str:
        """
        Генерирует безопасное имя файла с учётом индекса, породы, суффикса и расширения.
        
        Args:
            index (int): Порядковый номер изображения (начинается с 1)
            breed (str): Название породы кошки
            suffix (str): Суффикс для файла (например, "original" или метод обработки)
            ext (str): Расширение файла (по умолчанию "png")
        Returns:
            str: Сформированное имя файла
        """

        b = self._sanitize(breed.lower())
        return f"{index}_{b}_{suffix}.{ext}"



    #в качестве параметров лучше указать "has_breeds": 1,"mime_types": "jpg,png" для получения изображений с породами и в нужных форматах
    @log_execution_time
    def request_json(self,
                     path: str,     # Часть URL, которая идет после базового адреса

                     *,     # Звёздочка означает, что все последующие аргументы должны быть переданы по имени (например, api_key="my_key"), а не по позиции
                     params: Optional[Dict[str, Any]]=None,         #Optional означает, что можно передать словарь (Dict) или ничего (None).
                     timeout: Optional[Tuple[float, float]]=None,   #Tuple означает, что это кортеж из двух чисел с плавающей точкой.
                    ) -> Dict[str, Any]:
        """
        Функция для выполнения GET-запроса к API и обработки ответов
        
        Args:
            path (str): Часть URL, которая идет после базового адреса
            params (Optional[Dict[str, Any]]): Параметры для GET-запроса (то, что идет в URL после "?", например limit).
            timeout (Optional[Tuple[float, float]]): Таймауты для подключения и чтения ответа (в секундах).
        Returns:
            Dict[str, Any]: Возвращает словарь типа:
                            "ok": True, "status": status_code, "url": url, "data": payload
        """

        url = f"{self.base_url}{path if path.startswith('/') else '/'+path}"
        try:
            r = requests.request(
                                "GET",
                                 url,
                                 params=params,
                                 headers=self.headers,
                                 timeout=self.timeout
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


    # переписать с использованием aiohttp
    @log_execution_time
    def loadimage_from_url(self, url: str) -> np.ndarray:
        """
        Функция для загрузки изображения через ссылку
        
        Args:
            url (str): Ссылка на изображение
        Returns:
            np.ndarray: Изображение в формате cv2
        """

        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            image_array = np.frombuffer(resp.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

            if image is None:
                raise ValueError("Не удалось декодировать изображение.")
            return image
        
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке изображения: {e}")
        

    # распараллелить скачевание изображение и преобразование в объекты класса catImage
    @log_execution_time
    # NDArray — это специальный тип для аннотаций, а np.object_ уточняет, что dtype этого массива — object.
    def fetch_cats(self, 
                   limit:int = 1, 
                   has_breeds: int = 1, 
                   mime_types: str = "jpg,png",
                   as_gray: bool = False
                   ) -> NDArray[np.object_]:
        """
        Функция для получения списка данных кошек с API и создания объектов CatImage
        
        Args:
            limit (int): Количество изображений котиков
            has_breeds (int): Имеется требуется ли описание (лучше не менять)
            mime_types (str): Расширения картинок, которые мы хотим получать
            as_gray (bool): В каком формате сохранять (чб/цветном)
        Returns:
            NDArray[np.object_]: Массив numpy наполненный объектами класса
        """

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
                    img = self.loadimage_from_url(item.get("url", ""))
                    cat = CatImage.from_api_payload(item, img, as_gray=as_gray, idx = i+1) # Делегируем разбор JSON-объекта в класс CatImage
                    cats.append(cat)
                except Exception as e:
                    print(f"Ошибка при обработке данных: {e}")

            return np.array(cats, dtype = object)
        
        else:
            raise RuntimeError(f"Ошибка при запросе данных: {data.get('error', 'Unknown error')}")
        
    # переписать с использованием aiofiles
    @log_execution_time
    def save_image(self, 
                   image: np.ndarray, 
                   filename: str, 
                   path: Optional[str] = None
                   ) -> None:
        """
        Сохраняет изображение, гарантируя существование поддиректории.
        
        Args:
            image (np.ndarray): Изображение, которое нужно сохранить
            filename (str): Имя файла при сохранении
            path (Optional[str]): Путь сохранения (по умолчанию это savedimages/run_YYYYmmdd_HHMM)
        """
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_images", f"run_{TIME_NOW}")
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
            method (str): Метод обработки ("gray", "conv", "gamma", "edges", "corners", "circles", "add", "sub", "str")
            path (Optional[str]): Путь для сохранения изображений. Если None, используется путь по умолчанию.
            gamma (float): Коэффициент гамма-преобразования (используется, если method="gamma")
            kernel (np.ndarray): Ядро свёртки (используется, если method="conv")
        Returns:   
        """
        
        if method == "gray":
            for cat in cats:
                self.save_image(cat.image, self.make_filename(cat.index, cat.breed, "original"), path=path)
                self.save_image(cat.to_grayscale_cv2(), self.make_filename(cat.index, cat.breed, f"{method}_cv2"), path=path)
                self.save_image(cat.to_grayscale_self(), self.make_filename(cat.index, cat.breed, f"{method}_self"), path=path)

        elif (method == "conv"):
            for cat in cats:
                self.save_image(cat.image, self.make_filename(cat.index, cat.breed, "original"), path=path)
                self.save_image(cat.convolution_cv2(kernel=kernel), self.make_filename(cat.index, cat.breed, f"{method}_cv2"), path=path)
                self.save_image(cat.convolution_self(kernel=kernel), self.make_filename(cat.index, cat.breed, f"{method}_self"), path=path)
        
        elif (method == "gamma"):
            for cat in cats:
                self.save_image(cat.image, self.make_filename(cat.index, cat.breed, "original"), path=path)
                self.save_image(cat.gamma_correction_cv2(gamma = gamma), self.make_filename(cat.index, cat.breed, f"{method}_cv2"), path=path)
                self.save_image(cat.gamma_correction_self(gamma = gamma), self.make_filename(cat.index, cat.breed, f"{method}_self"), path=path)
       
        elif (method == "edges"):
            for cat in cats:
                self.save_image(cat.image, self.make_filename(cat.index, cat.breed, "original"), path=path)
                self.save_image(cat.edge_detection_cv2(), self.make_filename(cat.index, cat.breed, f"{method}_cv2"), path=path)
                self.save_image(cat.edge_detection_self(), self.make_filename(cat.index, cat.breed, f"{method}_self"), path=path)
       
        elif (method == "corners"):
            for cat in cats:
                self.save_image(cat.image, self.make_filename(cat.index, cat.breed, "original"), path=path)
                self.save_image(cat.corner_detection_cv2(), self.make_filename(cat.index, cat.breed, f"{method}_cv2"), path=path)
                self.save_image(cat.corner_detection_self(), self.make_filename(cat.index, cat.breed, f"{method}_self"), path=path)
        
        elif (method == "circles"):
            for cat in cats:
                self.save_image(cat.image, self.make_filename(cat.index, cat.breed, "original"), path=path)
                self.save_image(cat.circle_detection(), self.make_filename(cat.index, cat.breed, f"{method}"), path=path)
       
        elif (method == "add"):
            for cat in cats:
                self.save_image(cat.image, self.make_filename(cat.index, cat.breed, "original"), path=path)
                try:
                    other = next(c for c in cats if c is not cat)  # берём следующее изображение
                    #addedimage = cat + other
                    addedimage = cat.adding_with_correction(other)
                    self.save_image(addedimage, self.make_filename(cat.index, cat.breed, "add"), path=path)
                except StopIteration:
                    print("Нужно минимум 2 изображения для сложения.")
                
                addedimage = cat.adding_with_correction(other)

        elif (method == "sub"):
            for cat in cats:
                self.save_image(cat.image, self.make_filename(cat.index, cat.breed, "original"), path=path)
                try:
                    other = next(c for c in cats if c is not cat)
                    subtractedimage = cat - other
                    self.save_image(subtractedimage, self.make_filename(cat.index, cat.breed, "sub"), path=path)
                except StopIteration:
                    print("Нужно минимум 2 изображения для вычитания.")
        elif (method == "str"):
            for cat in cats:
                print(str(cat))


        else:
            raise ValueError(f"Неизвестный метод: {method}. Ожидаю один из: gray, conv, gamma, edges, corners, circles, add, sub.")


    