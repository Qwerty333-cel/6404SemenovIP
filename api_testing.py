"""
TODO:
- Реализовать методы __add__ и __sub__ для класса CatImage
- Добавить работу с запросами к API
- Добавить обработку ошибок при запросах
"""
import os
from dotenv import load_dotenv
import requests
import json
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import cv2
from ImageProcessing.implementation.image_processing import ImageProcessing as img_proc
load_dotenv()  # Загружает переменные из .env файла

API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.thecatapi.com/v1"
DEFAULT_TIMEOUT = (5, 20)  # (connect, read)
HEADERS = {"Accept": "application/json", "x-api-key": API_KEY} #означает, что мы готовы и ожидаем получить JSON-ответ от сервера

# try:
#     response = requests.get(

#     )



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
        """Применяет свёртку с указанным ядром к изображению"""
        return img_proc.convolution(self.image, kernel)
    
    def to_grayscale(self) -> np.ndarray:
        """Преобразует изображение в градации серого"""
        return img_proc.rgb_to_grayscale(self.image)
    
    def gamma_correction(self, gamma: float) -> np.ndarray:
        """Применяет гамма коррекцию к изображению"""
        return img_proc.gamma_correction(self.image, gamma)
    
    def edge_detection(self) -> np.ndarray:
        """Применяет детекцию границ к изображению"""
        return img_proc.edge_detection(self.image)
    
    def corner_detection(self) -> np.ndarray:
        """Применяет детекцию углов к изображению"""
        return img_proc.corner_detection(self.image)
    
    def circle_detection(self) -> np.ndarray:
        """Применяет детекцию кругов к изображению"""
        return img_proc.circle_detection(self.image)
    
    def __add__(self, other: 'CatImage') -> np.ndarray:
        """Сложение двух изображений"""
        pass
    
    def __sub__(self, other: 'CatImage') -> np.ndarray:
        """Вычитание двух изображений"""
        pass


class CatImageProcessor:
    """Класс для работы с API, загрузки, обработки и сохранения изображений кошек"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or API_KEY
        self.headers = HEADERS
    
