"""
Модуль с абстрактным классом CatImageAbs
"""

from abc import ABC, abstractmethod

import numpy as np

import cv2

from typing import Tuple

from functools import cached_property

class CatImageGrayscale:
    pass

class CatImageRGB:
    pass
    

class CatImageAbstract(ABC):
    """
    Класс обработки изображений котов.

    Публичные неизменяемые поля:
     * image (np.ndarray) - массив с изображением.
     * url (str) - строковое представление ссылки на изображение.
     * breed (str) - строковое представление породы.
    """

    def __init__(self: 'CatImageAbstract', image_array: np.ndarray, url: str, breed: str) -> None:
        """
        Класс обработки изображений котов.

        Публичные неизменяемые поля:
        * image (np.ndarray) - массив с изображением.
        * url (str) - строковое представление ссылки на изображение.
        * breed (str) - строковое представление породы.

        Args:
            image_array (np.ndarray): Массив пикселей.
            url (str): Ссылка на изображение на сайте.
            breed (str): Порода кошки.
        """
        self._image_array = image_array
        self._url = url
        self._breed = breed

    @property
    def image(self: 'CatImageAbstract') -> np.ndarray:
        """
        Неизменаяемый атрибут массива пикселей изображения.
        """
        return self._image_array

    @property
    def url(self: 'CatImageAbstract') -> str:
        """
        Неизменаяемый атрибут ссылки.
        """
        return self._url

    @property
    def breed(self: 'CatImageAbstract') -> str:
        """
        Неизменаяемый атрибут породы.
        """
        return self._breed

    @property
    def tuple_data(self: 'CatImageAbstract') -> Tuple[np.ndarray, str, str]:
        """
        Неизменяемый доступ к данным объекта в формате кортежа.

        Returns:
            Tuple[np.ndarray, str, str]: Кортеж в формате (картинка, url, порода).
        """
        return (
            self.image,
            self.url,
            self.breed,
        )

    @abstractmethod
    def edges(self: 'CatImageAbstract') -> 'CatImageAbstract':
        """
        Выполняет обнаружение границ на изображении.

        Returns:
            np.ndarray: Одноканальное изображение с выделенными границами.
        """
        pass

    @staticmethod
    def create_object(image: np.ndarray, url: str, breed: str) -> 'CatImageAbstract':
        cat_images = None
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            cat_images = CatImageGrayscale(image, url, breed)
        elif image.ndim == 3 and image.shape[2] != 1:
            cat_images = CatImageRGB(image, url, breed)
        return cat_images


class CatImageGrayscale(CatImageAbstract):
    """
    Класс для обработки чёрно-белых изображений котов.
    """

    def __init__(self: 'CatImageGrayscale', image_array: np.ndarray, url: str, breed: str) -> None:
        """
        Инициализация чёрно-белого изображения.

        Args:
            image_array (np.ndarray): Массив пикселей (1 канал).
            url (str): Ссылка на изображение.
            breed (str): Порода кошки.
        """
        super().__init__(image_array, url, breed)
        if image_array.ndim != 2:
            raise ValueError(f"Grayscale should must have 2 channels, but got {image_array.ndim}")

    @property
    def rgb_image(self: 'CatImageRGB') -> np.ndarray:
        """
        Преобразует чёрно-белое изображение в трёхканальное.

        Returns:
            np.ndarray: Трёхканальное изображение.
        """
        return np.stack([self.image] * 3, axis=-1)

    def _convolution(self: 'CatImageGrayscale', image: np.ndarray,
                     kernel: np.ndarray, padding: str = "same",
                     pad_h: int = 0, pad_w: int = 0) -> np.ndarray:
        """
        Выполняет свёртку изображения с заданным ядром.

        Args:
            image (np.ndarray): Входное изображение (чёрно-белое или цветное).
            kernel (np.ndarray): Ядро свёртки (матрица).
            padding (str): Формат выставления отступов.
            pad_h (int): Отступ по вертикали.
            pad_w (int): Отступ по горизонтали.

        Returns:
            np.ndarray: Изображение после применения свёртки.
        """

        if padding == "same":
            kernel_h, kernel_w = kernel.shape
            pad_h = kernel_h // 2
            pad_w = kernel_w // 2
            image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

        kern_h, kern_w = kernel.shape
        img_h, img_w = image.shape
        out_h, out_w = img_h - kern_h + 1, img_w - kern_w + 1
        conv_res = np.zeros((out_h, out_w))
        for heigh in range(out_h):
            for wid in range(out_w):
                conv_res[heigh, wid] = np.sum(
                    image[heigh:heigh+kern_h, wid:wid+kern_w] * kernel)

        return conv_res

    def edges(self: 'CatImageGrayscale', opencv_realization: bool = False) -> 'CatImageGrayscale':
        """
        Выполняет обнаружение границ на изображении.

        Returns:
            np.ndarray: Одноканальное изображение с выделенными границами.
        """
        if not opencv_realization:
            sobel_kernel = np.array([
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1],
            ])
            edges_x = self._convolution(self.image, sobel_kernel, padding="none")
            edges_y = self._convolution(self.image, sobel_kernel.T, padding="none")
            edges = np.sqrt(np.power(edges_x, 2) + np.power(edges_y, 2)).astype(np.uint8)
        else:
            edges = cv2.Canny(self.image, 100, 200)

        return CatImageGrayscale(
            image_array=edges,
            url=self.url,
            breed=self.breed,
        )


class CatImageRGB(CatImageAbstract):
    """
    Класс для обработки RGB изображений котов.
    """

    def __init__(self: 'CatImageRGB', image_array: np.ndarray, url: str, breed: str) -> None:
        """
        Инициализация RGB изображения.

        Args:
            image_array (np.ndarray): Массив пикселей (3 канала).
            url (str): Ссылка на изображение.
            breed (str): Порода кошки.
        """
        super().__init__(image_array, url, breed)
        if image_array.ndim != 3:
            raise ValueError(f"RGB image should have 3 channels, but got {image_array.ndim}")

    @property
    def grayscale_image(self: 'CatImageRGB') -> np.ndarray:
        """
        Преобразует RGB изображение в чёрно-белое.

        Returns:
            np.ndarray: Чёрно-белое изображение.
        """
        return (np.dot(self.image[:, :, :3], [0.299, 0.587, 0.114])).astype(np.uint8)

    def _convolution(self: 'CatImageRGB', image: np.ndarray,
                     kernel: np.ndarray, padding: str = "same",
                     pad_h: int = 0, pad_w: int = 0) -> np.ndarray:
        """
        Выполняет свёртку изображения с заданным ядром.

        Args:
            image (np.ndarray): Входное изображение (чёрно-белое или цветное).
            kernel (np.ndarray): Ядро свёртки (матрица).
            padding (str): Формат выставления отступов.
            pad_h (int): Отступ по вертикали.
            pad_w (int): Отступ по горизонтали.

        Returns:
            np.ndarray: Изображение после применения свёртки.
        """

        if padding == "same":
            kernel_h, kernel_w = kernel.shape
            pad_h = kernel_h // 2
            pad_w = kernel_w // 2
            image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

        kern_h, kern_w = kernel.shape
        img_h, img_w = image.shape[0:2]
        out_h, out_w, out_d = img_h - kern_h + 1, img_w - kern_w + 1, image.ndim
        if image.ndim == 2:
            conv_res = np.zeros((out_h, out_w))
            for heigh in range(out_h):
                for wid in range(out_w):
                    conv_res[heigh, wid] = np.sum(
                        image[heigh:heigh+kern_h, wid:wid+kern_w] * kernel)
        else:
            out_d = image.shape[2]
            conv_res = np.zeros((out_h, out_w, out_d))
            for heigh in range(out_h):
                for wid in range(out_w):
                    for dim in range(out_d):
                        conv_res[heigh, wid, dim] = np.sum(
                            image[heigh:heigh+kern_h, wid:wid+kern_w, dim] * kernel)

        return conv_res

    def edges(self: 'CatImageRGB', opencv_realization: bool = False) -> 'CatImageRGB':
        """
        Выполняет обнаружение границ на изображении.

        Returns:
            np.ndarray: Одноканальное изображение с выделенными границами.
        """
        if not opencv_realization:
            grayscale_image = self.grayscale_image
            sobel_kernel = np.array([
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1],
            ])
            edges_x = self._convolution(grayscale_image, sobel_kernel, padding="none")
            edges_y = self._convolution(grayscale_image, sobel_kernel.T, padding="none")
            edges = np.sqrt(np.power(edges_x, 2) + np.power(edges_y, 2)).astype(np.uint8)
        else:
            gray = self.grayscale_image
            edges = cv2.Canny(gray, 100, 200)

        return CatImageGrayscale(
            image_array=edges,
            url=self.url,
            breed=self.breed
        )
