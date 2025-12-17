"""
Модуль обработки картинок с котами
"""

from dotenv import load_dotenv
import functools
import io
import logging
import os
import time
from typing import Any, Callable, Dict, List, Tuple

import asyncio
import aiohttp
import aiofiles

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from PIL import Image

import numpy as np

import requests

from .logs.logging_config import setup_logger
from .cat_image_interface import CatImageAbstract

load_dotenv()

class CatImageProcessor(object):
    """
    Загрузка и обработка изображений с котами с помощью The Cat API
    """

    def __init__(self: 'CatImageProcessor', output_dir: str = "results", logging_path: str = "app.log", logging_dir: str = ".") -> None:
        """
        Загрузка и обработка изображений с котами с помощью The Cat API

        Args:
            output_dir (str, optional): Папка с выходными. Defaults to 'results'.
        """
        self.output_dir = output_dir
        self._logging_path = logging_path
        self._logging_dir = logging_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def async_time_counter(func: Callable) -> Callable:
        """
        Декоратор для обработки времени выполнения подсёта времени работы кода

        Args:
            func (function): Измеряемый метод

        Returns:
            function: Измеряемый метод
        """
        @functools.wraps(func)
        async def wrapper(*args: Tuple, **kwargs: Dict) -> Any:
            """
            Дополнительный функционал для измерения времени

            Args:
                args (Tuple): Кортеж параметров.
                kwargs (Dict): Словарь параметров.

            Returns:
                Any: Возвращаемое значение функции
            """
            start = time.time()
            result = await func(*args, **kwargs)
            end = time.time()
            total_time = end - start
            logging.debug(f"Method {func.__name__} executed in {total_time:.2f} seconds")
            return result
        return wrapper

    async def _download_image(self, session: aiohttp.ClientSession, url: str, 
                              breed: str, index: str) -> Tuple[int, CatImageAbstract]:
        logging.debug(  f"Method _download_image called for image with index: {index}")
        start = time.time()
        async with session.get(url) as response:
            img_data = await response.read()

        image = np.array(Image.open(io.BytesIO(img_data)))
        end = time.time()
        total_time = end - start
        logging.debug(f"Method _download_image for image#{index} executed in {total_time:.2f} seconds")
        return index, CatImageAbstract.create_object(
            image,
            url,
            breed
        )

    @async_time_counter
    async def download_data(self, limit: int = 1) -> List:
        """
        Получение данных о котах по API.

        Args:
            limit (int, optional): Количество изображений. Defaults to 1.

        Returns:
            List: Список с картинками и данными котов.
        """
        logging.info(f"Starting image download ({limit})")

        url = "https://api.thecatapi.com/v1/images/search"
        headers = {"x-api-key": os.getenv('CAT_API_KEY')}
        params = {"limit": limit, "has_breeds": str(True)}
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            logging.debug(f"Requested {limit} cats")
            async with session.get(url, params=params, headers=headers) as response:
                data = await response.json()

            logging.debug(f"Received response with {limit} cats")

            tasks = []
            logging.debug("Requesting cat images")
            for i, item in enumerate(data):
                image_url = item['url']
                breed = item['breeds'][0]['name'] if item.get('breeds') else "unknown"
                task = self._download_image(session, image_url, breed, i)
                tasks.append(task)

            results = await asyncio.gather(*tasks)
        return results

    @staticmethod
    def process_one_image(
            args: Tuple[int, Tuple[np.ndarray, str, str]],
            logging_dir: str = ".", logging_path: str = "app.log",
            ) -> Tuple[int, str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Обработка 1 изображения с помощью OpenCV и кастомного метода.

        Args:
            args (Tuple[int, CatImageAbstract, int]): Кортеж из (Id картинки, CatImage объекта, PID процесса)

        Returns:
            Tuple[int, str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Id картинки, url, порода, картинка до обработки и после.
        """
        setup_logger(log_dir=logging_dir, log_file=logging_path, for_child_process=True)
        logging.debug(f"Method process_one_image called for image {args[0]} in process with PID[{os.getpid()}]")
        start = time.time()
        model = CatImageAbstract.create_object(*args[1])
        processed_image_custom = model.edges(opencv_realization=False)
        processed_image_cv2 = model.edges(opencv_realization=True)
        end = time.time()
        total_time = end - start
        logging.debug(f"Method process_one_image in process with PID[{os.getpid()}] finished in {total_time:.2f} seconds")
        return args[0], model.url, model.breed, model.image, processed_image_custom.image, processed_image_cv2.image


    @async_time_counter
    async def process_images(
            self, downloaded_images: List[Tuple[int, CatImageAbstract]]
            ) -> List[Tuple[int, CatImageAbstract, CatImageAbstract, CatImageAbstract]]:
        """
        Параллельная обработка картинок.

        Args:
            downloaded_images (List[Tuple[int, CatImageAbstract]]): Загруженные картинки с id.

        Returns:
            List[Tuple[int, CatImageAbstract, CatImageAbstract, CatImageAbstract]]: 
            Список из (Id картинки и CatImage объектов до и после обработки.)
        """
        process_args = [(index, cat_image.tuple_data) 
            for index, cat_image in downloaded_images]

        logging.debug(f"Called processing {len(downloaded_images)} images with cats")

        loop = asyncio.get_event_loop()
        tasks = []
        with ProcessPoolExecutor() as executor:
            for args in process_args:
                future = loop.run_in_executor(executor, self.process_one_image, args, self._logging_dir, self._logging_path)
                tasks.append(future)
            
            result = await asyncio.gather(*tasks)

        cat_images_result = []
        for index, url, breed, original_image, processed_custom, processed_cv2 in result:
            cat_images_result.append((
                index,
                CatImageAbstract.create_object(original_image, url, breed),
                CatImageAbstract.create_object(processed_custom, url, breed),
                CatImageAbstract.create_object(processed_cv2, url, breed),
            ))
        return cat_images_result

    @async_time_counter
    async def save_one_dir_async(
            self, index: int, cat_image: CatImageAbstract, 
            cat_image_custom: CatImageAbstract, cat_image_cv2: CatImageAbstract,
            save_dir: str) -> None:
        """
        Асинхронное сохранение изображения

        Args:
            index: Порядковый номер
            cat_image: Объект CatImage
            save_dir: Директория для сохранения
            breed: Порода кота
        """
        logging.debug(f"Called saving dir for {cat_image.breed} cat with id = {index}")
        subdir = os.path.join(save_dir, f"{index}_{cat_image.breed}")
        os.makedirs(subdir, exist_ok=True)

        tasks = [
            self.save_image(
                os.path.join(subdir, f"{index}_{cat_image.breed}_original.jpg"),
                cat_image,
            ),
            self.save_image(
                os.path.join(subdir, f"{index}_{cat_image.breed}_custom.jpg"),
                cat_image_custom,
            ),
            self.save_image(
                os.path.join(subdir, f"{index}_{cat_image.breed}_cv2.jpg"),
                cat_image_cv2,
            ),
        ]
        await asyncio.gather(*tasks)

    @async_time_counter
    async def save_image(self, file_path: str, cat_img: CatImageAbstract):
        logging.debug(f"Called saving {cat_img.breed} processed image - {file_path}")
        image = cat_img.image
        image = Image.fromarray(image)

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')

        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(image_bytes.getvalue())

    @async_time_counter
    async def save_images_async(
            self, processed_images: List[Tuple[int, CatImageAbstract, CatImageAbstract, CatImageAbstract]]
            ) -> None:
        """
        Асинхронное сохранение всех изображений

        Args:
            processed_images: Список обработанных изображений
        """
        local_time = time.localtime()
        logging.debug(f"Called saving {len(processed_images)} processed images with cats")
        save_dir = os.path.join(
            self.output_dir, 
            time.strftime("%Y-%m-%d_%H-%M-%S", local_time)
        )
        await asyncio.to_thread(os.makedirs, save_dir, exist_ok=True)
        
        tasks = []
        for index, cat_image, cat_image_custom, cat_image_cv2 in processed_images:
            task = self.save_one_dir_async(
                index, 
                cat_image, 
                cat_image_custom,
                cat_image_cv2, 
                save_dir,
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)

    @async_time_counter
    async def run_async(self, limit: int = 1):
        try:
            logging.info("=== Starting cat image processing pipeline ===")
            downloaded_images = await self.download_data(limit)
            logging.info(f"Downloaded {len(downloaded_images)} images")
            
            processed_images = await self.process_images(downloaded_images)
            logging.info(f"Processed {len(processed_images)} images")
            
            await self.save_images_async(processed_images)
            logging.info("Results saved successfully")
            logging.info("=== Processing completed ===")
            
        except Exception as e:
            logging.error(f"Error in run_async: {e}")
            raise
