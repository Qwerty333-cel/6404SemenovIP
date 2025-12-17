import os
import unittest
import numpy as np
from PIL import Image
import tempfile
import asyncio
from unittest.mock import AsyncMock, patch
import io

from cat_app import CatImageRGB, CatImageGrayscale, CatImageProcessor


class TestCatImageProcessorFileIO(unittest.TestCase):
    def setUp(self):
        """
        Инициализаия данных для тестов.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.processor = CatImageProcessor(output_dir=self.temp_dir)

    def test_save_image_writes_correct_file(self):
        """
        Проверка правильности сохранения в файл.
        """
        img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        cat_img = CatImageRGB(img_array, "ссылка на кота", "кот")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            file_path = tmp.name
        os.unlink(file_path)

        asyncio.run(self.processor.save_image(file_path, cat_img))

        self.assertTrue(os.path.exists(file_path))
        reloaded_pil = Image.open(file_path)
        reloaded_array = np.array(reloaded_pil)

        self.assertEqual(reloaded_array.shape, (32, 32, 3))
        self.assertEqual(reloaded_array.dtype, np.uint8)
        os.remove(file_path)

    def test_save_images_async_creates_expected_structure(self):
        """
        Проверка сохранения в правильный файл и правильную структуру папок.
        """
        img = np.random.randint(0, 256, (24, 24, 3), dtype=np.uint8)
        cat = CatImageRGB(img, "рандомная ссылка на кота", "рандомная_порода")
        processed = [(0, cat, cat, cat)] #Формируем список в формате: (индекс, оригинал, обработанное_custom, обработанное_cv2)

        asyncio.run(self.processor.save_images_async(processed)) # Метод должен создать структуру папок и сохранить 3 файла

        timestamp_dirs = os.listdir(self.temp_dir)# Получаем список папок в temp_dir
        self.assertEqual(len(timestamp_dirs), 1)
        breed_dir = os.path.join(self.temp_dir, timestamp_dirs[0], "0_рандомная_порода")
        self.assertTrue(os.path.isdir(breed_dir))

        files = set(os.listdir(breed_dir))
        expected = {"0_рандомная_порода_original.jpg", "0_рандомная_порода_custom.jpg", "0_рандомная_порода_cv2.jpg"}
        self.assertEqual(files, expected) # Проверяем, что сохранены именно эти 3 файла

        for f in expected:
            with Image.open(os.path.join(breed_dir, f)) as img:
                self.assertIsInstance(img, Image.Image) # Проверяем, что открытый объект действительно является PIL изображением


class TestCatImageProcessorAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.processor = CatImageProcessor(output_dir=self.temp_dir)

    @patch('cat_app.cat_image_processor.aiohttp.ClientSession.get') # @patch заменяет реальные HTTP запросы на мок-объект
    async def test_download_data_calls_api_correctly(self, mock_get):
        """
        Тесттирование корректного обращения к API.
        """
        fake_img = Image.new('RGB', (1, 1), color='red')
        img_bytes = io.BytesIO() # Создаем буфер в памяти для хранения байтов изображения
        fake_img.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()

        mock_response_json = AsyncMock() # Создаем мок асинхронного ответа для JSON запроса к API
        mock_response_json.json.return_value = [{# Задаем возвращаемое значение метода json()
            "url": "http://example.com/cat123.jpg",
            "breeds": [{"name": "Scottish Fold"}]
        }]

        mock_response_img = AsyncMock()# Создаем мок асинхронного ответа для загрузки изображения
        mock_response_img.read.return_value = img_bytes# Задаем возвращаемое значение метода read() - байты фейкового изображени

        mock_get.side_effect = [# side_effect - последовательность значений, возвращаемых при каждом вызове
            mock_response_json,
            mock_response_img
        ]

        mock_response_json.__aenter__.return_value = mock_response_json # Настраиваем моки для работы с async context manager (async with) __aenter__ вызывается при входе в блок async with
        mock_response_img.__aenter__.return_value = mock_response_img

        results = await self.processor.download_data(limit=1)  # Вызываем тестируемый метод - запрос на скачивание 1 изображения

        self.assertEqual(len(results), 1)
        index, cat_img = results[0]
        self.assertEqual(index, 0)
        self.assertEqual(cat_img.url, "http://example.com/cat123.jpg")
        self.assertEqual(cat_img.breed, "Scottish Fold")
        self.assertEqual(cat_img.image.shape, (1, 1, 3))

        self.assertEqual(mock_get.call_count, 2) #Проверяем, что mock_get был вызван ровно 2 раза
        json_call, img_call = mock_get.call_args_list # Получаем список всех вызовов мока (аргументы каждого вызова)
        self.assertEqual(json_call[0][0], "https://api.thecatapi.com/v1/images/search")  # Проверяем, что первый вызов был к API endpoint
        self.assertEqual(img_call[0][0], "http://example.com/cat123.jpg") # Проверяем, что второй вызов был к URL изображения из ответа API

if __name__ == '__main__':
    unittest.main()