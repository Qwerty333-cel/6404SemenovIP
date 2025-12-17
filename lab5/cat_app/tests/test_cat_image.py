import os
import unittest
import numpy as np
from PIL import Image
import tempfile
import asyncio
from unittest.mock import AsyncMock, patch

from cat_app import CatImageRGB, CatImageGrayscale, CatImageProcessor


class TestCatImageGrayscale(unittest.TestCase):
    def setUp(self):
        """
        Инициализаия данных для тестов.
        """
        self.rgb_img_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        self.gray_img_array = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

        self.gray_img = CatImageGrayscale(self.gray_img_array, "какая-то рандомная ссылка 1", "какая-то порода 1")

    def test_grayscale_to_rgb(self):
        """
        Проверка корректности преобразования grayscale в RGB
        """
        rgb_from_gray = self.gray_img.rgb_image
        self.assertEqual(rgb_from_gray.shape, (64, 64, 3))
        self.assertTrue(np.all(rgb_from_gray[:, :, 0] == rgb_from_gray[:, :, 1]))
        self.assertTrue(np.all(rgb_from_gray[:, :, 1] == rgb_from_gray[:, :, 2]))

    def test_image_addition(self):
        """
        Проверка сложения двух grayscale изображений
        """
        other = CatImageGrayscale(self.gray_img_array, "какая-то рандомная ссылка 2", "какая-то порода 2")
        sum_img = self.gray_img + other
        expected = np.clip(self.gray_img_array.astype(np.int16) * 2, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(sum_img.image, expected)

    def test_custom_edges(self):
        """
        Проверка работы кастомного edge detection на grayscale
        """
        edges = self.gray_img.edges(opencv_realization=False)
        self.assertIsInstance(edges, CatImageGrayscale)
        self.assertEqual(edges.image.shape, (62, 62))


class TestCatImageRGB(unittest.TestCase):
    def setUp(self):
        """
        Инициализаия данных для тестов.
        """
        self.rgb_img_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        self.gray_img_array = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

        self.rgb_img = CatImageRGB(self.rgb_img_array, "какая-то рандомная ссылка 3", "какая-то порода 3")

    def test_rgb_to_grayscale(self):
        """
        Проверка корректности преобразования RGB в grayscale
        """
        gray_from_rgb = self.rgb_img.grayscale_image
        self.assertEqual(gray_from_rgb.shape, (64, 64))
        self.assertEqual(gray_from_rgb.dtype, np.uint8)

    def test_image_addition(self):
        """
        Проверка сложения двух RGB изображений
        """
        other = CatImageRGB(self.rgb_img_array, "какая-то рандомная ссылка 4", "какая-то порода 4")
        sum_img = self.rgb_img + other
        expected = np.clip(self.rgb_img_array.astype(np.int16) * 2, 0, 255).astype(np.uint8)# Ожидаемый результат: удвоение яркости, с учётом защиты от переполнения
        np.testing.assert_array_equal(sum_img.image, expected)

    def test_custom_edges(self):
        """
        Проверка работы кастомного edge detection на RGB и возврата grayscale
        """
        edges = self.rgb_img.edges(opencv_realization=False)
        self.assertIsInstance(edges, CatImageGrayscale)
        self.assertEqual(edges.image.shape, (62, 62))

if __name__ == '__main__':
    unittest.main()