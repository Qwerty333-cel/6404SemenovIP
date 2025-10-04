"""
Модуль image_processing.py

Реализация интерфейса IImageProcessing с использованием библиотеки OpenCV.

Содержит класс ImageProcessing, предоставляющий методы для обработки изображений:
- свёртка изображения с ядром
- преобразование RGB-изображения в оттенки серого
- гамма-коррекция
- обнаружение границ (оператор Кэнни)
- обнаружение углов (алгоритм Харриса)
- обнаружение окружностей (метод пока не реализован)

Модуль предназначен для учебных целей (лабораторная работа по курсу "Технологии программирования на Python").
"""

import cv2
import time as t

import interfaces

import numpy as np


class ImageProcessing(interfaces.IImageProcessing):
    """
    Реализация интерфейса IImageProcessing с использованием библиотеки OpenCV.

    Предоставляет методы для обработки изображений, включая свёртку, преобразование
    в оттенки серого, гамма-коррекцию, а также обнаружение границ, углов и окружностей.

    Методы:
        convolution(image, kernel): Выполняет свёртку изображения с ядром.
        rgb_to_grayscale(image): Преобразует RGB-изображение в оттенки серого.
        gamma_correction(image, gamma): Применяет гамма-коррекцию.
        edge_detection(image): Обнаруживает границы (Canny).
        corner_detection(image): Обнаруживает углы (Harris).
        circle_detection(image): Обнаруживает окружности (HoughCircles).
    """

    def _convolution(self, 
                                  image: np.ndarray, 
                                  kernel: np.ndarray,
                                  variant: str = "new") -> np.ndarray:
        """
        Выполняет свёртку изображения с заданным ядром. 
        (альтернативная реализация для работы с float значениями)

        Использует ручную реализацию свёртки.

        Args:
            image (np.ndarray): Входное изображение (может быть цветным или чёрно-белым).
            kernel (np.ndarray): Ядро свёртки (матрица).
            variant (str): свой вариант реализации(new) или через cv2(old)

        Returns:
            np.ndarray: Изображение после применения свёртки.
        """
        if variant == "new":
            #  Создаём выходное изображение с типом float64 для хранения точных результатов
            #  (аналог внутреннего буфера OpenCV с высокой точностью)
            output_image = np.zeros_like(image, dtype=np.float64)
            kernel = np.flipud(np.fliplr(kernel)) 
            # Получаем размеры из оригинального изображения
            if image.ndim == 3:
                height, width, _ = image.shape
                k_height, k_width = kernel.shape  
                pad = k_width // 2

                padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge') # Дополняем изображение краевыми значениями, чтобы свёртка работала на границах


                for y in range(height):
                    print(f"Processing row {y+1}/{height}", end='\r') # Вывод прогресса
                    for x in range(width):
                        region = padded_image[y:y+k_height, x:x+k_width] # Выделяем область kernel_y x kernel_x вокруг текущего пикселя

                        # Применяем ядро свёртки к каждому каналу (R, G, B)
                        r_c = np.sum(region[:,:,2] * kernel)
                        g_c = np.sum(region[:,:,1] * kernel)
                        b_c =np.sum(region[:,:,0] * kernel)

                        output_image[y,x] = [b_c, g_c, r_c] #Собираем новое значение пикселя из трёх каналов

            else:
                height, width = image.shape
                k_height, k_width = kernel.shape  
                pad = k_width // 2

                padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='edge') # Дополняем изображение краевыми значениями, чтобы свёртка работала на границах


                for y in range(height):
                    print(f"Processing row {y+1}/{height}", end='\r') # Вывод прогресса
                    for x in range(width):
                        region = padded_image[y:y+k_height, x:x+k_width] # Выделяем область kernel_y x kernel_x вокруг текущего пикселя

                        # Применяем ядро свёртки к каждому каналу (R, G, B)
                        pix = np.sum(region * kernel)
                    

                        output_image[y,x] = pix #Присваиваем новое значение пикселя 

           
        else:
            output_image = cv2.filter2D(image, -1, kernel)

        return output_image


    # Сделать вторую функцию с возможностью выбора ядра по имени
    def convolution(self, 
                     image: np.ndarray, 
                     kernel: np.ndarray,
                     variant: str = "new") -> np.ndarray:
        """
        Выполняет свёртку изображения с заданным ядром.

        Использует ручную реализацию свёртки.

        Args:
            image (np.ndarray): Входное изображение (может быть цветным или чёрно-белым).
            kernel (np.ndarray): Ядро свёртки (матрица).
            variant (str): свой вариант реализации(new) или через cv2(old)

        Returns:
            np.ndarray: Изображение после применения свёртки.
        """
         
        #  Создаём выходное изображение с типом float64 для хранения точных результатов
        #  (аналог внутреннего буфера OpenCV с высокой точностью)

        start, end = 0.0, 0.0
        start = t.time()

        output_image = self._convolution(image, kernel, variant)
        output_image = np.clip(output_image, 0, 255) # Ограничиваем значения пикселей диапазоном [0, 255]
        output_image = output_image.astype(np.uint8) # Возвращаем изображение в формате uint8

        end = t.time()

        print(f"Execution time: {end - start}")

        return output_image



    def rgb_to_grayscale(self,
                          image: np.ndarray,
                          var: int = 0,
                          variant: str = "new") -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого.

        Использует взвешенное среднее для преобразования каждого пикселя.

        Args:
            image (np.ndarray): Входное RGB-изображение.

        Returns:
            np.ndarray: Одноканальное изображение в оттенках серого.
        """
        start, end = 0.0, 0.0
        start = t.time()

        output_image = self._rgb_to_grayscale(image, var, variant)
        output_image = np.clip(output_image, 0, 255) # Ограничиваем значения пикселей диапазоном [0, 255]

        output_image = output_image.astype(np.uint8) # Возвращаем изображение в формате uint8
        end = t.time()

        print(f"Execution time: {end - start}")

        return output_image


    def _rgb_to_grayscale(self,
                          image: np.ndarray,
                          var: int = 0,
                          variant = "new") -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого. 
        (альтернативная реализация для работы с float значениями)

        Использует взвешенное среднее для преобразования каждого пикселя.

        Args:
            image (np.ndarray): Входное RGB-изображение.

        Returns:
            np.ndarray: Одноканальное изображение в оттенках серого.
        """
    
        if variant == "new":
            #Коэффициенты для взвешенного среднего
            r_coeff = 0.299
            g_coeff = 0.587
            b_coeff = 0.114
           
            if image.ndim == 3:
                
                if var == 1:
                    coeffs = np.array([b_coeff, g_coeff, r_coeff])

                    # Умножаем каждый канал на его коэффициент и суммируем по последней оси (оси каналов)
                    # image.astype(np.float64) - повышаем точность перед умножением
                    output_image = np.dot(image.astype(np.float64), coeffs)
                    
                    # Обрезаем и конвертируем
                    output_image = np.clip(output_image, 0, 255)


                else:
                    #  Создаём изображение с типом float64 для хранения и вычисления более точных результатов
                    mid_image = image.astype(np.float64)
                    # Получаем размеры из оригинального изображения
                    height, width, _ = image.shape
                    # Создаем 2D-массив (высота x ширина), заполненный нулями
                    output_image = np.zeros((height, width), dtype=np.float64)
                    
                    for y in range(height):
                        print(f"Processing row {y+1}/{height}", end='\r') # Вывод прогресса
                        for x in range(width):

                            # Применяем коэффициенты к каждому каналу (R, G, B)
                            r_pix = mid_image[y,x,2] * r_coeff
                            g_pix = mid_image[y,x,1] * g_coeff
                            b_pix = mid_image[y,x,0] * b_coeff
                            
                            output_image[y,x] = 255 - (b_pix + g_pix + r_pix) #Собираем новое значение пикселя из трёх каналов
            else:
                output_image = image
        else:
            output_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return output_image # Возвращаем float для точности



    def _gamma_correction(self, 
                          image: np.ndarray, 
                          gamma: float,
                          variant: str = "new") -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.

        Коррекция осуществляется с помощью таблицы преобразования значений пикселей.

        Args:
            image (np.ndarray): Входное изображение.
            gamma (float): Коэффициент гамма-коррекции (>0).

        Returns:
            np.ndarray: Изображение после гамма-коррекции.
        """
        if variant == "new":
            inv_gamma = 1.0 / gamma

            # Создаем таблицу поиска (LUT)
            table = np.array( [ ( (i / 255.0) ** inv_gamma * 255) for i in range(256) ] )
            output_image = table[image] # Применяем таблицу преобразования ко всему изображению
        else:
            output_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return output_image
    

    def gamma_correction(self, 
                          image: np.ndarray, 
                          gamma: float,
                          variant: str = "new") -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.

        Коррекция осуществляется с помощью таблицы преобразования значений пикселей.

        Args:
            image (np.ndarray): Входное изображение.
            gamma (float): Коэффициент гамма-коррекции (>0).

        Returns:
            np.ndarray: Изображение после гамма-коррекции.
        """

        start, end = 0.0, 0.0
        start = t.time()

        output_image = self._gamma_correction(image, gamma, variant).astype(np.uint8)
        end = t.time()

        print(f"Execution time: {end - start}")

        return output_image



    def edge_detection(self, 
                       image: np.ndarray,
                       variant: str = "new") -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении.

        Использует оператор Собеля для выделения границ.
        Предварительно изображение преобразуется в оттенки серого.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Одноканальное изображение с выделенными границами.
        """

        start, end = 0.0, 0.0
        start = t.time()
        
        if variant == "new":
            gray = self._rgb_to_grayscale(image)

            # Ядро для поиска вертикальных границ (чувствительно к изменениям по X)
            sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            # Ядро для поиска горизонтальных границ (чувствительно к изменениям по Y)
            sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

            sobel_x = self._convolution(gray, sobel_x_kernel)
            sobel_y = self._convolution(gray, sobel_y_kernel)

            sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2) 

            # Нормализация результата
            max_value = np.max(sobel_combined)
            if max_value > 0:
                sobel_normalized = (sobel_combined / max_value) * 255
            else:
                sobel_normalized = sobel_combined # Если все нули

            output_image = sobel_normalized.astype(np.uint8) # Возвращаем изображение в формате uint8

        else:
             gray = self._rgb_to_grayscale(image, variant ="old")
             output_image = cv2.Canny(gray, 100, 200)
    
        end = t.time()

        print(f"Execution time: {end - start}")

        return output_image


    def corner_detection(self, 
                         image: np.ndarray, 
                         k: float = 0.04, 
                         threshold_ratio: float = 0.03,
                         variant: str = "new") -> np.ndarray:
        """
        Выполняет обнаружение углов на изображении.

        Использует алгоритм Харриса для поиска углов.
        Углы выделяются красным цветом на копии исходного изображения.
        Функция последовательно вычисляет градиенты изображения и матрицу Харриса
            для каждого пикселя, и на основе "карты отклика" находит и выделяет углы.

        Args:
            image (np.ndarray): Входное цветное изображение в формате BGR
            k (float): Свободный параметр детектора Харриса, используемый в формуле
                вычисления отклика. Типичные значения лежат в диапазоне 0.04-0.06.
                Значение по умолчанию: 0.04.
            threshold_ratio (float): Коэффициент для определения порога отсечки
                слабых углов. Порог вычисляется как `R.max() * threshold_ratio`.
                Значение по умолчанию: 0.01 (то есть, отбираются углы "силой"
                в 1% от самого сильного).

        Returns:
            np.ndarray: Копия исходного цветного изображения, на которой найденные
                углы отмечены красными точками.
        """
        start, end = 0.0, 0.0
        start = t.time()
        if variant == "new":
            gray = self._rgb_to_grayscale(image)
            

            # Вычисляем производные по X и Y с помощью Собеля

            # Ядро для поиска вертикальных границ (чувствительно к изменениям по X)
            sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            # Ядро для поиска горизонтальных границ (чувствительно к изменениям по Y)
            sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

            sobel_x = self._convolution(gray, sobel_x_kernel)
            sobel_y = self._convolution(gray, sobel_y_kernel)

            # Вычисляем квадраты и произведения градиентов
            sobel_sqrt_x = sobel_x**2
            sobel_sqrt_y = sobel_y**2
            sobel_xy = sobel_x * sobel_y

            # Применяем Гауссово сглаживание к полученным изображениям
            Ghaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

            sob_sqrt_x = self._convolution(sobel_sqrt_x, Ghaussian_kernel)
            sob_sqrt_y = self._convolution(sobel_sqrt_y, Ghaussian_kernel)
            sob_xy = self._convolution(sobel_xy, Ghaussian_kernel)

            # Вычисляем отклик Харриса для каждого пикселя
            #    Формула: R = det(M) - k * (trace(M))**2
            #    det(M) = sob_sqrt_x * sob_sqrt_y - sob_xy**2
            #    trace(M) = sob_sqrt_x + sob_sqrt_y
            det_m = sob_sqrt_x * sob_sqrt_y - sob_xy**2
            trace_m = sob_sqrt_x + sob_sqrt_y
            R = det_m - k * (trace_m)**2

            # Находим углы и рисуем их на исходном изображении
            result_image = image.copy()
            threshold = threshold_ratio * R.max()

            # Находим координаты всех пикселей, которые превышают порог
            # np.where возвращает кортеж с двумя массивами: y-координаты и x-координаты
            corner_coords = np.where(R > threshold)
            
            # Рисуем красные точки по найденным координатам
            # cv2 использует BGR, поэтому красный - это [0, 0, 255]
            result_image[corner_coords] = [0, 0, 255]
            output_image = result_image.astype(np.uint8) # Возвращаем изображение в формате uint8
            
        else:
            gray = self._rgb_to_grayscale(image, variant = "old")
            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray, 2, 3, 0.04)
            dst = cv2.dilate(dst, None)
            result = image.copy()
            result[dst > 0.01 * dst.max()] = [255, 0, 0]
            output_image = result

        end = t.time()

        print(f"Execution time: {end - start}")

        return output_image


    def _draw_circle(self, 
                     image: np.ndarray, 
                     center_y: int, 
                     center_x: int, 
                     radius: int, 
                     color: tuple, 
                     thickness: int = 1) -> None:
        """
        Вспомогательный метод для рисования окружности на изображении без cv2.

        Args:
            image (np.ndarray): Изображение, на котором нужно рисовать.
            center_y (int): Y-координата центра.
            center_x (int): X-координата центра.
            radius (int): Радиус окружности.
            color (tuple): Цвет в формате BGR, например (0, 255, 0) для зелёного.
            thickness (int): Толщина линии.
        """
        height, width, _ = image.shape
        
        # Используем параметрическое уравнение окружности
        # x = x_c + r * cos(a)
        # y = y_c + r * sin(a)
        for angle_deg in range(0, 360):
            angle_rad = np.deg2rad(angle_deg)
            for t in range(thickness):
                r = radius + t - thickness // 2
                y = int(round(center_y + r * np.sin(angle_rad)))
                x = int(round(center_x + r * np.cos(angle_rad)))

                if 0 <= y < height and 0 <= x < width:
                    image[y, x] = color


    def circle_detection(self, 
                         image: np.ndarray, 
                         min_radius: int = 15, 
                         max_radius: int = 150, 
                         edge_threshold: int = 60,
                         accumulator_threshold: int = 35) -> np.ndarray:
        """
        Выполняет обнаружение окружностей на изображении без использования cv2.HoughCircles.

        Реализует круговое преобразование Хафа с использованием градиентов.
        Найденные окружности выделяются зелёным цветом, центры — красным.

        Args:
            image (np.ndarray): Входное изображение (RGB).
            min_radius (int): Минимальный радиус окружности для поиска.
            max_radius (int): Максимальный радиус окружности для поиска.
            edge_threshold (int): Порог для определения "сильных" краев.
            accumulator_threshold (int): Порог для аккумулятора. Чем выше значение,
                                       тем меньше окружностей будет найдено.

        Returns:
            np.ndarray: Изображение с выделенными окружностями.
        """
        # Преобразуем изображение в оттенки серого
        gray_img = self._rgb_to_grayscale(image)
        
        # Применяем размытие для снижения шума
        Ghaussian_kernel_3x3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16


        blurred_img = self._convolution(gray_img, Ghaussian_kernel_3x3)

         # Вычисляем градиенты по x и y
        sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        grad_x = self._convolution(blurred_img, sobel_x_kernel)
        grad_y = self._convolution(blurred_img, sobel_y_kernel)

        # Вычисляем величину и направление градиента
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        # arctan2(y, x) чтобы получить углы во всех квадрантах
        direction = np.arctan2(grad_y, grad_x) 
        
        # Находим пиксели, которые являются частью сильных краев
        edge_pixels_y, edge_pixels_x = np.where(magnitude > edge_threshold)


        # Создание аккумулятора и голосование
        height, width = image.shape[:2]
        num_radii = max_radius - min_radius
        # Аккумулятор: [радиус, y_центра, x_центра]
        accumulator = np.zeros((num_radii, height, width), dtype=np.uint16)
        
        total_pixels = len(edge_pixels_y)

        for i in range(total_pixels):
            if i % 1000 == 0:
                print(f"Обработано пикселей края: {i}/{total_pixels}", end='\r')
            
            y, x = edge_pixels_y[i], edge_pixels_x[i]
            theta = direction[y, x]
            
            # Для каждого возможного радиуса "голосуем" за центр
            for r_idx, r in enumerate(range(min_radius, max_radius)):
                # Вычисляем возможные координаты центра вдоль линии градиента
                # a = x - r * cos(teta), b = y - r * sin(teta)
                # Голосуем в обоих направлениях от края, так как градиент может
                # указывать как внутрь, так и наружу окружности.
                center_x1 = int(round(x - r * np.cos(theta)))
                center_y1 = int(round(y - r * np.sin(theta)))
                
                if 0 <= center_x1 < width and 0 <= center_y1 < height:
                    accumulator[r_idx, center_y1, center_x1] += 1
                
        
        # Находим ячейки аккумулятора, которые превышают порог
        # (это упрощённый поиск максимумов)
        r_indices, y_indices, x_indices = np.where(accumulator > accumulator_threshold)
        
        found_circles = []
        for r_idx, y, x in zip(r_indices, y_indices, x_indices):
            radius = min_radius + r_idx
            found_circles.append((x, y, radius))
            
        print(f"Найдено потенциальных окружностей: {len(found_circles)}")
        
        # Отрисовка результатов
        output_image = image.copy()
        
        # Рисуем найденные окружности
        for x, y, r in found_circles:
            # Окружность зелёным цветом
            self._draw_circle(output_image, y, x, r, (0, 255, 0), thickness=2)
            # Центр красным цветом
            self._draw_circle(output_image, y, x, 2, (0, 0, 255), thickness=2)


        return output_image.astype(np.uint8) # Возвращаем изображение в формате uint8
