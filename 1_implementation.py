#!/usr/bin/env python3
"""
Лабораторная работа: Анализ данных игровой индустрии
Автор: Student
Описание: Анализ данных о видеоиграх с использованием пайплайнов генераторов
"""

import argparse
import os
import time
from typing import Iterator, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator

# Константы для названий колонок
YEAR_COL = "Release.Year"
SALES_COL = "Metrics.Sales"
REVIEW_COL = "Metrics.Review Score"
PUBLISHER_COL = "Metadata.Publishers"
RATING_COL = "Release.Rating"


def csv_chunk_reader(file_path: str, chunk_size: int) -> Iterator[pd.DataFrame]:
    """
    Читает CSV-файл частями для обработки больших данных.
    
    Args:
        file_path: Путь к CSV файлу
        chunk_size: Размер чанка для чтения
        
    Yields:
        pd.DataFrame: Чанк данных из CSV файла
    """
    for data_chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield data_chunk


def data_type_converter(data_generator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """
    Приводит колонки к корректным типам данных.
    
    Args:
        data_generator: Генератор DataFrame'ов
        
    Yields:
        pd.DataFrame: DataFrame с приведенными типами данных
    """
    for dataframe in data_generator:
        if YEAR_COL in dataframe.columns:
            dataframe[YEAR_COL] = pd.to_numeric(dataframe[YEAR_COL], errors='coerce').astype('Int64')
        if SALES_COL in dataframe.columns:
            dataframe[SALES_COL] = pd.to_numeric(dataframe[SALES_COL], errors='coerce')
        if REVIEW_COL in dataframe.columns:
            dataframe[REVIEW_COL] = pd.to_numeric(dataframe[REVIEW_COL], errors='coerce')
        yield dataframe


def yearly_sales_aggregator(data_generator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
    """
    Агрегирует продажи по годам для каждого чанка данных.
    
    Args:
        data_generator: Генератор DataFrame'ов
        
    Yields:
        pd.Series: Серия с суммой продаж по годам для чанка
    """
    for dataframe in data_generator:
        clean_data = dataframe.dropna(subset=[YEAR_COL, SALES_COL])
        if clean_data.empty:
            continue
        yearly_sales = clean_data.groupby(clean_data[YEAR_COL])[SALES_COL].sum()
        yield yearly_sales


def rating_year_counter(data_generator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """
    Подсчитывает количество игр по рейтингам и годам для каждого чанка.
    
    Args:
        data_generator: Генератор DataFrame'ов
        
    Yields:
        pd.DataFrame: Сводная таблица с количеством игр по рейтингам и годам
    """
    for dataframe in data_generator:
        clean_data = dataframe.dropna(subset=[YEAR_COL, RATING_COL])
        if clean_data.empty:
            continue
        rating_pivot = clean_data.groupby([clean_data[YEAR_COL], clean_data[RATING_COL]]).size().unstack(fill_value=0)
        yield rating_pivot


def publisher_stats_calculator(data_generator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """
    Вычисляет частичную статистику (количество, сумма, сумма квадратов) по издателям.
    
    Args:
        data_generator: Генератор DataFrame'ов
        
    Yields:
        pd.DataFrame: DataFrame с частичной статистикой по издателям
    """
    for dataframe in data_generator:
        clean_data = dataframe[[PUBLISHER_COL, REVIEW_COL]].dropna(subset=[PUBLISHER_COL, REVIEW_COL])
        if clean_data.empty:
            continue
        
        publisher_groups = clean_data.groupby(PUBLISHER_COL)[REVIEW_COL].agg(['count', 'sum'])
        squared_reviews = clean_data.assign(review_squared=clean_data[REVIEW_COL] * clean_data[REVIEW_COL])
        sum_squares = squared_reviews.groupby(PUBLISHER_COL)['review_squared'].sum()
        publisher_groups['sum_squares'] = sum_squares
        yield publisher_groups


def series_sum_collector(series_generator: Iterator[pd.Series]) -> pd.Series:
    """
    Собирает и суммирует pd.Series из генератора, используя поэлементное сложение.
    Не создаёт промежуточных списков: серия-аккумулятор формируется сразу по мере обхода.
    
    Args:
        series_generator: Итератор серии агрегированных данных (например, суммы по годам) от пайплайна.
        
    Returns:
        pd.Series: Общая сумма по всем индексам после выравнивания и сложения.
    """
    acc: Optional[pd.Series] = None
    for s in series_generator:
        if s is None or s.empty:
            continue
        acc = s.copy() if acc is None else acc.add(s, fill_value=0)
    if acc is None:
        return pd.Series(dtype=float)
    return acc.sort_index()


def rating_counts_collector(pivot_generator: Iterator[pd.DataFrame]) -> pd.DataFrame:
    """
    Собирает и суммирует таблицы (pd.DataFrame) из генератора, используя add(..., fill_value=0).
    Каждый чанк — это сводная таблица с количеством (по годам/рейтингу).
    Нет промежуточных списков и конкатенации — только стриминговое сложение.
    
    Args:
        pivot_generator: Генератор DataFrame'ов с количеством рейтингов
        
    Returns:
        pd.DataFrame: Итоговые количества рейтингов по годам
    """
    acc: Optional[pd.DataFrame] = None
    for df in pivot_generator:
        if df is None or df.empty:
            continue
        acc = df.copy() if acc is None else acc.add(df, fill_value=0)
    if acc is None:
        return pd.DataFrame()
    acc = acc.fillna(0)
    # при необходимости привести к int, если это счетчики
    try:
        acc = acc.astype(int)
    except Exception:
        pass
    return acc.sort_index()


def publisher_stats_collector(stats_generator: Iterator[pd.DataFrame]) -> pd.DataFrame:
    """
    Собирает статистику издателей по генератору DataFrame, используя поэлементное сложение через add(..., fill_value=0).
    Для каждого издателя аккумулируется количество, сумма, сумма квадратов оценок.
    На выходе — итоговые метрики (count, mean_review, sample_variance) без хранения всех частичных таблиц.
    
    Args:
        stats_generator: Генератор DataFrame'ов с частичной статистикой
        
    Returns:
        pd.DataFrame: DataFrame с итоговой статистикой по издателям
    """
    acc: Optional[pd.DataFrame] = None
    for df in stats_generator:
        if df is None or df.empty:
            continue
        acc = df.copy() if acc is None else acc.add(df, fill_value=0)

    if acc is None or acc.empty:
        return pd.DataFrame()

    total_stats = acc
    n = total_stats['count']
    sum_reviews = total_stats['sum']
    sum_sq = total_stats['sum_squares']

    mean_review = sum_reviews / n
    sample_variance = (sum_sq - (sum_reviews * sum_reviews) / n) / (n - 1)
    sample_variance[n <= 1] = np.nan
    mean_review[n == 0] = np.nan

    results_df = pd.DataFrame({
        'count': n.astype('Int64', errors='ignore'),
        'mean_review': mean_review,
        'sample_variance': sample_variance
    })
    results_df.index.name = PUBLISHER_COL
    return results_df


def sales_visualization(sales_series: pd.Series, output_file: str = 'task1_sales_by_year.png', show_plot: bool = False) -> None:
    """
    Создает график продаж по годам.
    
    Args:
        sales_series: Серия с данными продаж по годам
        output_file: Имя файла для сохранения графика
        show_plot: Показывать ли график
    """
    if sales_series is None or sales_series.empty:
        return
    
    plt.figure(figsize=(12, 7))
    sales_series.plot(kind='bar', color='steelblue', alpha=0.8)
    plt.title('Общие продажи по годам', fontsize=16, pad=20)
    plt.xlabel('Год', fontsize=12)
    plt.ylabel('Продажи (млн. долларов)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


def publisher_variance_visualization(stats_df: pd.DataFrame, publishers: list, 
                                   output_file: str = 'task2_publisher_variance.png', 
                                   show_plot: bool = False) -> None:
    """
    Создает график дисперсии оценок по издателям.
    
    Args:
        stats_df: DataFrame со статистикой издателей
        publishers: Список издателей для отображения
        output_file: Имя файла для сохранения
        show_plot: Показывать ли график
    """
    selected_data = stats_df.loc[stats_df.index.intersection(publishers)]
    variance_values = selected_data['sample_variance'].fillna(0).values
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(range(len(variance_values)), variance_values, color='coral', alpha=0.8, capsize=6)
    plt.xticks(range(len(variance_values)), selected_data.index, rotation=45, ha='right')
    plt.title('Дисперсия оценок по издателям', fontsize=16, pad=20)
    plt.ylabel('Дисперсия оценок', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


def publisher_means_ci_visualization(selected_df: pd.DataFrame, 
                                   output_file: str = 'task2_publisher_means_ci.png', 
                                   show_plot: bool = False, alpha: float = 0.05) -> None:
    """
    Создает график средних оценок издателей с доверительными интервалами.
    
    Args:
        selected_df: DataFrame с выбранными издателями
        output_file: Имя файла для сохранения
        show_plot: Показывать ли график
        alpha: Уровень значимости для доверительного интервала
    """
    df_copy = selected_df.copy()
    n = df_copy['count']
    mean_vals = df_copy['mean_review']
    variance = df_copy['sample_variance']
    
    # Вычисляем доверительный интервал
    confidence_interval = stats.t.ppf(1 - alpha / 2, n - 1) * np.sqrt(variance / n)
    publishers_list = df_copy.index.tolist()
    
    plt.figure(figsize=(12, 7))
    plt.bar(range(len(mean_vals)), mean_vals, yerr=confidence_interval, 
            capsize=8, color='lightgreen', alpha=0.8, edgecolor='darkgreen')
    plt.xticks(range(len(mean_vals)), publishers_list, rotation=45, ha='right')
    plt.ylabel('Средняя оценка', fontsize=12)
    plt.title(f'Средние оценки издателей с {(1-alpha)*100:.0f}% доверительным интервалом', fontsize=16, pad=20)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


def rating_trends_visualization(counts_df: pd.DataFrame, rolling_df: Optional[pd.DataFrame], 
                               output_file: str = 'task3_rating_trends.png', 
                               show_plot: bool = False) -> None:
    """
    Создает график трендов рейтингов с скользящим средним.
    
    Args:
        counts_df: DataFrame с количествами игр по рейтингам
        rolling_df: DataFrame со скользящими средними
        output_file: Имя файла для сохранения
        show_plot: Показывать ли график
    """
    if counts_df is None or counts_df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    year_values = counts_df.index
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, rating in enumerate(counts_df.columns):
        color = colors[i % len(colors)]
        ax.plot(year_values, counts_df[rating], label=f'{rating} (количество)', 
                alpha=0.7, color=color, linewidth=2)
        
        if rolling_df is not None and rating in rolling_df.columns:
            ax.plot(year_values, rolling_df[rating], label=f'{rating} (скользящее среднее)', 
                    linestyle='--', alpha=0.9, color=color, linewidth=2.5)
    
    ax.legend(title='Рейтинг игр', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title('Количество игр по рейтингам по годам (со скользящим средним)', fontsize=16, pad=20)
    ax.set_xlabel('Год', fontsize=12)
    ax.set_ylabel('Количество игр', fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


def correlation_scatter_plot(x_data: pd.Series, y_data: pd.Series, 
                           output_file: str = 'task4_correlation_scatter.png', 
                           show_plot: bool = False, correlation: Optional[float] = None) -> None:
    """
    Создает диаграмму рассеяния для корреляционного анализа.
    
    Args:
        x_data: Данные по оси X
        y_data: Данные по оси Y
        output_file: Имя файла для сохранения
        show_plot: Показывать ли график
        correlation: Коэффициент корреляции для отображения в заголовке
    """
    if x_data is None or y_data is None or len(x_data) == 0:
        return
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x_data, y_data, s=15, alpha=0.6, color='purple', edgecolors='black', linewidth=0.5)
    plt.xlabel('Оценка игры', fontsize=12)
    plt.ylabel('Продажи (млн. долларов)', fontsize=12)
    
    title_text = 'Корреляция между оценкой игры и продажами'
    if correlation is not None:
        title_text += f' (r = {correlation:.4f})'
    
    plt.title(title_text, fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


def execute_task1_pipeline(csv_path: str, chunk_size: int) -> pd.Series:
    """
    Выполняет пайплайн для первого задания - агрегация продаж по годам.
    
    Args:
        csv_path: Путь к CSV файлу
        chunk_size: Размер чанка для чтения
        
    Returns:
        pd.Series: Серия с общими продажами по годам
    """
    data_gen = csv_chunk_reader(csv_path, chunk_size)
    data_gen = data_type_converter(data_gen)
    sales_gen = yearly_sales_aggregator(data_gen)
    return series_sum_collector(sales_gen)


def execute_task2_pipeline(csv_path: str, chunk_size: int, 
                          min_games_count: int = 5, top_k: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Выполняет пайплайн для второго задания - анализ дисперсии оценок издателей.
    
    Args:
        csv_path: Путь к CSV файлу
        chunk_size: Размер чанка для чтения
        min_games_count: Минимальное количество игр для включения издателя
        top_k: Количество издателей с наибольшей/наименьшей дисперсией
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Полная статистика и выбранные издатели
    """
    data_gen = csv_chunk_reader(csv_path, chunk_size)
    data_gen = data_type_converter(data_gen)
    stats_gen = publisher_stats_calculator(data_gen)
    publisher_stats = publisher_stats_collector(stats_gen)
    
    publisher_stats['count'] = publisher_stats['count'].astype(int)
    filtered_publishers = publisher_stats[publisher_stats['count'] >= min_games_count]
    
    if len(filtered_publishers) >= 2 * top_k:
        selected_publishers = pd.concat([
            filtered_publishers.nlargest(top_k, 'sample_variance'),
            filtered_publishers.nsmallest(top_k, 'sample_variance')
        ]).drop_duplicates()
    else:
        selected_publishers = filtered_publishers.copy()
    
    return publisher_stats, selected_publishers


def execute_task3_pipeline(csv_path: str, chunk_size: int, 
                          moving_avg_window: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Выполняет пайплайн для третьего задания - анализ рейтингов по годам.
    
    Args:
        csv_path: Путь к CSV файлу
        chunk_size: Размер чанка для чтения
        moving_avg_window: Размер окна для скользящего среднего
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Количества и скользящие средние
    """
    data_gen = csv_chunk_reader(csv_path, chunk_size)
    data_gen = data_type_converter(data_gen)
    counts_gen = rating_year_counter(data_gen)
    rating_counts = rating_counts_collector(counts_gen)
    
    if rating_counts is None or rating_counts.empty:
        return rating_counts, pd.DataFrame()
    
    rolling_means = rating_counts.rolling(window=moving_avg_window, min_periods=1).mean()
    return rating_counts, rolling_means


def execute_task4_pipeline(csv_path: str, parquet_path: str, use_parquet: bool) -> Optional[pd.DataFrame]:
    """
    Выполняет пайплайн для четвертого задания - корреляционный анализ.
    
    Args:
        csv_path: Путь к CSV файлу
        parquet_path: Путь к Parquet файлу
        use_parquet: Использовать ли Parquet файл
        
    Returns:
        Optional[pd.DataFrame]: DataFrame с данными для корреляционного анализа
    """
    try:
        if use_parquet and os.path.exists(parquet_path):
            dataframe = pd.read_parquet(parquet_path, columns=[REVIEW_COL, SALES_COL])
        else:
            dataframe = pd.read_csv(csv_path, usecols=[REVIEW_COL, SALES_COL])
    except Exception:
        dataframe = pd.read_csv(csv_path)
    
    df_clean = dataframe.copy()
    df_clean[REVIEW_COL] = pd.to_numeric(df_clean[REVIEW_COL], errors='coerce')
    df_clean[SALES_COL] = pd.to_numeric(df_clean[SALES_COL], errors='coerce')
    correlation_data = df_clean.dropna(subset=[REVIEW_COL, SALES_COL])
    
    if correlation_data.shape[0] < 2:
        return None
    
    # Вычисляем корреляцию Пирсона
    r_value, p_value = stats.pearsonr(correlation_data[REVIEW_COL], correlation_data[SALES_COL])
    
    # Сохраняем статистики в атрибутах DataFrame
    correlation_data.attrs['correlation'] = r_value
    correlation_data.attrs['p_value'] = p_value
    
    return correlation_data


def main():
    """Главная функция программы с интерфейсом командной строки."""
    parser = argparse.ArgumentParser(description="Анализ данных игровой индустрии с использованием пайплайнов")
    parser.add_argument('--csv', required=True, help='Путь к CSV файлу с данными игр')
    parser.add_argument('--chunksize', type=int, default=50000, help='Размер чанка для чтения файла')
    parser.add_argument('--parquet', default='video_games.parquet', help='Путь к Parquet файлу')
    parser.add_argument('--no-parquet', action='store_true', help='Отключить использование Parquet')
    parser.add_argument('--ma-window', type=int, default=3, help='Размер окна скользящего среднего')
    parser.add_argument('--min-count', type=int, default=5, help='Минимальное количество игр для издателя')
    parser.add_argument('--top-k', type=int, default=3, help='Количество топ издателей')
    parser.add_argument('--show', action='store_true', help='Показывать графики')
    args = parser.parse_args()
    
    # Настройка работы с Parquet
    use_parquet = False
    if not args.no_parquet:
        if not os.path.exists(args.parquet):
            print("Parquet файл не найден — создаем из CSV (может потребовать много памяти)...")
            try:
                full_dataframe = pd.read_csv(args.csv)
                full_dataframe.to_parquet(args.parquet, engine='pyarrow', index=False)
                print(f"Parquet файл создан: {args.parquet}")
            except Exception as e:
                print(f"Ошибка при создании Parquet файла: {e}")
        
        if os.path.exists(args.parquet):
            use_parquet = True
    
    print("="*60)
    print("АНАЛИЗ ДАННЫХ ИГРОВОЙ ИНДУСТРИИ")
    print("="*60)
    
    # Задание 1: Агрегация продаж по годам
    print("\nЗадание 1: Агрегация продаж по годам...")
    start_time = time.perf_counter()
    total_sales_data = execute_task1_pipeline(args.csv, args.chunksize)
    task1_time = time.perf_counter()
    
    if not total_sales_data.empty:
        # Находим лучший и худший годы
        best_year = total_sales_data.idxmax()
        worst_year = total_sales_data.idxmin()
        max_sales = total_sales_data.max()
        min_sales = total_sales_data.min()
        
        print(f"Лучший год для игровой индустрии: {best_year} ({max_sales:.2f} млн. долларов)")
        print(f"Худший год для игровой индустрии: {worst_year} ({min_sales:.2f} млн. долларов)")
        
        sales_visualization(total_sales_data, show_plot=args.show)
        print("Сохранен график: task1_sales_by_year.png")
    else:
        print("Недостаточно данных для анализа продаж по годам.")
    
    # Задание 2: Дисперсия оценок издателей с доверительными интервалами
    print("\nЗадание 2: Дисперсия оценок издателей...")
    all_publisher_stats, selected_publishers = execute_task2_pipeline(
        args.csv, args.chunksize, min_games_count=args.min_count, top_k=args.top_k)
    task2_time = time.perf_counter()
    
    if not all_publisher_stats.empty and not selected_publishers.empty:
        # Определяем издателей с наибольшей и наименьшей дисперсией
        top_variance = all_publisher_stats.nlargest(args.top_k, 'sample_variance')
        bottom_variance = all_publisher_stats.nsmallest(args.top_k, 'sample_variance')
        
        print(f"\nИздатели с наибольшим разбросом оценок:")
        for i, (publisher, row) in enumerate(top_variance.iterrows(), 1):
            print(f"{i}. {publisher}: дисперсия = {row['sample_variance']:.4f}")
        
        print(f"\nИздатели с наименьшим разбросом оценок:")
        for i, (publisher, row) in enumerate(bottom_variance.iterrows(), 1):
            print(f"{i}. {publisher}: дисперсия = {row['sample_variance']:.4f}")
        
        publisher_variance_visualization(selected_publishers, selected_publishers.index.tolist(), show_plot=args.show)
        print("Сохранен график: task2_publisher_variance.png")
        
        try:
            publisher_means_ci_visualization(selected_publishers, show_plot=args.show)
            print("Сохранен график: task2_publisher_means_ci.png")
        except Exception as e:
            print(f"Ошибка при создании графика доверительных интервалов: {e}")
    else:
        print("Недостаточно данных для анализа дисперсии издателей.")
    
    # Задание 3: Рейтинги по годам со скользящим средним
    print("\nЗадание 3: Количество игр по рейтингам по годам...")
    rating_counts_data, rolling_averages = execute_task3_pipeline(
        args.csv, args.chunksize, moving_avg_window=args.ma_window)
    task3_time = time.perf_counter()
    
    if rating_counts_data is not None and not rating_counts_data.empty:
        print("\nОбщее количество игр по рейтингам:")
        for rating in rating_counts_data.columns:
            total_games = rating_counts_data[rating].sum()
            print(f"Рейтинг {rating}: {total_games} игр")
        
        rating_trends_visualization(rating_counts_data, rolling_averages, show_plot=args.show)
        print("Сохранен график: task3_rating_trends.png")
    else:
        print("Недостаточно данных для анализа рейтингов по годам.")
    
    # Задание 4: Корреляционный анализ (дополнительное задание)
    print("\nЗадание 4 (дополнительное): Корреляция между оценкой и продажами...")
    correlation_result = execute_task4_pipeline(args.csv, args.parquet, use_parquet)
    task4_time = time.perf_counter()
    
    if correlation_result is not None:
        correlation_coeff = correlation_result.attrs['correlation']
        p_value = correlation_result.attrs['p_value']
        
        print(f"Коэффициент корреляции Пирсона: {correlation_coeff:.4f}")
        print(f"P-значение: {p_value:.4e}")
        
        # Интерпретация корреляции
        if abs(correlation_coeff) < 0.1:
            interpretation = "очень слабая"
        elif abs(correlation_coeff) < 0.3:
            interpretation = "слабая"
        elif abs(correlation_coeff) < 0.5:
            interpretation = "умеренная"
        elif abs(correlation_coeff) < 0.7:
            interpretation = "сильная"
        else:
            interpretation = "очень сильная"
        
        print(f"Интерпретация: {interpretation} {'положительная' if correlation_coeff > 0 else 'отрицательная'} корреляция")
        
        correlation_scatter_plot(
            correlation_result[REVIEW_COL], correlation_result[SALES_COL], 
            show_plot=args.show, correlation=correlation_coeff)
        print("Сохранен график: task4_correlation_scatter.png")
    else:
        print("Недостаточно данных для корреляционного анализа.")
    
    # Сравнение скорости чтения CSV vs Parquet
    if not args.no_parquet and os.path.exists(args.parquet):
        print("\nСравнение скорости чтения: CSV vs Parquet...")
        
        # Время чтения CSV
        csv_start = time.perf_counter()
        pd.read_csv(args.csv)
        csv_end = time.perf_counter()
        csv_time = csv_end - csv_start
        
        # Время чтения Parquet
        parquet_start = time.perf_counter()
        pd.read_parquet(args.parquet)
        parquet_end = time.perf_counter()
        parquet_time = parquet_end - parquet_start
        
        print(f"CSV: {csv_time:.3f} секунд")
        print(f"Parquet: {parquet_time:.3f} секунд")
        
        if parquet_time > 0 and csv_time > 0:
            speedup = csv_time / parquet_time
            print(f"Parquet в {speedup:.2f} раза быстрее CSV для полного чтения")
    else:
        print("\nParquet недоступен или отключен — пропускаем сравнение скорости чтения.")
    
    # Вывод времени выполнения заданий
    print("\n" + "="*60)
    print("ВРЕМЯ ВЫПОЛНЕНИЯ:")
    print(f"Задание 1: {task1_time - start_time:.2f} сек")
    print(f"Задание 2: {task2_time - task1_time:.2f} сек") 
    print(f"Задание 3: {task3_time - task2_time:.2f} сек")
    print(f"Задание 4: {task4_time - task3_time:.2f} сек")
    print(f"Общее время: {task4_time - start_time:.2f} сек")
    print("="*60)
    print("Анализ завершен!")


if __name__ == '__main__':
    main()
