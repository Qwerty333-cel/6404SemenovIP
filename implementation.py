import os
import numpy as np
import pandas as pd
from  matplotlib import pyplot as plt

def readfile_gen(lines = 0):
    with open("video_games.csv", 'r', encoding="utf-8") as dataset:
        next(dataset) # пропуск первой строки содержащей шапку таблицы(названия столбцов)
        if not lines: 
            for line in dataset:
                yield line.strip() # убираем пробелы
        else:
            count = 0
            for line in dataset:
                if count == lines:
                    break
                yield line.strip() # убираем пробелы
                count += 1

def str_clearer(line: str) -> list:
    z = line.split('"')
    w = []
    for item in z:
        if item == ',':
            pass
        elif item == "":
            item = "filled_data"
            w.append(item)
        else:
            w.append(item)
    return w

def lineprocessing_gen(readfile_geneartor):
    for line in readfile_geneartor:
        l_list = str_clearer(line)
        yield l_list[11], l_list[16] # 11 - продажи, 16 - год

def extremum_by_years(lineprocessing_generator) -> dict:
    ext_b_y = {}
    for sale, year in lineprocessing_generator:
        if year in ext_b_y.keys():
            ext_b_y[year] += float(sale)
        else:
            ext_b_y[year] = float(sale)
    for year in ext_b_y:
        ext_b_y[year] = round(ext_b_y[year], 2)
    return ext_b_y



r = extremum_by_years(lineprocessing_gen(readfile_gen()))



min_val = min(r.values())
max_val = max(r.values())

# Эта конструкция говорит: "Найди мне ключ в словаре `r`, для которого значение `r[key]` максимально/минимально"
max_year = max(r, key=r.get)
min_year = min(r, key=r.get)


fig, (barplot, textplot) = plt.subplots(nrows=2, ncols=1, gridspec_kw={"height_ratios":[10,2]}, figsize=(10,5))

barplot.bar(r.keys(), r.values(), color="gray")
barplot.set_title('Продажи по годам')
barplot.set_ylabel("Продажи (млн. долларов)")
barplot.set_xlabel("Года")
textplot.axis("off")
textplot.text(x = 0,
              y = 0,
              s = f"Лучшие продажи были в {max_year} году и составляли {max_val} млн. долларов\nХудшие продажи были в в {min_year} году и составляли {min_val} млн. долларов")
fig.savefig("SalesByYears")
print(f"Лучшие продажи были в {max_year} году и составляли {max_val} млн. долларов\nХудшие продажи были в в {min_year} году и составляли {min_val} млн. долларов")