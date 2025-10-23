import os
import numpy as np
import pandas as pd
from  matplotlib import pyplot as plt


def readfile_gen(chunksize=100):
    chunk = pd.read_csv("video_games.csv", chunksize=chunksize)
    for dataframe in chunk:
        yield dataframe

def get_year_sales(readfile_gen):
    for df in readfile_gen:
        years_sales = pd.DataFrame({'sale': df["Metrics.Sales"], 'year': df["Release.Year"]})
        years_sales.astype(dtype={'sale':np.float16, 'year': np.int16 }) # для уменьшения расхода памяти
        yield years_sales

def sum_sales_dict(get_year_sales):
    result = pd.Series(dtype=np.float16)
    for df in get_year_sales:
        sales = df.groupby("year")['sale'].sum()
        result = result.add(sales, fill_value=0)
    return result.to_dict()

def first_task():
    result = sum_sales_dict(get_year_sales(readfile_gen()))
    min_val = min(result.values())
    max_val = max(result.values())

    # Эта конструкция говорит: "Найди мне ключ в словаре `result`, для которого значение `result[key]` максимально/минимально"
    max_year = max(result, key=result.get)
    min_year = min(result, key=result.get)

    fig, (barplot, textplot) = plt.subplots(nrows=2, ncols=1, gridspec_kw={"height_ratios":[10,2]}, figsize=(10,5))

    bars = barplot.bar(result.keys(), result.values(), color="gray")
    barplot.bar_label(bars)

    current_ylim = barplot.get_ylim()
    barplot.set_ylim(top=current_ylim[1] * 1.05)



    barplot.set_title('Продажи по годам')
    barplot.set_ylabel("Продажи (млн. долларов)")
    barplot.set_xlabel("Года")
    textplot.axis("off")
    textplot.text(x = 0,
                y = 0,
                s = f"Лучшие продажи были в {max_year} году и составляли {max_val:.2f} млн. долларов\nХудшие продажи были в в {min_year} году и составляли {min_val:.2f} млн. долларов")
    fig.savefig("SalesByYears")
    print(f"Лучшие продажи были в {max_year} году и составляли {max_val:.2f} млн. долларов\nХудшие продажи были в в {min_year} году и составляли {min_val:.2f} млн. долларов")


def get_publishers_reviews_gen(readfile_gen):
    for df in readfile_gen:
        publishers_reviews = pd.DataFrame({'publishers': df["Metadata.Publishers"], 'reviews': df["Metrics.Review Score"]})
        publishers_reviews = publishers_reviews.dropna()
        yield publishers_reviews

def disp_reviews_gen(get_publishers_reviews):
    for publishers_reviews in get_publishers_reviews:
        disp_reviews = publishers_reviews.groupby("publishers")['reviews'].var()
        yield disp_reviews.dropna()

def get_topbottom_reviews(disp_reviews_gen):
    top_3 = pd.Series()
    for disp_reviews in disp_reviews_gen:
        top_3 = disp_reviews.sort_values(ascending = False).head(3) # ascending = False - сортировка по убыванию
        print(top_3)
        bottom_3 = disp_reviews.sort_values(ascending = True).head(3) # ascending = True - сортировка по возрастанию

get_topbottom_reviews(
                    disp_reviews_gen(
                                    get_publishers_reviews_gen(
                                                                readfile_gen()
                                                              )
                                    )
                    )