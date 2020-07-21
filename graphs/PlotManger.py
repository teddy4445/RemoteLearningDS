import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class PlotManager:

    def __init__(self):
        pass

    @staticmethod
    def corolation_matrix(df) -> None:
        fig, ax = plt.subplots(figsize=(20, 20))
        plot = sns.heatmap(df.corr(),
                           annot=False,
                           fmt='.2g',
                           vmin=-1,
                           vmax=1,
                           center=0,
                           cmap='coolwarm',
                           linewidths=0,
                           linecolor='black',
                           ax=ax)
        save_path = "answers/corolation_matrix.png"
        plot.figure.savefig(save_path)
        print("PlotManager: Save distribution chart to: '{}'".format(save_path))
        plt.close()

    @staticmethod
    def pie_chart(sizes: list,
                  labels: list,
                  name: str = ""):
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False)
        ax1.axis('equal')
        save_path = "answers/pie_{}.png".format(name)
        plt.savefig(save_path)
        print("PlotManager: Save pie chart to: '{}'".format(save_path))
        plt.close()

    @staticmethod
    def show_distribution(data,
                          bins = None,
                          kde: bool = False,
                          rug: bool = False,
                          hist: bool = False,
                          name: str = "",
                          title: str = "",
                          x_axis: str = "",
                          y_axis: str = ""):
        plot = sns.distplot(data, bins=bins, kde=kde, rug=rug, hist=hist)
        plot.set(xlabel=x_axis, ylabel=y_axis)
        plot.set_title(title)
        save_path = "answers/distribution_{}.png".format(name)
        plt.savefig(save_path)
        print("PlotManager: Save distribution chart to: '{}'".format(save_path))
        plt.close()
