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
        plot.figure.savefig("answers/corolation_matrix.png")

    @staticmethod
    def show_distribution(data,
                          name: str = "",
                          title: str = "",
                          x_axis: str = "",
                          y_axis: str = ""):
        plot = sns.distplot(data, kde=False, rug=True)
        plot.set(xlabel=x_axis, ylabel=y_axis)
        plot.set_title(title)
        plot.figure.savefig("answers/distribution_{}.png".format(name))
