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
