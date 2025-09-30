import matplotlib

from config.name.data_analysis.AnalysisTaskTypeEnName import AnalysisTaskTypeEnName

# 指定使用AGG后端，避免GUI线程冲突
matplotlib.use('Agg')  # 在导入pyplot之前设置
import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import networkx as nx
import pandas as pd

from entity.machine_learning_model.BayesianRegression import BayesianRegression
from utils.common.MyColor import MyColor


if __name__ == '__main__':
    window_data = {
        10: 0.3687,
        20: 0.4097,
        30: 0.3594,
        40: 0.2994,
        50: 0.3439,
        60: 0.3131,
        70: 0.4095,
        80: 0.2346
    }

    plt.clf()

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot([_ for _ in window_data.keys()], [_ for _ in window_data.values()], color=MyColor.COLOR_1, marker='o', linewidth=2, markersize=8)
    plt.xlabel("Window Size", fontsize=14, labelpad=16)
    plt.ylabel("F1 Score", fontsize=14, labelpad=16)
    plt.grid(True)
    # plt.legend(loc="upper right", fontsize=14)

    plt.savefig(f"./time_window.svg")
