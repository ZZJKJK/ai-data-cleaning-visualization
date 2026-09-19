# -*- coding: utf-8 -*-
"""
基础可视化：直方图、箱线图、条形图、散点图。
输入：titanic_featured.csv
输出：figures/ 下的图片
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "titanic_featured.csv"
FIGURES_DIR = BASE_DIR / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE)
    df["survived_int"] = df["survived"].astype(int)

    # 1. 年龄分布直方图
    plt.figure(figsize=(8, 5))
    sns.histplot(df["age"], bins=30, kde=True, color="steelblue")
    plt.title("年龄分布")
    plt.xlabel("年龄")
    plt.ylabel("人数")
    plt.savefig(FIGURES_DIR / "age_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. 票价分布直方图
    plt.figure(figsize=(8, 5))
    sns.histplot(df["fare"], bins=30, kde=True, color="salmon")
    plt.title("票价分布")
    plt.xlabel("票价")
    plt.ylabel("人数")
    plt.savefig(FIGURES_DIR / "fare_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. 按舱位看票价箱线图
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="pclass", y="fare", data=df)
    plt.title("不同舱位的票价分布")
    plt.xlabel("舱位")
    plt.ylabel("票价")
    plt.savefig(FIGURES_DIR / "fare_by_pclass_box.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. 性别与生存人数条形图
    plt.figure(figsize=(8, 5))
    sns.countplot(x="sex", hue="survived", data=df)
    plt.title("不同性别的生存人数")
    plt.xlabel("性别")
    plt.ylabel("人数")
    plt.savefig(FIGURES_DIR / "survived_by_sex_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. 年龄 vs 票价散点图，颜色区分生存
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="age", y="fare", hue="survived", data=df, alpha=0.7)
    plt.title("年龄与票价散点图（按生存着色）")
    plt.xlabel("年龄")
    plt.ylabel("票价")
    plt.savefig(FIGURES_DIR / "age_fare_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("基础图表已保存到 figures/ 文件夹。")


if __name__ == "__main__":
    main()