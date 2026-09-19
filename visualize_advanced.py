# -*- coding: utf-8 -*-
"""
高级可视化：相关性热力图、成对图、分组柱状图、核密度图、小提琴图。
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

    # 1. 相关性热力图（只选数值列）
    num_cols = ["survived_int", "pclass", "age", "sibsp", "parch", "fare",
                "FamilySize", "IsAlone", "age_missing", "has_deck"]
    corr = df[num_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("数值特征相关性热力图")
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. 成对图（选取部分列，避免太慢）
    pair_cols = ["survived_int", "age", "fare", "FamilySize", "pclass"]
    sns.pairplot(df[pair_cols], hue="survived_int", diag_kind="kde")
    plt.savefig(FIGURES_DIR / "pairplot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. 性别 × 舱位 生存率分组柱状图
    pivot = df.pivot_table(
        index="sex", columns="pclass", values="survived_int", aggfunc="mean"
    )
    pivot.plot(kind="bar", figsize=(8, 5))
    plt.title("性别 × 舱位 生存率")
    plt.xlabel("性别")
    plt.ylabel("生存率")
    plt.legend(title="舱位")
    plt.savefig(FIGURES_DIR / "survival_sex_pclass_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. 年龄分布核密度图（按生存）
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x="age", hue="survived_int", fill=True, common_norm=False)
    plt.title("不同生存状态的年龄分布")
    plt.xlabel("年龄")
    plt.ylabel("密度")
    plt.savefig(FIGURES_DIR / "age_kde_by_survived.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. 按舱位和生存的小提琴图（年龄）
    plt.figure(figsize=(8, 5))
    sns.violinplot(x="pclass", y="age", hue="survived_int", data=df, split=True)
    plt.title("不同舱位与生存状态的年龄分布")
    plt.xlabel("舱位")
    plt.ylabel("年龄")
    plt.savefig(FIGURES_DIR / "age_violin_pclass_survived.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("高级图表已保存到 figures/ 文件夹。")


if __name__ == "__main__":
    main()