# -*- coding: utf-8 -*-
"""
数据质量探索：缺失值、重复值、类型、异常值、可视化。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# 中文字体设置，防止图表标题乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    # 读取昨天保存的原始数据
    df = pd.read_csv("titanic_raw.csv")

    print("数据形状：", df.shape)
    print("\n前 5 行：")
    print(df.head())

    print("\n数据类型：")
    print(df.dtypes)

    print("\n描述统计（包含数值和类别）：")
    print(df.describe(include='all'))

    # ---------- 缺失值统计 ----------
    missing = df.isnull().sum()
    missing_pct = df.isnull().mean() * 100
    missing_df = pd.DataFrame({
        "缺失数": missing,
        "缺失比例(%)": missing_pct.round(2)
    })
    missing_df = missing_df[missing_df["缺失数"] > 0].sort_values("缺失比例(%)", ascending=False)
    print("\n缺失值统计：")
    print(missing_df)

    # ---------- 重复值 ----------
    dup_count = df.duplicated().sum()
    print(f"\n重复行数：{dup_count}")
    if dup_count > 0:
        print("重复行示例：")
        print(df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist()).head())

    # ---------- 类别列唯一值 ----------
    # 注意：pandas 3.x 起字符串列的 dtype 是 str 而不是 object，
    # 写 include=['object'] 会触发 Pandas4Warning，这里显式用 'str'
    cat_cols = df.select_dtypes(include=['str', 'category', 'bool']).columns
    print("\n类别列唯一值：")
    for col in cat_cols:
        print(f"{col}: {df[col].unique()}")

    # ---------- 数值列异常值初步（IQR 方法） ----------
    num_cols = df.select_dtypes(include=[np.number]).columns
    print("\n数值列 IQR 异常值计数：")
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"{col}: {len(outliers)} 个异常值 (下界={lower:.2f}, 上界={upper:.2f})")

    # ---------- 缺失值可视化 ----------
    plt.figure(figsize=(10, 5))
    msno.matrix(df)
    plt.title("缺失值矩阵")
    plt.savefig("figures/missing_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    msno.bar(df)
    plt.title("缺失值条形图")
    plt.savefig("figures/missing_bar.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ---------- 箱线图看年龄和票价 ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=df['age'], ax=axes[0])
    axes[0].set_title('年龄箱线图')
    sns.boxplot(y=df['fare'], ax=axes[1])
    axes[1].set_title('票价箱线图')
    plt.tight_layout()
    plt.savefig("figures/boxplot_age_fare.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\n图表已保存到 figures/ 文件夹。")


if __name__ == "__main__":
    main()
