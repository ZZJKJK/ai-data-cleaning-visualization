# -*- coding: utf-8 -*-
"""
加载 Titanic 数据集，查看基本信息。
"""

import seaborn as sns
import pandas as pd


def main():
    # 从 seaborn 内置数据集加载 Titanic
    df = sns.load_dataset("titanic")

    # 打印前 5 行
    print("前 5 行数据：")
    print(df.head())

    # 打印数据形状（行数, 列数）
    print("\n数据形状：", df.shape)

    # 打印所有列名
    print("\n列名：", df.columns.tolist())

    # 打印每列的数据类型和非空数量
    print("\n数据类型与非空数量：")
    print(df.info())

    # 保存一份原始数据快照到 CSV，方便后续使用
    df.to_csv("titanic_raw.csv", index=False)
    print("\n已保存原始数据到 titanic_raw.csv")


if __name__ == "__main__":
    main()