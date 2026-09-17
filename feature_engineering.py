# -*- coding: utf-8 -*-
"""
特征工程：构造 FamilySize、IsAlone、AgeGroup、FareGroup。
输入：titanic_clean.csv
输出：titanic_featured.csv
"""

import pandas as pd
import numpy as np


def main():
    df = pd.read_csv("titanic_clean.csv")
    print("原始形状：", df.shape)

    # 1. 家庭规模：兄弟姐妹/配偶 + 父母/子女 + 自己
    df["FamilySize"] = df["sibsp"] + df["parch"] + 1

    # 2. 是否独行
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # 3. 年龄分箱
    bins = [0, 12, 18, 35, 60, 100]
    labels = ["Child", "Teen", "YoungAdult", "Adult", "Senior"]
    df["AgeGroup"] = pd.cut(df["age"], bins=bins, labels=labels)

    # 4. 票价分箱（按四分位数）
    df["FareGroup"] = pd.qcut(
        df["fare"], q=4, labels=["Low", "Medium", "High", "VeryHigh"]
    )

    # 5. 删除重复列（原始 alone 与 IsAlone 重复）
    if "alone" in df.columns:
        df = df.drop(columns=["alone"])

    # 保存
    df.to_csv("titanic_featured.csv", index=False)
    print("特征工程后形状：", df.shape)
    print("\n新增列：")
    print(df[["FamilySize", "IsAlone", "AgeGroup", "FareGroup"]].head())


if __name__ == "__main__":
    main()