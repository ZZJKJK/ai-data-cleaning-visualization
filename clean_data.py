# -*- coding: utf-8 -*-
"""
数据清洗：处理缺失值、重复值、类型转换、冗余列和异常值。
输入：titanic_raw.csv
输出：titanic_clean.csv
"""

import pandas as pd
import numpy as np


def main():
    # 读取原始数据
    df = pd.read_csv("titanic_raw.csv")
    print("原始数据形状：", df.shape)

    # ---------- 1. 删除重复行 ----------
    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"删除重复行：{before - df.shape[0]} 行")

    # ---------- 2. 处理缺失值较少的列 ----------
    # embarked 和 embark_town 缺失极少，用众数填充
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])

    # ---------- 3. 处理 age ----------
    # 先记录缺失指示变量
    df["age_missing"] = df["age"].isnull().astype(int)

    # 按 pclass 和 sex 分组，用组内中位数填充
    df["age"] = df.groupby(["pclass", "sex"])["age"].transform(
        lambda x: x.fillna(x.median())
    )
    # 若仍有缺失（某组全为空），用全局中位数兜底
    df["age"] = df["age"].fillna(df["age"].median())

    # ---------- 4. 处理 deck ----------
    # deck 缺失超过 70%，转为是否有甲板信息的二值特征，然后删除原列
    df["has_deck"] = df["deck"].notnull().astype(int)
    df = df.drop(columns=["deck"])

    # ---------- 5. 删除冗余列 ----------
    # class 与 pclass 重复；who、adult_male 可由 sex/age 推导；
    # alive 与 survived 重复；embark_town 与 embarked 重复。
    redundant_cols = ["class", "who", "adult_male", "alive", "embark_town"]
    df = df.drop(columns=redundant_cols, errors="ignore")

    # ---------- 6. 类型转换 ----------
    df["survived"] = df["survived"].astype("category")
    df["pclass"] = df["pclass"].astype("category")
    df["sex"] = df["sex"].astype("category")
    df["embarked"] = df["embarked"].astype("category")
    df["alone"] = df["alone"].astype("category")

    # ---------- 7. 异常值处理（以 fare 为例） ----------
    # 先查看 fare 的 IQR 边界
    Q1 = df["fare"].quantile(0.25)
    Q3 = df["fare"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    print(f"\n票价 IQR 下界={lower:.2f}, 上界={upper:.2f}")
    outliers = df[(df["fare"] < lower) | (df["fare"] > upper)]
    print(f"票价异常值数量：{len(outliers)}")

    # 策略：这里选择截断（clip），把超出边界的值压到边界。
    # 也可以选择保留，只在报告中记录。请根据分析目标决定。
    df["fare"] = df["fare"].clip(lower, upper)

    # 年龄异常值检查：小于 0 或大于 100 视为异常
    age_outliers = df[(df["age"] < 0) | (df["age"] > 100)]
    print(f"年龄异常值数量：{len(age_outliers)}")
    # 如有，可用中位数替换或删除；Titanic 中通常没有极端异常。

    # ---------- 8. 保存清洗后数据 ----------
    df.to_csv("titanic_clean.csv", index=False)
    print("\n清洗后数据形状：", df.shape)
    print("\n数据类型：")
    print(df.dtypes)
    print("\n前 5 行：")
    print(df.head())


if __name__ == "__main__":
    main()