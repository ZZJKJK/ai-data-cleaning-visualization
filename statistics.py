# -*- coding: utf-8 -*-
"""
分组统计：按性别、舱位、是否独行、家庭规模、年龄组、票价组计算生存率。
输入：titanic_featured.csv
输出：终端打印统计结果，并保存 statistics.md
"""

import pandas as pd


def main():
    df = pd.read_csv("titanic_featured.csv")

    # 生存率需要数值型
    df["survived_int"] = df["survived"].astype(int)

    lines = []
    lines.append("# Titanic 分组统计\n")

    # 总体生存率
    overall = df["survived_int"].mean()
    lines.append(f"## 总体生存率\n\n- 总体生存率：{overall:.2%}\n")

    # 按性别
    lines.append("## 按性别\n")
    sex_stat = df.groupby("sex")["survived_int"].agg(["count", "mean"])
    sex_stat.columns = ["人数", "生存率"]
    sex_stat["生存率"] = sex_stat["生存率"].map(lambda x: f"{x:.2%}")
    lines.append(sex_stat.to_markdown() + "\n")

    # 按舱位
    lines.append("## 按舱位\n")
    pclass_stat = df.groupby("pclass")["survived_int"].agg(["count", "mean"])
    pclass_stat.columns = ["人数", "生存率"]
    pclass_stat["生存率"] = pclass_stat["生存率"].map(lambda x: f"{x:.2%}")
    lines.append(pclass_stat.to_markdown() + "\n")

    # 按是否独行
    lines.append("## 按是否独行\n")
    alone_stat = df.groupby("IsAlone")["survived_int"].agg(["count", "mean"])
    alone_stat.columns = ["人数", "生存率"]
    alone_stat["生存率"] = alone_stat["生存率"].map(lambda x: f"{x:.2%}")
    alone_stat.index = ["有家人", "独行"]
    lines.append(alone_stat.to_markdown() + "\n")

    # 按家庭规模
    lines.append("## 按家庭规模\n")
    family_stat = df.groupby("FamilySize")["survived_int"].agg(["count", "mean"])
    family_stat.columns = ["人数", "生存率"]
    family_stat["生存率"] = family_stat["生存率"].map(lambda x: f"{x:.2%}")
    lines.append(family_stat.to_markdown() + "\n")

    # 按年龄组
    lines.append("## 按年龄组\n")
    age_stat = df.groupby("AgeGroup", observed=True)["survived_int"].agg(["count", "mean"])
    age_stat.columns = ["人数", "生存率"]
    age_stat["生存率"] = age_stat["生存率"].map(lambda x: f"{x:.2%}")
    lines.append(age_stat.to_markdown() + "\n")

    # 按票价组
    lines.append("## 按票价组\n")
    fare_stat = df.groupby("FareGroup", observed=True)["survived_int"].agg(["count", "mean"])
    fare_stat.columns = ["人数", "生存率"]
    fare_stat["生存率"] = fare_stat["生存率"].map(lambda x: f"{x:.2%}")
    lines.append(fare_stat.to_markdown() + "\n")

    # 交叉表：性别 + 舱位
    lines.append("## 性别 × 舱位 生存率\n")
    pivot = df.pivot_table(
        index="sex", columns="pclass", values="survived_int", aggfunc="mean"
    )
    pivot = pivot.map(lambda x: f"{x:.2%}")
    lines.append(pivot.to_markdown() + "\n")

    # 写入文件
    with open("statistics.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("统计结果已保存到 statistics.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()