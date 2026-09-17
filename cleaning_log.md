# 数据清洗日志：Titanic

## 1. 重复行
- 删除重复行 X 行。

## 2. 缺失值处理
| 列名 | 缺失比例 | 处理方式 | 理由 |
|------|----------|----------|------|
| age | ~20% | 按 pclass+sex 分组中位数填充，并添加 age_missing 指示 | 年龄重要，分组填充更合理 |
| embarked | ~0.2% | 众数填充 | 缺失极少，众数安全 |
| embark_town | ~0.2% | 众数填充 | 同上 |
| deck | >70% | 转为 has_deck 二值特征，删除原列 | 缺失太多，填充会引入偏差 |

## 3. 冗余列删除
- 删除 class, who, adult_male, alive, embark_town，理由：与已有列重复或可推导。

## 4. 类型转换
- survived, pclass, sex, embarked, alone 转为 category。

## 5. 异常值处理
- fare：IQR 方法检测，选择截断到上下界。
- age：检查 <0 或 >100，无异常。

## 6. 输出
- 清洗后数据：titanic_clean.csv
- 形状：...