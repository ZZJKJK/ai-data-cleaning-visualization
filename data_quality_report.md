# 数据质量报告：Titanic

> 数据源：`titanic_raw.csv`（由 `load_data.py` 从 seaborn 内置数据集导出）
> 探索脚本：`explore.py` ｜ 生成日期：2026-09-16
> 环境：Python 3.14.6 / pandas 3.0.3 / numpy 2.4.6

## 1. 数据集概况
- 行数：891
- 列数：15
- 主要列：survived、pclass、sex、age、sibsp、parch、fare、embarked、class、who、adult_male、deck、embark_town、alive、alone
- 文件大小：57,910 字节；无完全空列；各列唯一值个数在 2~248 之间（fare 最多，248 个）
- 需要留意：其中 class、who、adult_male、alive、alone、embark_town 这 6 列都是由其它列派生出来的冗余列（详见第 4、7 节），`alive` 与目标列 `survived` 完全等价

## 2. 缺失值
| 列名 | 缺失数 | 缺失比例 | 处理建议 |
|------|--------|----------|----------|
| deck | 688 | 77.22% | 不填补。缺失与舱位强相关（3 等舱缺失 97.56%、2 等舱 91.30%、1 等舱 18.98%），说明它只在部分乘客登记了舱位时才有值。建议转成「是否有记录」的 0/1 指示列（deck_recorded），或直接删列 |
| age | 177 | 19.87% | 按 sex × pclass 分组中位数填补；更稳妥的做法是同时新增 age_missing 指示列，把缺失本身当作信息保留。不要用全局均值 |
| embarked | 2 | 0.22% | 用众数 'S' 填补（这 2 行是 1 等舱女性、fare=80、deck=B，也可据此推断为 'C'）。仅 2 行，对结果影响可忽略 |
| embark_town | 2 | 0.22% | 与 embarked 同源，填补后必须让两列映射保持一致（S=Southampton、C=Cherbourg、Q=Queenstown） |

其余 11 列无缺失。全表缺失单元 869 / 13,365 ≈ 6.5%。

缺失并非完全随机：age 缺失率在 3 等舱为 27.7%、1 等舱 13.9%、2 等舱 6.0%，与舱位相关，因此填补要分舱位进行。

## 3. 重复值
- 重复行数：107（完全相同的整行；涉及 53 个重复组、160 行，最大的组有 13 行）
- 是否删除：**保留，不删除（已确认）**，理由如下
  - 本数据集是 seaborn 从 Kaggle train.csv 派生的，已去掉 name、ticket、cabin 等身份列，不同乘客在剩下 15 列上完全可能一模一样，这些「重复」是特征撞车而不是重复录入
  - 160 行重复行中有 93 行（58.1%）age 缺失、158 行 deck 缺失，大量 NaN 相同会直接构成「重复」
  - 实测：剔除 age 缺失的行后全列重复从 107 降到 36；再剔除 deck 缺失，只剩 1 行——说明重复主要由缺失造成
  - 结论：直接 `drop_duplicates()` 会误删真实乘客。若需要，可另存一份去重版本做敏感性对比

## 4. 数据类型
- 数值列：survived(int)、pclass(int)、sibsp(int)、parch(int)、age(float)、fare(float)
- 类别列：sex、embarked、class、who、deck、embark_town（pandas 3.x 中 dtype 显示为 str）；布尔列：adult_male、alone
- 需要转换的列：
  - survived、pclass → category（或建模时 one-hot）
  - sex、embarked、class、deck（若保留）→ category
  - adult_male、alone、alive → category；注意 `alive` 是 `survived` 的直接映射，属标签泄漏，必须排除
  - age、fare 保持 float；fare 需 log1p 变换处理长尾

## 5. 异常值初步
IQR 方法（1.5 × IQR）结果：

| 列 | Q1 | Q3 | IQR | 下界 | 上界 | 低于下界 | 高于上界 |
|----|----|----|-----|------|------|----------|----------|
| survived | 0.000 | 1.000 | 1.000 | -1.500 | 2.500 | 0 | 0 |
| pclass | 2.000 | 3.000 | 1.000 | 0.500 | 4.500 | 0 | 0 |
| age | 20.125 | 38.000 | 17.875 | -6.688 | 64.812 | 0 | 11 |
| sibsp | 0.000 | 1.000 | 1.000 | -1.500 | 2.500 | 0 | 46 |
| parch | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 213 |
| fare | 7.910 | 31.000 | 23.090 | -26.724 | 65.634 | 0 | 116 |

- age：11 个值超过上界 64.81，最大 80 岁；最小值 0.42 岁，另有 7 个不足 1 岁的婴儿。这些是真实的乘客年龄分布，不是录入错误，建议保留，需要时可分箱
- fare：116 个值超过上界 65.63，最大值 512.33，偏度 4.79、峰度 33.4，长尾非常严重；另有 15 行 fare = 0。建议做 log1p 变换或分箱，并给 fare = 0 单独加一个指示列
- 其他：
  - sibsp 的 46 个、parch 的 213 个「异常值」是 IQR 方法失效造成的假象：这两列 Q1 = Q3 = 0、IQR = 0，上界被算成 0，于是所有大于 0 的值都被判为异常。它们并非数据错误，建议改用业务分箱（0 / 1–2 / 3+）
  - pclass、survived 无异常值
  - 取值范围全部合法：无负值；pclass ∈ {1,2,3}；survived ∈ {0,1}；sibsp ∈ [0,8]；parch ∈ [0,6]；age ∈ [0.42, 80]；fare ∈ [0, 512.33]
  - 箱线图见 figures/boxplot_age_fare.png

## 6. 类别一致性
- sex：只有 male / female 两个取值，无大小写不一致、无多余空格
- embarked：S / C / Q 三个取值 + 2 个缺失；与 embark_town 的映射 0 处冲突，两列完全自洽
- class：First / Second / Third，与 pclass 的映射 0 处冲突
- 另两列派生关系也完全自洽：alive 与 survived 0 处冲突；alone 与 (sibsp + parch == 0) 0 处冲突
- who：man / woman / child；child 的最大年龄为 15，即该列用 age ≤ 15 定义儿童
- adult_male：True 537 行，而 sex = 'male' 有 577 行，差的 40 行是未成年男性（年龄 0.42~15 岁），说明它只是 sex 与 age 的组合派生列

## 7. 处理建议
- 缺失值：deck 不填补（转 deck_recorded 指示列或删除）；age 按 sex × pclass 分组中位数填补并增加 age_missing 指示列；embarked / embark_town 用众数 'S' 填补，并保持两列映射一致
- 异常值：fare 做 log1p 变换或分箱，fare = 0 单独标记；age 的极端值保留（真实老年乘客），需要时分箱为儿童 / 青年 / 中年 / 老年；sibsp、parch 不要用 IQR，改用 0 / 1–2 / 3+ 分箱
- 类型转换：survived、pclass、sex、embarked、deck 转 category；age、fare 保持数值；建模前对类别列做 one-hot（pclass、deck 也可用有序编码）
- 冗余列：删除 class、who、adult_male、alone、alive，embark_town 与 embarked 只保留其一，避免多重共线性，并彻底排除 alive 造成的标签泄漏
- 重复值：保留 107 条重复行（已确认，本次不生成去重版本）

## 8. 结论
数据整体可用：形状规整（891 × 15）、无完全空列、无非法取值范围、类别取值干净。主要问题集中在三处：一是 deck（77.22%）和 age（19.87%）的缺失，且缺失与舱位相关，填补必须分组进行；二是 fare 的长尾与极值（偏度 4.79、最大值 512.33、15 行为 0）；三是 6 个派生冗余列，其中 alive 与目标列 survived 等价，必须排除以免标签泄漏。

107 条重复行是身份列被删除后的特征撞车，已确认保留而不删除。

按第 7 节处理后，即可进入特征工程与建模。本次探索的图表见 figures/ 目录：missing_matrix.png、missing_bar.png、boxplot_age_fare.png。
