# PEST++参数评估和不确定分析

具体操作过程和文中的图片，请参考"PEST++参数评估和不确定分析.pdf"

历史匹配方法的细节参考Corson-Dosch
(2022)，实施历史匹配，为嵌套模型和LGR模型细化参数评估，使用与区域模型相同的观测数据。

嵌套模型实施PEST++ (ver.5.0)的iES (White et al.,
2021)，定量参数评估的不确定性，在可接受的测量值范围内复演模拟的水头和河流径流。

1.  Corson-Dosch, N.T., Fienen, M.N., Finkelstein, J.S., Leaf, A.T.,
    White, J.T., Woda, J., and Williams, J.H., 2022, Areas contributing
    recharge to priority wells in valley-fill aquifers in the Neversink
    River and Rondout Creek drainage basins, New York: U.S. Geological
    Survey Scientific Investigations Report 2021--5112, 50 p., accessed
    August 18, 2022, at https://doi.org/ 10.3133/ sir20215112.

2.  White, J.T., 2018, A model-independent iterative ensemble smoother
    for efficient history-matching and uncertainty quantification in
    very high dimensions: Environmental Modelling & Software, v. 109, p.
    191--201. \[Also available at https://doi.org/ 10.1016/ j .envsoft.2
    018.06.009.\]

3.  White, J.T., Hunt, R.J., Fienen, M.N., and Doherty, J.E., 2021,
    Approaches to highly parameterized inversion---PEST++ version 5, a
    software suite for parameter estimation, uncer­tainty analysis,
    management optimization and sensitivity analysis: U.S. Geological
    Survey Techniques and Methods 7C26, 52 p., https://doi.org/ 10.3133/
    tm7C26.

4.  White, J.T., Foster, L.K., Fienen, M.N., Knowling, M.J.,
    Hemmings, B. and Winterle, J.R., 2020, Toward repro­ducible
    environmental modeling for decision support---A worked example:
    Frontiers in Earth Science, v. 8, 11 p.

## 迭代系综光滑方法和不确定度定量化

iES是一种系综方法，意思是在分析的每个阶段，生成参数集的系综（或realization），与内在的不确定度和观测数据的假设不确定度一致。然后，使用从该系综得到的各参数集实施模拟，产生模型输出的一个范围。iES使用参数和观测系综之间的经验关系，迭代降低不确定度以及模拟和实测之间差异，提供一个反映参数的内在不确定度的后验参数系综。一次\"base-case\"实现表征最小的误差方差解，当模拟时需要一套参数值时使用。使用该\"base-case\"实现用于场景测试。

## 观测数据

表2和表3罗列了使用的观测数据和观测权重（Pleasant Lake and Plainfield
Tunnel Channel Lake）

观测水位和径流在月末时分配值。

水位观测的位置（2个湖泊）见图21和图22.
标记对应观测名称，用于PEST++输入文件。

## 参数化

嵌套模型的参数化策略是对大多数参数实施乘子，到可能的取值范围。

嵌套模型的初始参数值可得到从区域模型历史匹配的模拟结果。

表4和表5罗列了2个湖泊的参数化。

## 历史匹配结果

系综方法得到多次迭代的目标函数值的系综（Pleasant湖的嵌套模型见图23）。

蓝色线表示基础系综实现，青灰色表示所有其他系综实现的轨迹（图23）
