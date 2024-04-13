# ESM as a puzzle: making CliMA accessible piece by piece

https://clima.caltech.edu/2023/09/28/esm-as-a-puzzle-making-clima-accessible-piece-by-piece/

BY ALEXANDRE A. RENCHON, KATHERINE DECK, RENATO BRAGHIERE:

In the realm of scientific advancement, enhancing Earth System Models (ESMs) stands out as a paramount objective. Presently, however, these models remain enigmatic enclaves for many researchers, akin to inscrutable black boxes. The labyrinthine nature of ESMs, coupled with their high computational demands, usage of esoteric programming languages, and the absence of lucid documentation and user interfaces, contribute to this opacity.

To surmount these obstacles, CliMA is creating a new era of accessible ESM features for the global scientific community:

Modernized Programming: Departing from convention, CliMA adopts a contemporary programming language, Julia. This strategic choice facilitates a more seamless and intuitive engagement with the model’s underlying mechanics.

Modular Architecture: Each distinct component within CliMA can operate autonomously as a standalone model or seamlessly integrate into a more expansive model, thereby enabling intricate feedback mechanisms and allowing the building of multiple model configurations, with pieces fitting together like in a jigsaw puzzle.

Decipherable Complexity: CliMA is unraveling complexity by partitioning the intricate framework into discrete, comprehensible units, which are extensively tested and validated independently before being joined. Moreover, the accompanying web-based documentation explains each element, elucidating diagnostic and prognostic equations. These explanations encompass vital attributes such as parameters, constants, scientific units, and typical ranges. To further enhance clarity, dynamic interactive visualizations illustrate the sensitivity of outcomes to various parameters, fostering an enhanced grasp of the model’s behavior (github.com/CliMA/ParamViz.jl). Additionally, the model’s ability to reproduce and ingest empirical observations is receiving the attention it deserves.

User-Friendly Application: CliMA is propelling accessibility to greater heights through user-oriented web applications. For example, we are currently prototyping platforms that will allow users to execute the land model across myriad FLUXNET sites. By enabling users to adjust parameters, domain specifications, and timestepper settings, the application caters to individual research needs. Upon configuration, a simple “run” command precipitates the generation of insightful data-versus-model output visuals. Furthermore, the application enables users to download resulting output files, encapsulating the synergy of user-friendliness and substantive scientific output.

In essence, CliMA is shifting the paradigm of  Earth System Modeling from exclusivity to inclusivity. By embracing contemporary programming, modular design, pedagogically enriched documentation, and user-centric applications, CliMA transcends the barriers that have historically inhibited progress in this domain.


# Nature 编程工具推荐：兼顾语法与速度的 Julia 语言

在涉及气候建模方面，对计算的要求可谓是分秒必争。这些模型旨在解释大气、陆地、太阳和海洋以及连接它们的复杂物理因素，代码达到百万行并在最强大的计算机上执行。因此，当气候模型联盟( Climate Modeling Alliance，CliMA ): 一个由美国科学家，工程师和数学家组成的联盟，从头开始设计能够满足他们需求的用于对气候建模的高性能计算的代码，他们选择了 Julia 。

# Nature｜机器学习和物理模型的「双向奔赴」，3种AI气候建模方法
https://baijiahao.baidu.com/s?id=1794826976150549411&wfr=spider&for=pc

两全其美 (https://www.nature.com/articles/d41586-024-00780-8)

Schneider 说，第三种方法是将机器学习组件嵌入基于物理的模型中以生成混合模型——这是一种妥协。

「我们发现机器学习模型在模拟某些现象方面，比物理模型更有效。」Schneider 说。Watson-Parris 同意这一观点。

到 2024 年年底，Schneider 团队希望完成一个可以与大气和陆地模型耦合的海洋混合模型，作为 Climate Modeling Alliance（CliMA）项目的一部分。

Schneider 表示，最终目标是创建地球系统的数字模型，由人工智能提供支持，可以以闪电般的速度模拟天气和气候，精确度可达公里级。
我们还没有达到这个目标，但研究人员表示这个目标现在就在眼前。