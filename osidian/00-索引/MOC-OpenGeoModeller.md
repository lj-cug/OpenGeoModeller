---



tags: [MOC, 索引, OpenGeoModeller]



aliases: [OpenGeoModeller知识图谱, 学科图谱入口]



---



# MOC · OpenGeoModeller



基于仓库根 `README.md`（仓库） 与 `docs/学科关系图.md`（仓库） 建立的 Obsidian 知识图谱入口。



## 如何可视化



1. 打开本笔记后，用左侧 **关系图谱**（Graph view）查看全局双向链接。



2. 打开 [[00-索引/OpenGeoModeller-知识图谱.canvas|OpenGeoModeller-知识图谱]] Canvas，查看分层布局的知识图谱。



3. 图谱按标签着色：枢纽 / 学科 / 主题簇 / hpc-base模块 / agent-dev模块。

4. hpc-base 分层 Canvas：[[04-hpc-base/hpc-base-知识图谱.canvas|hpc-base-知识图谱]]。

5. agent-dev 分层 Canvas：[[05-agent-dev/agent-dev-知识图谱.canvas|agent-dev-知识图谱]]。



## 枢纽（4）



- [[hpc-base]] — 算力基座



- [[Meshing]] — 前处理



- [[VirtualReality]] — 后处理



- [[ESM-Coupler]] — 耦合



## 学科



- [[Meteorology]] · [[Hydrology]] · [[SurfaceWater]] · [[ShallowWater]]



- [[Underground]] · [[GroundWater]]



- [[Geophysics]] · [[Geological-Modelling]]



- [[Turbulence]] · [[Hemodynamics]]



## 主题簇（内容耦合）



- [[主题-ESM大气海洋]]



- [[主题-水文与地下水]]



- [[主题-网格与SCHISM族]]



- [[主题-Firedrake与FWI]]



- [[主题-OpenFOAM族]]



- [[主题-原位可视化]]





## hpc-base 模块图谱



- 入口：[[MOC-hpc-base]]

- 关系说明：[[hpc-base模块关系]]

- Canvas：[[04-hpc-base/hpc-base-知识图谱.canvas|hpc-base-知识图谱]]

- 模块：[[Cluster-DIY]] · [[Build]] · [[Language]] · [[CUDA]] · [[OpenCL]] · [[MPI]] · [[Linear_Solver]] · [[HPC-IO]] · [[DSL]] · [[HPX]] · [[garbage]]



图谱过滤可用：`tag:#hpc-base模块`



## agent-dev 模块图谱



- 入口：[[MOC-agent-dev]]

- 总览：[[agent-dev]]

- 关系说明：[[agent-dev模块关系]]

- Canvas：[[05-agent-dev/agent-dev-知识图谱.canvas|agent-dev-知识图谱]]

- 模块：[[Claude_Usage]] · [[CodeX_Cursor]] · [[LSP]] · [[CLAUDE-md]] · [[Skills]] · [[Loop]] · [[MCP]] · [[RAG]] · [[LangChain]] · [[Osidian]]

- 主题：[[主题-LLM路由]] · [[主题-PETSc-AmgX-Agent]] · [[主题-Fortran-Agent]]



图谱过滤可用：`tag:#agent-dev模块`



## 说明文档



- [[学科关系总览]]



- [[知识图谱使用说明]]



- 仓库：`docs/CLAUDE.md`（仓库） · `docs/学科关系图.md`（仓库）



