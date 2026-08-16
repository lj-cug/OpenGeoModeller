---
tags: [MOC, agent-dev模块, OpenGeoModeller]
aliases: [agent-dev知识图谱入口]
created: 2026-08-16
---

# MOC · agent-dev

Agent Coding 备忘图谱。上级总览：[[agent-dev]]。总库入口：[[MOC-OpenGeoModeller]]。

仓库入口：`agent-dev/README.md`（仅一行说明，子目录无跨文件夹相对链接）。本库用 wikilink 补上内容耦合边。

## 如何看图

1. 关系图谱过滤：`tag:#agent-dev模块`
2. Canvas：[[05-agent-dev/agent-dev-知识图谱.canvas|agent-dev-知识图谱]]
3. 文字关系：[[agent-dev模块关系]]

## 按分层

### 编辑器 / 运行时

- [[Claude_Usage]] · [[CodeX_Cursor]] · [[LSP]]

### 项目记忆 / 规范

- [[CLAUDE-md]]

### Skills / 循环

- [[Skills]] · [[Loop]]

### MCP 工具

- [[MCP]]

### 知识管线

- [[RAG]] · [[LangChain]]

### 知识库

- [[Osidian]]

## 主题簇（内容耦合）

- [[主题-LLM路由]]
- [[主题-PETSc-AmgX-Agent]]
- [[主题-Fortran-Agent]]

## 连向科学栈（桥接）

- [[MCP]] / [[RAG]] / [[LangChain]] → [[Linear_Solver]] · [[CUDA]]
- [[MCP]] → [[Meshing]] · [[VirtualReality]] · [[Turbulence]]
- [[Skills]] / [[CLAUDE-md]] / [[LSP]] → [[Language]] · [[hpc-base]]
- [[Osidian]] → [[MOC-OpenGeoModeller]] · [[知识图谱使用说明]]

## 仓库对照

- `agent-dev/`
- `agent-dev/README.md`
