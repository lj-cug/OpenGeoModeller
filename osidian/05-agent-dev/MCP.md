---
tags: [agent-dev模块, MCP, OpenGeoModeller]
aliases: [MCP, MCP Server]
repo_path: agent-dev/MCP/
layer: MCP 工具
created: 2026-08-16
---

# MCP

把编译器、求解器、网格、GPU、IDE、可视化暴露给 Agent。含脚手架入门与科学栈服务器（PETSc / Gmsh / Fortran / CUDA / GDB / MATLAB / CAD / Qt / VS / OpenFOAM）。

## 所属分层

MCP 工具（见 [[agent-dev模块关系]]）

## 相关模块

[[Claude_Usage]] · [[RAG]] · [[Skills]] · [[主题-PETSc-AmgX-Agent]] · [[主题-Fortran-Agent]]

## 上级

- [[agent-dev]]

## 科学栈桥接

- [[Linear_Solver]]（PETSc MCP）
- [[CUDA]]（GPU MCP）
- [[Meshing]]（Gmsh MCP）
- [[Turbulence]] · [[主题-OpenFOAM族]]（Foam-Agent）
- [[VirtualReality]]（Blender / ParaView MCP）

## 主要子文档

- `MCP-快速编写.md` · `后台运行python脚本.md`
- `petsc_mcp_server_Install.md` · `GPU_MCP_server.md`
- `Gmsh_MCP.md` · `OpenFOAM_Agent.md`
- `claude-desktop-fortran-mcp.md`
- `GDB_MCP_Server.md` · `GDB调试程序.md`
- `Blender_MCP.md` · `CAD_MCP_Server.md`
- `MATLAB_MCP_Server.md` · `Qt_MCP.md` · `VIsual Studio_MCP.md`

## 仓库路径

`agent-dev/MCP/`

## 索引

- [[MOC-agent-dev]] · [[MOC-OpenGeoModeller]] · [[agent-dev模块关系]]
