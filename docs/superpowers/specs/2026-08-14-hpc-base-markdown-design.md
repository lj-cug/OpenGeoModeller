# hpc-base Markdown 规范与路径关系梳理（方案 B）

日期：2026-08-14  
范围：`hpc-base/`（约 235 个 `.md`）  
目标：统一格式 + 补全索引与交叉链接 + 产出模块关系说明；**不重写技术正文**。

---

## 1. 背景与问题

`hpc-base` 是 OpenGeoModeller 的高性能计算基座文档与示例集合。现状问题：

| 问题 | 量级 / 表现 |
|------|-------------|
| 多 H1 | 约 79 个文件含多个 `#` 标题，破坏“一文一主题”结构 |
| 代码块不规范 | 仅约 59 个文件使用 ``` 围栏；命令常裸写或用单行反引号 |
| 索引缺链接 | 根与多数子 README 用 Tab 缩进散文描述子目录，几乎无相对链接 |
| README 大小写 | 存在 `readme.md` / `Readme.md` / `README.md` 混用（Windows 不敏感，Linux/Git 敏感） |
| 转换稿 | 约 33 个 `*.docx.md`，内容可用但格式噪声大 |
| 交叉引用弱 | 模块间依赖（如 CUDA↔MPI↔PETSc）在文档层几乎未显式链接 |

方案 B = **格式规范 + 关系图写入 README**，不合并删改正文、不深度改写 `.docx.md`。

---

## 2. 格式约定（全局）

适用于本次会改动的 README / 索引页；正文页按“触及即修、不主动翻修全书”原则。

1. **标题**：每个文件恰好一个 H1；子节用 H2/H3；禁止用第二个 H1 分段。
2. **代码**：命令与代码块使用围栏，并标注语言（优先 `bash` / `shell` / `python` / `cmake` / `makefile`）。
3. **列表与链接**：目录索引用 Markdown 列表 + 相对路径链接，例如 `- [Docker](./Docker/)`；禁止仅用 Tab 缩进散文充当目录。
4. **入口文件名**：目录入口统一为 `README.md`（将 `readme.md` / `Readme.md` 规范为 `README.md`，仅在 Git 可正确识别大小写变更时执行）。
5. **相关文档**：一级模块 README 增加 `## 相关文档` 小节，列出 2–6 条跨模块相对链接。
6. **`.docx.md` 与 `garbage/`**：不改写正文；仅在上层 README 中标注为“参考/归档”，并给出链接。
7. **不改**：安装脚本、源码、二进制、图片、非 `.md` 文件；不删除文档。

---

## 3. 模块关系（内容依赖）

逻辑分层（上依赖下、侧向协作）：

```text
                    ┌─────────────────────┐
                    │    Cluster-DIY      │  硬件/SSH/NFS/Slurm/监控
                    └──────────┬──────────┘
                               │ 运行环境
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
   ┌──────────┐         ┌──────────┐          ┌──────────┐
   │  Build   │         │ Language │          │ garbage  │
   │编译/容器 │         │语言与混编│          │运维备忘  │
   └────┬─────┘         └────┬─────┘          └──────────┘
        │                    │
        ▼                    ▼
   ┌──────────────────────────────────────┐
   │     并行与加速： CUDA | OpenCL | MPI │
   │     MPI-GPUDirect 依赖 CUDA/GPUDirect│
   └──────────────────┬───────────────────┘
                      ▼
   ┌──────────────────────────────────────┐
   │  Linear_Solver：AmgX / AmgCL / PETSc │
   │  / Trilinos / paralution             │
   │  AmgX↔CUDA；AmgCL↔OpenCL；PETSc↔MPI │
   └──────────────────┬───────────────────┘
                      ▼
   ┌──────────────────────────────────────┐
   │  HPC-IO：HDF5(+GDS) / NetCDF / VTK   │
   │  GDS 与 CUDA/GPUDirect 相关          │
   └──────────────────┬───────────────────┘
                      ▼
   ┌──────────────────────────────────────┐
   │  高层抽象： DSL (OP2/Devito/Firedrake)│
   │           HPX（异步运行时）           │
   └──────────────────────────────────────┘
```

### 3.1 各一级目录职责与主要交叉关系

| 目录 | 职责 | 应链接到 |
|------|------|----------|
| `Cluster-DIY/` | 自建集群硬件与配置 | `MPI/`（通信测试）、`garbage/`（系统命令）、`Build/`（节点软件栈） |
| `Build/` | Makefile/CMake/SCons/Meson、编译器、Docker/Singularity/Spack | `Language/`、`CUDA/`（Makefile_CUDA）、`Cluster-DIY/` |
| `Language/` | R/Python/Julia/Java、混编 | `Build/`、`MPI/`（mpi4py 等） |
| `CUDA/` | CUDA 基础、GPUDirect、样例、NVHPC | `MPI/MPI-GPUDirect/`、`HPC-IO/`（GDS）、`OpenCL/`（对比）、`Linear_Solver/AmgX/` |
| `OpenCL/` | 跨厂商 GPGPU | `CUDA/`、`Linear_Solver/AmgCL/` |
| `MPI/` | OpenMPI/MPICH、IB、CUDA-aware MPI | `CUDA/GPUDirect/`、`Cluster-DIY/`、`Linear_Solver/PETSc/` |
| `Linear_Solver/` | AmgX/AmgCL/PETSc/Trilinos/paralution | `CUDA/`、`OpenCL/`、`MPI/`、`Build/` |
| `HPC-IO/` | HDF5/NetCDF/VTK、GDS | `CUDA/GPUDirect/GDS/`、`MPI/` |
| `DSL/` | OP2 / Devito / Firedrake 导读 | `Language/`、仓库内 `SurfaceWater/Firedrake`、`Geophysics`（仅说明性外链，可选） |
| `HPX/` | HPX 编译与应用导读 | `Build/`、`Language/`（C++） |
| `garbage/` | Ubuntu/git/markdown 备忘（归档向） | 根 README 标明“非主线” |

### 3.2 与上级仓库的关系（只说明，不改上级学科文档）

- `hpc-base` 为子模块，被 `SurfaceWater`、`Hydrology`、`Geophysics`、`Turbulence` 等依赖（编译器、MPI、PETSc、CUDA、容器）。
- DSL README 中已提及 Firedrake/Devito：交叉链接可指向父仓库相对路径（若从子模块视角不便，则写绝对 gitee 路径或“见父仓库 …”文字说明）。**默认**：子模块内相对链接只指向 `hpc-base` 内部；父仓库路径用文字 + 可选 `../../SurfaceWater/Firedrake/`（在 monorepo 检出时可用）。

---

## 4. 实施范围与批次

按“先索引、后入口正文、再抽样正文”执行，避免一次改 235 个文件导致不可审。

### 批次 1 — 导航骨架（必做）

改这些入口文件，补目录链接 + `## 相关文档` + 基本格式：

1. `hpc-base/README.md`
2. `Build/README.md`
3. `CUDA/readme.md` → `CUDA/README.md`（若需大小写规范）
4. `MPI/README.md`
5. `Linear_Solver/README.md`
6. `Language/README.md`
7. `Cluster-DIY/README.md`
8. `HPC-IO/README.md`
9. `DSL/README.md`
10. `HPX/README.md`
11. `OpenCL/readme.md` → `OpenCL/README.md`
12. `garbage/README.md`（标注归档，补子目录链接）

另：`Build/Docker/README.md` 当前为裸命令堆叠，纳入批次 1 做围栏与小节整理（内容不删）。

### 批次 2 — 二级关键入口（建议）

对已有 README 的二级目录补子页链接与向上/横向“相关文档”，优先：

- `Build/{Docker,CMake,Makefile,Spack,Singularity,OneAPI,Meson,Scons}/`
- `CUDA/{GPUDirect,Ubuntu-GPU}/`
- `MPI/` 下已有说明页
- `Linear_Solver/{PETSc,AmgX,AmgCL,Trilinos,paralution}/`
- `HPC-IO/` 下 HDF5/NetCDF 若无 README，仅在一级 README 链到现有 `doc/`/`install/`

### 批次 3 — 正文触及即修（可选、有限）

不批量重写 200+ 正文。仅当批次 1–2 编辑中打开的文件存在明显格式问题时顺手修复：

- 裸命令 → 围栏  
- 多余空 H1 → 降为 H2  
- 断裂的相对链接（若发现）

`.docx.md`：**不**批量美化。

### 批次 4 — 关系说明落盘（必做）

在 `hpc-base/README.md` 增加简短「模块关系」小节（可用 mermaid 或缩进文本图），与本文档第 3 节一致、更短。

本设计文档保留在父仓库 `docs/superpowers/specs/`，作为变更依据；不强制把全文复制进子模块。

---

## 5. 明确不做

- 不改写科学/安装步骤的技术对错（除非明显笔误且用户另行要求）
- 不删除 `garbage/` 或 `.docx.md`
- 不统一全文中英混排风格（保留作者原有中文叙述）
- 不批量 rename 正文文件名（仅规范目录入口 `README.md`）
- 不修改父仓库其它学科目录的 Markdown（本轮仅 `hpc-base`）

---

## 6. 验收标准

1. 打开 `hpc-base/README.md` 可一键跳到全部一级模块。
2. 每个一级模块 README：有相对链接目录 + `## 相关文档`（至少 2 条真实可点链接）。
3. 批次 1 文件：单 H1、命令块使用围栏、无 Tab 散文目录。
4. `Build/Docker/README.md` 命令可读、可复制。
5. Git diff 以 README/索引为主，正文无大规模无关重排。

---

## 7. 风险与注意

- **子模块**：`hpc-base` 有独立 git 历史；改动需在子模块内提交，父仓库再更新 submodule 指针（若用户要求提交时再执行）。
- **大小写重命名**：在 Windows 上需 `git mv` 两步（临时名 → `README.md`）才能被 Git 记录。
- **外链失效**：交叉链接只用已存在路径；实施前用目录枚举校验。

---

## 8. 建议实施顺序（批准后）

1. 写 implementation plan（writing-plans）
2. 执行批次 1 → 自检链接
3. 执行批次 2（二级入口）
4. 批次 3 仅限触及文件
5. 更新根 README 关系图
6. 向用户汇报 diff 摘要；**仅在用户要求时** commit
