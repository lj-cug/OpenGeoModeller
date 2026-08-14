# 非 hpc-base Markdown 规范与索引整理（方案 B）

日期：2026-08-14  
范围：仓库内除 `hpc-base/` 外的学科文档（约 429 个 `.md`）  
目标：统一格式 + 补全索引与交叉链接；**不重写技术正文**。  
约定：与 `2026-08-14-hpc-base-markdown-design.md` 同一套全局格式。

---

## 1. 背景与问题

| 问题 | 量级 / 表现 |
|------|-------------|
| 多 H1 | 约 127 个文件含多个 `#` 标题 |
| 代码块不规范 | 约 337 个文件无 ``` 围栏 |
| 索引缺链接 | 根与多数学科 README 用编号散文列组件，几乎无相对链接 |
| README 大小写 | `ReadMe.md` / `ReadME.md` / `Readme.md` / `READEM.md` 混用 |
| 转换稿 | 约 53 个 `*.docx.md`，内容可用但格式噪声大 |

方案 B = **格式规范 + 关系图/目录写入 README**，不合并删改正文、不深度改写 `.docx.md`。

---

## 2. 格式约定（全局，与 hpc-base 一致）

1. **标题**：每个文件恰好一个 H1；子节用 H2/H3；禁止用第二个 H1 分段。
2. **代码**：命令与代码块使用围栏，并标注语言（优先 `bash` / `shell` / `python` / `cmake` / `text`）。
3. **列表与链接**：目录索引用 Markdown 列表 + 相对路径链接，例如 `- [SCHISM](./SCHISM/)`；禁止仅用 Tab 缩进散文充当目录。
4. **入口文件名**：目录入口统一为 `README.md`（将 `readme.md` / `ReadMe.md` / `ReadME.md` / `READEM.md` 等规范为 `README.md`）。
5. **相关文档**：学科级 README 增加 `## 相关文档` 小节，列出 2–6 条跨模块相对链接（常含 `../hpc-base/`、`../Meshing/` 等）。
6. **`.docx.md`**：不改写正文；仅在上层 README 中可标注“参考/转换稿”并给出链接。
7. **不改**：安装脚本、源码、二进制、图片、非 `.md` 文件；不删除文档；不统一全文中英混排。

---

## 3. 仓库模块关系（学科层）

```text
                         ┌──────────────┐
                         │  Meteorology │  NWP / WRF / RegCM
                         └──────┬───────┘
                                │ 大气强迫
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
   ┌───────────┐         ┌────────────┐          ┌────────────┐
   │Hydrology  │         │SurfaceWater│          │ShallowWater│
   │分布式水文 │         │河流/海洋   │          │浅水/洪水   │
   └─────┬─────┘         └──────┬─────┘          └────────────┘
         │                      │
         ▼                      ▼
   ┌───────────┐         ┌────────────┐     ┌─────────────────┐
   │Underground│◄────────│  Meshing   │────►│Geological-Model.│
   │地下/多孔  │  网格   │网格前处理  │     │地震解释/隐式建模│
   └───────────┘         └────────────┘     └────────┬────────┘
         ▲                      │                    │
         │                      ▼                    ▼
   ┌───────────┐         ┌────────────┐     ┌─────────────────┐
   │Hemodynamics│        │Turbulence  │     │   Geophysics    │
   │血液动力学  │        │CFD/湍流    │     │正演/反演/FWI    │
   └───────────┘         └────────────┘     └─────────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
   ┌───────────┐         ┌────────────┐          ┌────────────┐
   │ESM-Coupler│         │VirtualReal.│          │  hpc-base  │
   │耦合器/ESM │         │可视化/原位 │          │编译/MPI/GPU│
   └───────────┘         └────────────┘          └────────────┘
```

---

## 4. 实施批次

### 批次 1 — 导航骨架（必做）

1. 根 `README.md`：模块列表改为相对链接；单 H1；裸 URL 改为链接。
2. 学科入口 README（14 个）：`SurfaceWater` / `Hydrology` / `Geophysics` / `Meshing` / `Underground` / `ESM-Coupler` / `VirtualReality` / `Turbulence` / `Meteorology` / `ShallowWater` / `Geological-Modelling` / `Hemodynamics` / `GroundWater` + 根。
3. 每个学科 README：子目录列表带相对链接 + `## 相关文档`；多余 H1 降为 H2。

### 批次 2 — 入口文件名（必做）

- `Hydrology/SHUD/READEM.md` → `README.md`
- `Turbulence/GOTM/ReadMe.md` → `README.md`
- `VirtualReality/Computer_Graphics/OpenGL/ReadME.md` → `README.md`
- `VirtualReality/In-situ-Visualization/{ascent,isaac}/Readme.md` → `README.md`

### 批次 3 — 二级模块 README（建议）

已有 README 的模型目录：补 `install/` / `doc/` 链接与向上 `## 相关文档`；触及即修多 H1 / 裸命令。

### 批次 4 — 正文触及即修（可选）

不批量重写 400+ 正文；`.docx.md` 不批量美化。

---

## 5. 明确不做

- 不改写科学/安装步骤的技术对错
- 不删除 `.docx.md` 或空目录说明
- 不批量 rename 正文文件名（仅规范目录入口 `README.md`）
- 不修改 `hpc-base/`（已单独规范）

---

## 6. 验收标准

1. 打开根 `README.md` 可一键跳到全部学科目录。
2. 每个学科 README：有相对链接子目录 + `## 相关文档`（至少 2 条真实可点链接，`GroundWater` 可例外说明）。
3. 批次 1 文件：单 H1、目录为列表链接。
4. Git diff 以 README/索引为主，正文无大规模无关重排。
