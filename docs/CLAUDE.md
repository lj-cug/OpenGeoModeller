# CLAUDE.md — OpenGeoModeller 文档整理备忘

供后续 Agent / 协作者接续工作时使用。记录 2026-08-14 起对仓库 Markdown 的规范化、学科/`hpc-base` 关系分析，以及 Obsidian 知识图谱落盘过程。

---

## 1. 目标与原则

- **目标**：统一 Markdown 格式，补全索引与交叉链接，厘清学科文档关系；**不重写**科学/安装技术正文。
- **方案 B**（与 `hpc-base` 一致）：格式规范 + README 导航/关系说明；不合并删改正文；不深度美化 Word 转换稿。
- **范围划分**：
  - `hpc-base/`：独立子模块，有单独规范；父仓库整理时默认**不改**其子模块正文（除非用户明确要求）。
  - 父仓库其余学科目录：按同一套全局约定整理。

---

## 2. 全局格式约定

1. **标题**：每个 `.md` 恰好一个 H1；分段用 H2/H3，禁止第二个 H1。
2. **代码**：命令与代码块用围栏，并标注语言（优先 `bash` / `shell` / `python` / `cmake` / `text`）。
3. **目录索引**：Markdown 列表 + 相对路径链接，例如 `- [SCHISM](./SCHISM/)`；禁止仅用 Tab 缩进散文充当目录。
4. **入口文件名**：目录入口统一为 `README.md`（`readme.md` / `ReadMe.md` / `ReadME.md` / `READEM.md` 等需规范；Windows 上用 `git mv` 两步改大小写）。
5. **相关文档**：学科级（及重要模型级）README 增加 `## 相关文档`，列 2–6 条真实可点的相对链接（常含 `../hpc-base/`、兄弟学科）。
6. **不改**：安装脚本、源码、二进制、图片、非 `.md`；不删除文档；不统一全文中英混排。
7. **提交**：仅在用户明确要求时 `git commit` / `push`。

---

## 3. 已完成工作（父仓库，除 hpc-base）

### 3.1 规范与关系文档

| 文件 | 作用 |
|------|------|
| [superpowers/specs/2026-08-14-hpc-base-markdown-design.md](./superpowers/specs/2026-08-14-hpc-base-markdown-design.md) | `hpc-base` 格式与模块关系方案 |
| [superpowers/specs/2026-08-14-non-hpc-base-markdown-design.md](./superpowers/specs/2026-08-14-non-hpc-base-markdown-design.md) | 父仓库学科 Markdown 方案 B |
| [学科关系图.md](./学科关系图.md) | 工作流分层、显式链接、主题耦合、建议补链清单 |

### 3.2 导航骨架（批次 1）

- 根 [`README.md`](../README.md)：学科列表改为相对链接；「仓库建设内容」末尾链到 [学科关系图](./学科关系图.md)；「相关文档」含关系图与两份规范。
- 学科入口 README（均单 H1 + 子目录链接 + `## 相关文档`）：
  - `SurfaceWater` / `Hydrology` / `Geophysics` / `Meshing` / `Underground`
  - `ESM-Coupler` / `VirtualReality` / `Turbulence` / `Meteorology` / `ShallowWater`
  - `Geological-Modelling` / `Hemodynamics` / `GroundWater`

### 3.3 入口文件名（批次 2）

已 `git mv` 规范为 `README.md`：

- `Hydrology/SHUD/READEM.md`
- `Turbulence/GOTM/ReadMe.md`
- `VirtualReality/Computer_Graphics/OpenGL/ReadME.md`
- `VirtualReality/In-situ-Visualization/ascent/Readme.md`
- `VirtualReality/In-situ-Visualization/isaac/Readme.md`

### 3.4 二级模型 README 抽修（批次 3，部分）

对多 H1、缺 `install`/`doc` 链接的模型入口做了触及即修，例如：SCHISM、Firedrake、telemac、ROMS、NEMO、PIHM、WRF-Hydro、ParFLOW、Devito、JUDI、OpenSWPC、WRFV4、MODFLOW6、OPM、DuMux、ParaView、In-situ-Visualization、OpenFOAM、ESMF、RegESM、Gmsh、OCSMesh、OceanMesh2D 等。

**刻意跳过**：巨型转换稿（如 OASIS 耦合目录下上百 H1 的 README）、嵌套第三方示例 README、批量正文重写。

### 3.5 `*.docx.md` → `*.md`

- 排除 `hpc-base` 后约 **53** 个文件已 `git mv` 去掉 `.docx` 中缀。
- 无同名冲突；仓库内几乎无指向旧 `.docx.md` 路径的业务链接（规范文中仍可能文字提及历史后缀）。

### 3.6 学科关系分析 → 落盘

- 依据：目录结构、README 交叉链接、正文关键词共现。
- 结论要点：枢纽为 `hpc-base`（算力）、`Meshing`（前处理）、`VirtualReality`（后处理）、`ESM-Coupler`（耦合）；内容耦合往往强于显式链接。
- 全文见 [学科关系图.md](./学科关系图.md)；根 README 在「仓库建设内容」后与「相关文档」两处入口。

### 3.7 其它仓库整理（同会话）

- **空目录**：删除无文件空文件夹（含 `Underground/MODFLOW6/{MODPATH7,Central_Sands-Case}` 等），并去掉 README 中失效目录链接。
- **根目录图片** → `docs/`：`Reproducible.jpg`、`Architecture.jpg`、`Geoscience-for-the-Future.jpg`；根 README 图片链接改为 `./docs/...`。

---

## 4. 学科关系分析过程（方法）

用于产出 [学科关系图.md](./学科关系图.md) 与后续 Obsidian 图谱，可复用：

1. **结构统计**：各学科 `.md` 数量、`install/` / `doc/` 目录存在性。
2. **显式链接**：解析 Markdown 相对链接；区分学科内链接 vs `../` 跨学科边；学科 README 的 `## 相关文档` 作为权威边表。
3. **内容耦合**：关键词在「非本学科」目录中的出现次数（如 WRF@ESM、SCHISM@Meshing、PETSc@SurfaceWater/Underground），发现链接滞后于正文点名之处。
4. **三层结论**：工作流依赖 / 显式超链接 / 主题耦合；枢纽角色与建议补链清单写入关系图文档。

---

## 5. Obsidian 知识图谱（vault：`osidian/`）

Vault 路径：`E:\open-geo-modeller\osidian`（Obsidian CLI 库名 `osidian`）。与仓库并列，用 `[[wikilink]]` 镜像 README 关系，**不替代**仓库内 Markdown。

### 5.1 父仓库学科图谱（2026-08-15）

| 路径 | 内容 |
|------|------|
| `00-索引/MOC-OpenGeoModeller.md` | 总 MOC 入口 |
| `00-索引/学科关系总览.md` | Mermaid + 显式边摘要 |
| `00-索引/知识图谱使用说明.md` | Graph / Canvas 用法 |
| `00-索引/OpenGeoModeller-知识图谱.canvas` | 学科分层 Canvas |
| `01-学科/*.md` | 各学科笔记（上游/下游 wikilink） |
| `02-枢纽/` | `hpc-base`、`Meshing`、`VirtualReality`、`ESM-Coupler` |
| `03-主题/` | ESM、水文–地下水、SCHISM 网格、Firedrake–FWI、OpenFOAM、原位可视化 |

约定：

- 标签：`#学科` `#枢纽` `#主题簇` `#MOC` `#OpenGeoModeller`
- Graph 着色见 `osidian/.obsidian/graph.json`
- 指向仓库外路径用纯文本/`code`，避免 unresolved 噪声；Canvas 用 `[[path/file.canvas|显示名]]`

### 5.2 hpc-base 模块关系分析 → 入图

**分析依据**：

- `hpc-base/README.md`（模块关系 mermaid + 依赖速查表）
- [superpowers/specs/2026-08-14-hpc-base-markdown-design.md](./superpowers/specs/2026-08-14-hpc-base-markdown-design.md)
- 各一级模块 README 的 `## 相关文档`（Build/CUDA/MPI/Linear_Solver/Language/Cluster-DIY/HPC-IO/DSL/HPX/OpenCL/garbage）

**分层结论**（上依赖下）：

```text
Cluster-DIY
    → Build / Language / garbage(非主线)
        → CUDA / OpenCL / MPI
            → Linear_Solver / HPC-IO
                → DSL / HPX
```

**关键交叉点**：MPI-GPUDirect↔CUDA/GPUDirect；AmgX↔CUDA；AmgCL↔OpenCL；PETSc↔MPI；HPC-IO/GDS↔GPUDirect Storage；DSL↔父仓 Firedrake/Devito/OP2。

**Obsidian 落盘**（`osidian/04-hpc-base/`）：

| 文件 | 作用 |
|------|------|
| `MOC-hpc-base.md` | hpc-base 子图谱入口 |
| `hpc-base模块关系.md` | 分层 mermaid + 依赖速查 + 父仓桥接 |
| `hpc-base-知识图谱.canvas` | 按分层摆放的模块 Canvas |
| `Cluster-DIY.md` … `garbage.md`（11 个） | 一级模块笔记 + 相关模块 wikilink + 父仓学科桥接 |

同步更新：

- `02-枢纽/hpc-base.md`：展开内部模块表与学科桥接
- `00-索引/MOC-OpenGeoModeller.md`：增加「hpc-base 模块图谱」小节
- 主 Canvas：右侧增加 CUDA/MPI/Linear_Solver/DSL 展开节点
- Graph：新增颜色组 `tag:#hpc-base模块`
- 日笔记可记入当日 daily（如 `2026-08-15.md`）

**父仓桥接（笔记层）**：

- DSL → SurfaceWater / Geophysics / ShallowWater
- Linear_Solver → Underground / Hemodynamics / SurfaceWater
- CUDA → ShallowWater / VirtualReality / Turbulence
- HPC-IO → Meteorology / VirtualReality

### 5.3 Obsidian CLI 常用命令

```powershell
obsidian vault
obsidian reload
obsidian open path="00-索引/MOC-OpenGeoModeller.md"
obsidian open path="04-hpc-base/MOC-hpc-base.md"
obsidian open path="04-hpc-base/hpc-base-知识图谱.canvas" newtab
obsidian command id=graph:open
obsidian unresolved total
obsidian search query="tag:#hpc-base模块" total
```

Windows 上若直接敲 `obsidian` 报「无法识别」，需确保 Obsidian CLI 已在 PATH（或先打开 Obsidian 应用后再用 CLI）。

### 5.4 交互式 HTML 图谱（brain-map skill）

来源：[vladignatyev/brain-map-skill](https://github.com/vladignatyev/brain-map-skill)（MIT）。

- **作用**：把含 YAML frontmatter + `[[wikilinks]]` 的 Markdown 目录打成**单文件**交互 HTML（Cytoscape 力导向图 + 时间轴 + 筛选/搜索 + 节点详情）。
- **主题色**：按 vault **一级子目录**着色（本库即 `00-索引` / `01-学科` / `02-枢纽` / `03-主题` / `04-hpc-base`）。
- **边**：解析并解析成功的 `[[wikilink]]`；节点大小随度数变化。
- **Skill 安装路径**（本机）：`C:\Users\lijian\.agents\skills\brain-map\`（因 GitHub git clone 超时，曾用 jsDelivr 拉取 `SKILL.md` + `scripts/build_map.py`）。
- **对本库生成命令**：

```powershell
python C:\Users\lijian\.agents\skills\brain-map\scripts\build_map.py `
  E:\open-geo-modeller\osidian `
  E:\open-geo-modeller\docs\OpenGeoModeller-brain-map.html `
  --title "OpenGeoModeller Knowledge Map"
start E:\open-geo-modeller\docs\OpenGeoModeller-brain-map.html
```

- **产出**：[`docs/OpenGeoModeller-brain-map.html`](./OpenGeoModeller-brain-map.html)（只读 vault，不写回笔记）。
- 可选依赖：`pip install networkx numpy scipy` 可预计算布局；无依赖时浏览器内 `cose` 布局。

与 Obsidian Graph/Canvas 的关系：Obsidian 适合日常编辑与双向链接；brain-map HTML 适合演示、分享、时间轴回放，无需打开 Obsidian。

---

## 6. 编码注意（Windows）

- 部分工具写入可能导致 **GBK** 落盘；应用 UTF-8（无 BOM）校验中文关键词（如「相关文档」「子目录」）。
- 若出现乱码：用编码 936 读出再以 UTF-8 写回；改完后用 PowerShell/`git diff` 确认 CJK 正常。
- Cursor `Read` 工具偶发对个别 CRLF+UTF-8 文件显示异常；以磁盘 UTF-8 字节与终端输出为准。

---

## 7. 建议后续（未做或未做完）

1. **模型级补链**（见学科关系图 §6）：ESM↔WRF/ROMS/CROCO；SCHISM↔OCSMesh/ACE；spyro↔Firedrake；GSFLOW/pywatershed↔MODFLOW6；porousMultiphaseFoam↔OpenFOAM；RegESM↔Catalyst 等。
2. **失效相对链接**：约数十条历史路径需按现存树校验。
3. **更多二级 README**：统一 `## 相关文档` 与单 H1；仍避免批量改 `.md` 转换稿正文。
4. **`hpc-base` 内** 正文/`*.docx.md` 美化：在子模块内单独处理；Obsidian 一级模块图谱已建，二级（如 PETSc/AmgX/GPUDirect）可按需再拆笔记。
5. **Obsidian**：可将更多模型级 README 链入 vault；保持 unresolved=0。
6. **提交**：用户要求时再整理 commit（注意 `hpc-base` 子模块与父仓库分开提交；`osidian/` 是否入库由用户决定）。

---

## 8. 明确不要做的事

- 不借「整理格式」改写公式、安装步骤对错或删文档。
- 不强制统一中英混排文风。
- 不批量 rename 正文文件名（仅入口 `README.md` 与已完成的 `.docx.md`→`.md`）。
- 未获要求时不 `commit` / `push` / 改 git config。
- 不以 Obsidian 笔记覆盖仓库内权威 README（vault 为镜像/导航层）。

---

## 9. 快速检查命令（PowerShell 示例）

```powershell
# 学科入口是否单 H1、是否含「相关文档」
$roots = 'SurfaceWater','Hydrology','Geophysics','Meshing','Underground','ESM-Coupler','VirtualReality','Turbulence','Meteorology','ShallowWater','Geological-Modelling','Hemodynamics','GroundWater'
foreach ($r in $roots) {
  $c = [IO.File]::ReadAllText("$r\README.md", [Text.Encoding]::UTF8)
  $h1 = ([regex]::Matches($c, '(?m)^#\s+')).Count
  $rel = $c -match '(?m)^##\s*相关文档\s*$'
  Write-Host "$r H1=$h1 related=$rel"
}

# 父仓库是否仍残留 *.docx.md（应排除 hpc-base 后为 0）
Get-ChildItem -Recurse -Filter '*.docx.md' | Where-Object { $_.FullName -notmatch '\\hpc-base\\' }

# hpc-base 一级模块是否含「相关文档」
$hpc = 'Build','CUDA','MPI','Linear_Solver','Language','Cluster-DIY','HPC-IO','DSL','HPX','OpenCL','garbage'
foreach ($m in $hpc) {
  $c = [IO.File]::ReadAllText("hpc-base\$m\README.md", [Text.Encoding]::UTF8)
  Write-Host ("hpc-base/$m related=" + ($c -match '(?m)^##\s*相关文档\s*$'))
}
```
