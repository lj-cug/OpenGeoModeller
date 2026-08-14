# CLAUDE.md — OpenGeoModeller 文档整理备忘

供后续 Agent / 协作者接续工作时使用。记录 2026-08-14 起对仓库 Markdown 的规范化过程、约定与落盘位置。

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

---

## 4. 编码注意（Windows）

- 部分工具写入可能导致 **GBK** 落盘；应用 UTF-8（无 BOM）校验中文关键词（如「相关文档」「子目录」）。
- 若出现乱码：用编码 936 读出再以 UTF-8 写回；改完后用 PowerShell/`git diff` 确认 CJK 正常。
- Cursor `Read` 工具偶发对个别 CRLF+UTF-8 文件显示异常；以磁盘 UTF-8 字节与终端输出为准。

---

## 5. 建议后续（未做或未做完）

1. **模型级补链**（见学科关系图 §6）：ESM↔WRF/ROMS/CROCO；SCHISM↔OCSMesh/ACE；spyro↔Firedrake；GSFLOW/pywatershed↔MODFLOW6；porousMultiphaseFoam↔OpenFOAM；RegESM↔Catalyst 等。
2. **失效相对链接**：约数十条历史路径需按现存树校验。
3. **更多二级 README**：统一 `## 相关文档` 与单 H1；仍避免批量改 `.md` 转换稿正文。
4. **`hpc-base` 内** `*.docx.md` 与 README 规范：在子模块仓库内单独处理并更新 submodule 指针。
5. **提交**：用户要求时再整理 commit（注意 `hpc-base` 子模块与父仓库分开提交）。

---

## 6. 明确不要做的事

- 不借「整理格式」改写公式、安装步骤对错或删文档。
- 不强制统一中英混排文风。
- 不批量 rename 正文文件名（仅入口 `README.md` 与已完成的 `.docx.md`→`.md`）。
- 未获要求时不 `commit` / `push` / 改 git config。

---

## 7. 快速检查命令（PowerShell 示例）

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
```
