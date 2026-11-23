# RQ1 / RQ2 / RQ3：中英文 LaTeX

两版共用 2026-11-23 冻结的实验数据。中文版用于组内理解，按“问题与目的 → 实验设计 → 实验结果 → 结果分析 → 结论”组织；英文版是研究论文稿，按 introduction、background/research questions、method、protocol、results、discussion/limitations、conclusion 组织。

英文版参考 Wang et al., *Equivariant Diffusion Policy*（CoRL 2024，本地 `wangdian.pdf`）的研究叙述方式，但不是其文字的改写，也不是已套用官方会议模板的投稿终稿。当前使用标准 `article` 排版，便于移植。作者栏为占位信息，投稿前需要填写真实姓名、单位，并按目标会议要求调整篇幅和模板。

## 文件选择

| 目的 | 文件 |
|---|---|
| 直接阅读中文 PDF | [main_zh.pdf](main_zh.pdf) |
| 直接阅读英文 PDF | [main_en.pdf](main_en.pdf) |
| 独立编译中文 | [main_zh.tex](main_zh.tex) |
| 独立编译英文 | [main_en.tex](main_en.tex) |
| 将中文正文复制到已有模板 | [body_zh.tex](body_zh.tex) |
| 将英文正文复制到已有模板 | [body_en.tex](body_en.tex) |
| 合并参考文献 | [references.bib](references.bib) |
| 图表文件 | `figures/` 中的三个 PDF |
| 一次上传 Overleaf | 上一级目录的 [latex_bilingual.zip](../latex_bilingual.zip) |

正文文件没有 `\documentclass`、宏包导入、作者信息或 `\begin{document}`，可以直接复制。它们包含 abstract、各节正文、表格和图片引用；正文中的表格是完整 LaTeX，不依赖外部数据宏或训练环境。

## 接到已有 LaTeX 模板

1. 复制需要的 `body_en.tex` 或 `body_zh.tex`；也可以只复制其中某个 section。
2. 将 `figures/` 放在主 `.tex` 文件所在目录旁。
3. 将 `references.bib` 的条目合并到已有 `.bib`。若你的模板已有同名 citation key，请统一引用名称。
4. 在原有导言区补齐下面尚未加载的宏包；已有的宏包无需重复加载。

```latex
\usepackage{amsmath,amssymb,booktabs,graphicx}
\graphicspath{{figures/}}
```

中文模板需要支持 UTF-8 中文，例如使用 `ctexart` 与 XeLaTeX；不要把第二个 `\documentclass` 粘到现有文档中。两种语言的正文建议分别编译；不应原样同时插入同一篇文档形成两个 abstract。

将正文插在 `\begin{document}` 之后合适的位置：

```latex
\input{body_en} % 或 body_zh
```

如果模板已经有 abstract，请把正文文件中的 abstract 内容合并到模板原来的摘要位置。保留目标会议原来的参考文献样式；只需合并 `.bib`，不要重复添加 bibliography。独立示例使用 `plain`，不代表 CoRL/ICRA/RSS 的官方要求。

所有标签以 `eqstudy:` 开头，中文标签额外含 `zh:`，以降低与已有模板冲突的可能。英文版的贡献是实证适配与边界研究，没有把本组实现写成原 EquiDiff 的理论或声称全面复现原论文。

## 独立编译 / Overleaf

上传整个 ZIP 后，选择 `main_en.tex` 或 `main_zh.tex` 为 Main document，编译器选择 **XeLaTeX**。中文使用 TeX 发行版自带的 Fandol 字体，不依赖本机的宋体安装。

本地已有 TeX Live 时，在本目录运行：

```bash
latexmk -xelatex main_en.tex
latexmk -xelatex main_zh.tex
```

没有 `latexmk` 时，分别使用 `xelatex`、`bibtex`、`xelatex`、`xelatex`。本项目另用独立安装在忽略目录中的 Tectonic 实际编译检查，未改训练环境或系统 TeX 配置。编译日志和中间文件不纳入 Git；可读 PDF 与源文件纳入 Git。

## 数据与表述边界

- 数值来源：[summary.json](../summary.json) 和 [完整报告](../REPORT_ZH.md)。两版使用相同数据。
- `coverage` 是物体与目标交集面积除以目标面积，不是 IoU，也不是 success rate。
- 主表是 16k/18k/20k 三个 EMA checkpoint 的测试均值；20k/40k/60k 收敛曲线是独立验证集研究。
- H1 未获支持；H2 的方向性动机有条件地获支持；H3 中失效部分获支持，完全恢复部分未获支持。
- 原 proposal 明确写的是 **Optional E4**。两版均解释了它的可选身份，没有使用未完成的 E4 迁移结果。
- `main_*.pdf` 是这份冻结报告的版本，不随后台训练自动变化。
