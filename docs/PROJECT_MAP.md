# 项目目录与 GitHub 同步范围

GitHub 仓库：<https://github.com/KeruZheng/LIU_MU_ZHENG_8001_FINAL>。

**仓库根目录是本地 `wangdian/codes/`，不是整个 `wangdian/`。** 因此，GitHub 首页的 `README.md` 对应 `codes/README.md`，GitHub 的 `tasks/` 对应本地 `codes/tasks/`。

## 先找什么

| 你要看什么 | 仓库内入口 |
|---|---|
| 背景、research gap、三个问题与结论 | [主实验完整报告](../reports/main_results_20261123/REPORT_ZH.md) |
| 中英文 LaTeX 和 PDF | [LaTeX 使用说明](../reports/main_results_20261123/latex/README.md) |
| 三个研究任务如何分工 | [任务总览](../tasks/README.md) |
| RQ1 数据效率与迁移 | [rq1_transfer](../tasks/rq1_transfer/README.md) |
| RQ2 精确增强与 C8 | [rq2_exact_aug](../tasks/rq2_exact_aug/README.md) |
| RQ3 对称性失效与修正 | [rq3_symmetry_breaking](../tasks/rq3_symmetry_breaking/README.md) |
| 冻结科学代码与复现入口 | [experiments](../experiments/README.md) |
| 本地和两个集群的监控机制 | [fleet](../operations/fleet/README.md) |
| 仓库中早期 notes 的性质 | [历史文档说明](notes/README.md) |

## 工作区与仓库的关系

```text
wangdian/                           本地完整工作区；不是这个 Git 仓库的根目录
  03_Proposal_A_EquiDiff_final.md    原 proposal
  wangdian.pdf                      参考论文
  notes/                            本地接管、运行和排障记录
    README.md                       本地 notes 导航
  development/                      实际实验工作区、冻结 release、环境与归档
  presentation/                     本地汇报准备
  codes/                            Git 仓库根目录
    README.md                       GitHub 首页
    docs/PROJECT_MAP.md             当前这份目录说明
    docs/notes/                     部分早期笔记的公开快照
    reports/                        可提交的报告、LaTeX、图表和冻结数据摘要
    tasks/                          按 RQ1 / RQ2 / RQ3 组织的入口
    experiments/releases/           纳入版本管理的冻结科学代码快照
    operations/                     调度、监控和归档工具
```

**并非 `codes/` 下面的所有内容都会上传。** Git 只推送已跟踪并提交的文件。
当前 `.gitignore` 排除了环境、原始数据、运行目录、原始结果、日志、私有集群设置和动态 `reports/latest.json` 等内容。
本报告目录内的 `evidence.json` 是有意提交的轻量评估证据，不含模型权重或完整原始数据集。

可在本地确认：

```bash
git -C codes rev-parse --show-toplevel
git -C codes ls-files
git -C codes status --short
```

## 为什么会有两套 notes？

外层 `notes/` 保存了持续运行期间的历史记录和工具；里面有按日期命名的部署、迁移、归档及恢复事件。
`codes/docs/notes/` 则是早期发布过的少量文档快照，不是对外层 notes 的自动镜像。
两者中的历史状态不能当作当前实时状态。

以后推荐这样分工：提交用的论文内容进入 `reports/`；科学问题与实验来源进入 `tasks/`；
运行工具进入 `operations/`；本地事件日志继续在外层 `notes/` 并由其 README 索引。
新增事件笔记可使用 `YYYY-MM-DD_topic.md`，已被运行脚本引用的旧文件保持原路径。
本次补充导航入口，没有迁移正在使用的脚本、实验目录或日志。
