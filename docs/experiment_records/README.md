# 实验记录

这里收录实验过程的公开记录：为什么这样设计、实际怎样运行、哪里失败过、怎样恢复、
代码和数据来自哪个版本。仓库根仍是本地 `wangdian/codes/`；外层工作目录保持原位。

**画图和数值分析请先看 [实验结果与画图数据地图](../../reports/main_results_20261123/DATA_MAP_ZH.md)。**
主结论见 [完整报告](../../reports/main_results_20261123/REPORT_ZH.md)。
过程笔记中带有各日期的历史状态，不能将旧的“正在跑”“待完成”直接当作当前进度。

## 当前公开记录

[2026-11-24 快照目录](2026-11-24/README.md) 收录 67 份源记录，包括外层 `notes/` 的全部
23 份 Markdown、3 份主协议、26 份 release 协议、6 份数据/版本清单和 9 份轻量证据。
另有 [release 目录清单](2026-11-24/release_inventory.json) 和
[原始文件 / 发布文件 SHA256](2026-11-24/manifest.json)。记录中的账号、内网地址和机器绝对路径
替换为占位符；日期、实验参数、结果、失败原因、作业编号及文件哈希保留。
发布转换和断开的本地引用均在清单中说明。未复制环境、数据集、模型权重或成批日志。

| 想查什么 | 记录入口 |
|---|---|
| 总体进度与执行清单的演变 | [progress](2026-11-24/notes/progress.md)、[list](2026-11-24/notes/list.md) |
| pilot、fallback 与正式 EquiDiff 的边界 | [复现笔记](2026-11-24/notes/reproduction.md)、[11/18 主线建立](2026-11-24/notes/autonomous_20261118.md) |
| C4、RQ3、C8 的原始实验约定 | [C4](2026-11-24/protocols/PROTOCOL.md)、[RQ3](2026-11-24/protocols/RQ3_PROTOCOL.md)、[C8](2026-11-24/protocols/C8_PROTOCOL.md) |
| 数据来源、划分、评估 seed | [upstream](2026-11-24/manifests/upstream.json)、[split](2026-11-24/manifests/split.json)、[evaluation seeds](2026-11-24/manifests/evaluation_seeds.json) |
| quota、checkpoint 归档及节点失败 | [归档与 quota](2026-11-24/notes/storage_archive_20261119.md)、[恢复记录](2026-11-24/notes/autonomous_20261119.md) |
| 两集群和 5090 的分工、补充实验与迁移 | [资源优化](2026-11-24/notes/resource_optimization_20261120.md)、[补排](2026-11-24/notes/fill_clusters_20261120.md)、[C4 迁移](2026-11-24/notes/migrate_c4_legacy_20261122.md) |
| 参考实验、E4 训练和 EGL 恢复 | [Stack-D1](2026-11-24/notes/stack_d1_priority_20261122.md)、[E4 并行](2026-11-24/notes/e4_parallel_20261123.md) |
| 为什么 E4 旧 D1/D2 得分不能作最终结论 | [评估审计与修正](2026-11-24/notes/2026-11-24_e4_evaluation_audit.md)、[修正代码与已通过预检](../../operations/e4_corrected_eval_20261124/README.md) |
| 最终主结果与绘图数值 | [报告](../../reports/main_results_20261123/REPORT_ZH.md)、[数据地图](../../reports/main_results_20261123/DATA_MAP_ZH.md) |

## 实验进程如何理解

- 11/16--17：固定上游代码与数据，确认权限、CUDA 和项目环境，做工程检查与 pilot。
- 11/18：区分 FrameAvg fallback 和原版 EquiDiff-C4 适配，冻结原生 C4 协议并启动主矩阵。
- 11/19：处理 quota、损坏 checkpoint 和独立的 GPU 节点故障；建立校验归档后清理的流程。
- 11/20--22：补齐收敛、图像 RQ3、参考实验；调配两套集群和 5090，保留各 release 的身份。
- 11/23：冻结 RQ1/RQ2/RQ3 主报告与逐 episode 证据，输出中英文 LaTeX；完成 114 次主研究及扩展/收敛训练。
- 11/24：发现 E4 软重置没有逐 episode 重采样插柱，独立修正并重评估；旧迁移分数暂不采用。

114 次 = 36 主矩阵 + 24 state RQ3 + 12 新增 image RQ3 + 6 C8 + 36 收敛。
官方 DP 和 Stack-D1 参考检查另计；Optional E4 也另计。
本次快照中的 E4 状态是导出时刻的一次观察，不会随远端作业自动更新。

## development 与 notes 的职责

| 本地目录 | 作用 | GitHub 对应位置 |
|---|---|---|
| `notes/` | 实验决策、过程、排障、交接和历史工具 | 本目录的带日期 `notes/` 快照；执行工具以 operations 中的版本为准 |
| `development/proposal_v2/releases/` | 每批实验冻结源码、配置、训练目录及模型 | `experiments/releases/` 中的精选科学代码；本目录中补充协议与版本清单 |
| `development/proposal_v2/results/` | 回传结果、模型备份、验证和归档收据 | `reports/` 中的冻结科学数据；本目录中精选轻量证据 |
| `development/proposal_v2/{src,configs,scripts,tests,slurm}/` | 开发工作区和实验工具 | 各冻结 release 与 `operations/`，按实际运行版本追溯 |
| `development/proposal_v2/{envs,data,upstream}/` | 依赖环境、输入数据、上游代码 | 版本与来源清单；大文件不直接上传 |
| `development/proposal_v2/operations/` | 运维脚本运行副本、上传包和回执 | `operations/` 中的公开工具；本目录中精选回执 |

目录统计于 11/24：`development/` 约 842 GB，`notes/` 约 468 KB。
完整目录不适合作为普通 Git 历史；公开的是可核对的代码、协议、数值与过程记录。
精确训练权重和大体积原始文件继续按本地校验归档流程保存。

## 后续更新

新增实验事件继续记录在本地 notes；达到阶段节点时发布一个新的日期快照并提交 Git，
保留旧快照。不要覆盖旧记录来掩盖失败或改变结果解释。本次没有建立后台自动推送服务。

从仓库根运行以下命令可导出或校验；脚本不训练、不提交作业、不移动原文件：

```bash
python3 scripts/publish_experiment_records.py --snapshot NEW_DATE_OR_LABEL
python3 scripts/publish_experiment_records.py --snapshot 2026-11-24 --verify
```

导出需要原本地工作区，校验只需要仓库中的公开快照。
各发布文件的哈希与源文件哈希分别记录，因为隐私占位符、历史标记及链接适配会改变字节。

## Latest Recovery Record

[2026-11-26: E4 continuation and independent evaluator recovery](2026-11-26/README.md)
records passed checkpoint/D2 regression checks, automatic submission deferral,
and the new full-scene evaluation protocol. It does not claim E4 is complete.
