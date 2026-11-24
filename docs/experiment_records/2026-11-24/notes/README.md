> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `notes/README.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# 本地 notes 导航

本目录是运行与接管记录，不是最终报告目录。**GitHub 的仓库根是 `../codes/`；本目录没有整体上传。**
仓库里的 `codes/docs/notes/` 是部分早期公开快照，并不自动跟随这里更新。

## 优先入口

| 需求 | 入口 | 如何使用 |
|---|---|---|
| 理解 background、gap、三个问题和结论 | [主实验完整报告](../../../../reports/main_results_20261123/REPORT_ZH.md) | 已冻结结果，适合汇报与写作 |
| 中文快速掌握 / 英文论文稿 | [中英文 LaTeX](../../../../reports/main_results_20261123/latex/README.md) | 正文、独立入口、PDF 和 Overleaf 包 |
| 了解哪些内容上传 GitHub | [目录与同步范围](../../../PROJECT_MAP.md) | `codes/` 是仓库根 |
| 查三个设备的实时状态 | Fleet 状态 (`../codes/operations/fleet/results/STATUS.md`; local-only historical reference) | 先看更新时间，不用历史笔记猜实时状态 |
| 查 E4 实时状态 | E4 状态 (`../codes/tasks/rq1_transfer/results/e4_monitor/STATUS.md`; local-only historical reference) | 区分训练、定期评估、最终评估和归档 |
| 查 E4 修正评估 | 修正评估状态 (`../codes/tasks/rq1_transfer/results/e4_corrected_eval_20261124/STATUS.md`; local-only historical reference) | 11/24 发现场景重置缺陷；旧 D1/D2 分数待复评，训练完成数不代表科学验证完成 |
| 查实验对应的源码和证据 | [任务总览](../../../../tasks/README.md) | 按 RQ1/RQ2/RQ3 查找，不按机器找实验身份 |
| 接管上下文 | [progress.md](progress.md)、[continue_prompt.md](continue_prompt.md) | 包含历史条目；以时间戳和实时结果核实 |

## 研究计划、协议与阅读材料

- [原始 proposal](../../../../reports/main_results_20261123/proposal_snapshot.md)：RQ1/RQ2/RQ3 和 Optional E4 的原始定义。
- [list.md](list.md)：计划演变与早期任务清单，不是当前唯一完成度来源。
- [reproduction.md](reproduction.md)：复现过程、pilot 和实现边界，保留历史结果。
- [study_list.md](study_list.md)：论文阅读路线。
- [claude.md](claude.md)：已有协作记录，保留原文。

## 集群、资源与队列事件

- [cluster.md](cluster.md)：访问、环境、配额和集群历史。
- [autonomous_20261118.md](autonomous_20261118.md)、[autonomous_20261119.md](autonomous_20261119.md)：早期自主部署与恢复。
- [parallel_20261119.md](parallel_20261119.md)、[fleet_20261119.md](fleet_20261119.md)：并行与监控机制。
- [fill_clusters_20261120.md](fill_clusters_20261120.md)、[resource_optimization_20261120.md](resource_optimization_20261120.md)：资源补排与优化事件。
- [local_followup_20261122.md](local_followup_20261122.md)、[migrate_c4_legacy_20261122.md](migrate_c4_legacy_20261122.md)：C4 后续实验和旧集群迁移。

## 参考实验与 E4

- [2026-11-24 E4 评估审计](2026-11-24_e4_evaluation_audit.md)：发现 D1/D2 软重置未逐回合重采样目标柱；已有分数需暂列待复评，不能作为完整目标分布的迁移结论。
- [stack_d1_20261122.md](stack_d1_20261122.md)、[stack_d1_priority_20261122.md](stack_d1_priority_20261122.md)：Stack-D1 参考检查及优先级交接。
- [e4_monitor_20261122.md](e4_monitor_20261122.md)：E4 本地启动与监控记录。
- [e4_parallel_20261123.md](e4_parallel_20261123.md)：E4 分配、渲染恢复与并行部署历史。
- [live_status.md](live_status.md)：旧监控的历史输出；当前优先读顶部链接的 Fleet/E4 状态。

## 存储和运维工具

[storage_archive_20261119.md](storage_archive_20261119.md) 说明远端结果转存与校验。
本目录还包含 `archive_remote_20261119.py`、`recover_c8_20261119.py`、
`storage_guard_20261119.py`、`storage_guard_20261119.sbatch` 及相应 `test_*.py`。
它们是历史运行工具，不能仅凭文件名判断是否仍应执行；先读相关记录并核对当前队列。

## 命名与使用约定

新的事件笔记可统一采用 `YYYY-MM-DD_topic.md`；已存在的文件不批量重命名，避免破坏脚本和文档引用。
每篇事件记录应写明日期、执行对象、事实证据与后续状态来源。
论文与结果解释进入 `codes/reports/`，实验身份进入 `codes/tasks/`，运行机制进入 `codes/operations/`。
本目录负责保留执行历史，不再承担最终报告的作用。
