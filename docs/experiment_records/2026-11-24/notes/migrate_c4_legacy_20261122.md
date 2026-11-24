> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `notes/migrate_c4_legacy_20261122.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# C4 收敛实验迁移到旧集群

2026-11-22：用户要求将为 Stack-D1 暂停的收敛实验迁移至旧集群。
本次接续 25/100 demos × seeds 0/1/2 共六条曲线，每条保留
20k/40k/60k checkpoint 和每个 checkpoint 的 100-episode validation。
这是探索性收敛检查，不能替代主实验 held-out test。

| 作业 | Release | 实验 | 部署状态 |
| --- | --- | --- | --- |
| 232837 | `conv_c4_n25_s1_legacy_20261122_v1` | 25 demos，seed 1 | fstsvr10 A40，已开始训练 |
| 232837 | `conv_c4_n100_s02_legacy_20261122_v1` | 100 demos，seed 0 后接 seed 2 | fstsvr04 A40，seed 0 已开始训练 |
| 232838 | `conv_c4_n25_s02_legacy_20261122_v1` | 25 demos，seed 0 从 12k 续训，之后 seed 2 | 暂挂，等待断点上传校验 |
| 232838 | `conv_c4_n100_s1_legacy_20261122_v1` | 100 demos，seed 1 | 随同作业等待放行 |

第一组已通过完整 batch-64 CUDA 前后向和原生 C4 回放验证，两条训练状态均持续推进。
第二组已申请另外两张 GPU；`JobHeldUser` 是主动等待输入准备，不是训练失败。
后台 `stage_resume.py` 完成上传、远端大小与 SHA256 校验、原子改名后才释放该作业。
释放后能否立即运行取决于空闲资源。恢复 optimizer/RNG 的 GPU 检查在作业内执行，
在实际通过之前不能声称跨环境断点续训已成功。

原始断点为 `equidiff_c4_n25_s0/last.pt`，12,000 steps，4,125,372,702 bytes，SHA256：

```text
b93aa2922a156cf8023bbe6395171f16dacd342ca94f491d3f0c04d4fa978f17
```

本地原断点、训练日志和冻结源码全部保留。原 C4 自动续跑 watcher 已停止，并从活跃监控移除，
避免 Stack-D1 完成后重复运行六条曲线。Stack-D1 原训练进程继续，10:39 检查已到 epoch 310/500 的评估阶段。

旧集群只新增项目环境 `envs/c4_legacy_20261122`，没有修改集群系统配置或现有环境。
运行时从本地 Python 3.10 / PyTorch 2.12 / CUDA 12.8 / RTX 5090，变为
Python 3.11 / PyTorch 2.6 / CUDA 12.4 / A40；科学配置和训练源码保持一致，
但不能声称跨硬件逐位一致。模型、EMA、optimizer 与保存的随机状态一并迁移。
部署时配额使用约 242/500 GB，数据和上游源码共 1,316 个文件已校验。

四个 release 已接入现有逐实验归档服务：完成训练和评估后，校验转存本地，
再清理远端重复 checkpoint，之后才运行该 lane 的下一 seed。
结果入口为 `codes/tasks/rq1_transfer/results/<release>/`。

首次跨节点 Slurm 提交的未绑定 step 被调度至同一个节点；通过额外绑定 fstsvr10 的
step 232837.2 使用已分配的第二张卡。未中止训练，也未修改运行中的批处理或源码。
原始提交脚本已保留，未来提交模板增加节点绑定。原 lane 锁及归档完成回执防止重复训练。

统一监控已补齐四份源码快照元数据，采集器恢复更新，10:37 快照无告警。
后台断点传输仍由同一监控检查心跳。上传守护四项测试通过：正常放行、坏哈希拒绝、
错误作业拒绝、已经运行的作业幂等处理。

代码与详细说明：[迁移操作目录](../../../../operations/migrate_c4_legacy_20261122/README.md)。
本地私有部署回执、上传日志和状态在
`development/proposal_v2/operations/migrate_c4_legacy_20261122/`。
