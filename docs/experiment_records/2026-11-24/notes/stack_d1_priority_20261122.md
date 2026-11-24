> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `notes/stack_d1_priority_20261122.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# Stack-D1 最高优先级启动（2026-11-22）

用户明确要求在本地 RTX 5090 启动 Stack-D1，作为第一优先级。
已完成安全交接，正式 seed 0 于约 02:35 CST 启动。

## 交接与恢复

- 原 C4 25/100-demo 补充队列在第 12,000 步原子保存后暂停。
  停止前验证了 checkpoint 可加载，包含 model/EMA/optimizer 和全部 RNG 状态，
  计算 SHA256 并 fsync；未修改其冻结源码和配置。
- 保存文件：`development/proposal_v2/releases/conv_c4_n25_n100_local_20261122_v1/runs/equidiff_c4_n25_s0/last.pt`。
- checkpoint SHA256：`b93aa2922a156cf8023bbe6395171f16dacd342ca94f491d3f0c04d4fa978f17`。
- 暂停回执及原始续跑命令：
  `development/proposal_v2/operations/stack_d1_priority_20261122/c4_pause_receipt.json`。
- 自动续跑守护 PID 123485：等待 Stack-D1 完整训练、最终评估、模型本地 hash 清单全部完成并复核，
  再验证 C4 checkpoint 和冻结源码后启动原 supervisor。Stack 失败时保持 C4 暂停，供排障。
- 不停止其他项目 GPU 服务，不改变两个集群上的任务。

## 正式 Stack-D1

- 冻结 release：`development/proposal_v2/releases/stack_d1_s0_local_20261122_v1`。
- Supervisor PID：119932；训练 PID：120717。
- 完整配置：seed 0、100 demos、500 epochs、batch 128、C8 图像模型、完整 U-Net、EMA、100 步扩散。
- 9,876 个训练序列，每 epoch 77 个 batch，共 38,500 次更新。
- 每 10 epochs 原版评估：6 个训练初态 + 50 个测试初态，各至多 400 步。
  最后另外评估 epoch-500 最终权重。
- 操作参数：评估环境并行数 28 改为 4；训练/验证 DataLoader workers 各 2；
  禁用外部 W&B 上传；PyTorch allocator 上限 80%。训练和评估预算均不缩减。
- checkpoint 使用同步写临时文件、fsync、原子重命名；保留 latest、top-5 和 final。
- 保存每 epoch 上游日志、10 秒进度心跳、最终评估和本地模型 hash 清单。

## 验证与初始耗时

batch-128 预检完成 10 次更新、batch-128 的 100 步采样和 4 worker 短 rollout。
稳定速度 0.4369 秒/更新，峰值分配显存 14.16 GiB；正式第一个 epoch 也约 2.3 更新/秒。
纯更新约 4.7 小时；初始评估每组四个 episodes 约 35 秒，一轮有 14 组。
加上 50 轮计划评估、最终评估、验证、采样和存盘，初步预计 12-16 小时，仍需持续测速校准。
不能把第一个 epoch 或短预检的零成功率作为最终性能结论。

## 监控和溯源

- 对应任务结果入口：`codes/tasks/rq1_transfer/results/stack_d1_s0_local_20261122_v1`
  是指向冻结 release 的本地符号链接。
- 实时状态：release 内 `runs/stack_d1_n100_s0/status.json`。
- 训练日志：release 内 `logs/train.log`；科学指标：run 内 `logs.json.txt`。
- 可提交操作代码及说明：`codes/operations/stack_d1_local_20261122/`。
- fleet 已识别 Stack supervisor 和 C4 continuation watcher，两者 alive。
  Stack 是独立参考实验，不混入 Push-T 主矩阵完成数量。
- 自动续跑的 3 项保护测试通过：成功只启动一次、Stack checkpoint 损坏拒绝启动、Stack 失败拒绝启动。

02:45 左右确认：首个 epoch 的训练、56-episode 完整评估、验证、latest/top-k
checkpoint 保存全部完成，已进入 epoch 1（第 2 个 epoch）。首次完整评估耗时
484.7 秒（8.08 分钟），训练速度仍约 2.3 updates/s。代码已提交为 `ea24000`；
GitHub 推送仍缺少 HTTPS 认证，尚未成功。
