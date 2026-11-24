> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `notes/autonomous_20261118.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# 自动推进记录：2026-11-18

用户授权：自动向完整 proposal 推进，尽量完成；保留已有修改，不停止训练，不修改运行队列源码。

## 22:33 最新状态

已安排 **66 次训练 + 198 次 100-episode checkpoint 测试**，并非已经完成：

| 队列 | 范围 | 状态 |
|---|---|---|
| 本地 C4 主矩阵 | 25/50/100/200 demos x DP/aug/EquiDiff-C4 x 3 seeds = 36 runs | 首条 C4 50-demo seed 0 正在训练 |
| SICC 27692 | fixed/rotated goal x state DP/equi/goal/gate x 3 seeds = 24 runs | 2080 Ti 上首条 fixed-goal state DP 正在训练 |
| SICC 27693 | C8 在 50/200 demos x 3 seeds = 6 runs | 已提交，依赖 27690 成功后运行，先做完整 GPU 预检 |
| SICC 27690 | A100 缓存版 DP 标定 | 排队，先前预计 11/19 08:03 开始，非保证时间 |
| SICC 27691 | 普通 GPU 分区缓存标定 | COMPLETED，0:0，2m19s |

**关键新证据**：普通 GPU 分区有空闲 RTX 2080 Ti。缓存版 DP 在该卡的 200-step 循环约 43.8 秒；数据准备 499 ms/batch -> 15.0 ms/batch。原先“A100 标定很慢”不能继续用作“集群整体不可用”的判断。A100 本身的新吞吐仍待 27690 实测。

RQ3 已完成状态表示、固定/旋转目标、显式 goal-input、零初始化世界参考 gate；验证了无 goal 方法不能读取 goal，gate=0 保持等变、gate 可获得梯度并在非零时打破等变。八种配置的 CPU 训练/保存/评估 smoke 和远端 GPU 预检通过。该队列使用单独 `envs/rq3`，只读引用已有远端 PyTorch 2.7.1 环境，新依赖仅装入新环境。

C8 作为架构消融：通过 C4 子群单步/采样/梯度检查，不能声称方形物理环境、像素旋转或分量 clipping 具有精确 45 度对称性。完整 GPU 检查在其 Slurm 作业内执行，失败则不会进入六条长期训练。

最终本地测试：**27 项通过**，另有汇总 CSV/JSON/绘图 smoke。原 `codes/` 基线和三个冻结训练版本在 22:33 的 SHA256 审计均无变更，证据 `results/final_source_audit.json`。

后台只读监控 **PID 2555086**，每 10 分钟同步远端结果（不传 checkpoint）、记录 Slurm 状态、汇总三 checkpoint 指标与配对 bootstrap，实际结果满足条件后自动绘图。源码独立冻结于 `releases/monitor_20261118_v1/`。入口：

- 实时状态：[live_status.md](live_status.md)
- 详细监控：`development/proposal_v2/results/monitor/latest.json`
- 自动汇总：`development/proposal_v2/results/summary/matrix_summary.json`、`matrix_summary.csv`
- 远端结果镜像：`development/proposal_v2/results/remote/`
- 本地主队列 supervisor PID 2538616；首条训练 PID 2538617。

**仍未覆盖完整 proposal**：额外 18 条 RQ3 图像实验、官方 DP 完整配方、MimicGen sanity check/可选 E4、完整 max-over-training 指标、PPT/论文成稿。图像中原本可见 goal，不能把状态输入下的“看不到目标”假设直接套到图像实验；该部分需先明确观测与对照协议。集群组剩余配额仍未核实。下列为此前 22:02 的详细实施记录。

## 隔离与运行入口

- 原 `codes/` 及其已有修改保留。已建立逐文件 SHA256 基线，22:01 核对无改动。
- 开发副本：`development/proposal_v2/`。原数据和上游目录只读复用。
- 新依赖只安装在 `development/proposal_v2/envs/dev/`；原项目环境和基座环境未修改。
- 本地主实验冻结版本：`development/proposal_v2/releases/c4_main_20261118_v1/`。
- 队列清单：该版本的 `manifests/queue.json`；状态：`results/supervisor_status.json`。
- 运行日志：该版本的 `logs/`；训练与逐 episode 结果：`runs/`。
- 已启动自动队列：36 次训练、108 次保留 checkpoint 的正式测试、一次首个 C4 模型的验证评估。
- 顺序：50 / 200 / 25 / 100 demos，每个数据量三种子，方法 EquiDiff-C4 / DP / DP+C4 aug。
- 首个任务：`equidiff_c4_n50_s0`，20k updates，batch 64。限制新训练进程的 PyTorch GPU 内存比例为 40%，与原队列共存。
- 调度器在每项任务前核验冻结源码，遇到失败停止后续派发，不停止其他进程。重启时按完成标记跳过已完成项，训练通过现有 checkpoint 显式恢复。

## 已完成的新工作与证据

1. **原版架构 C4 适配已打通**：复用上游 `EquiResBlock`、`EquiDiffusionUNet`，输入改为 96x96 图像、二维位置与二维动作 rho1；保留群元素维度上的 U-Net 权重共享。不是 FrameAvg。参数量 257,559,008。
2. 原版 escnn fork 固定在 `fc4714cb6dc0d2a32f9fcea35771968b89911109`。C4 不需要 SO(3) 的可选 py3nj 加速；未安装系统 gfortran。
3. 小模型单步等变性、反向传播、耦合噪声 DDPM 等变性，以及训练/EMA 保存/加载/仿真闭环通过。
4. 完整模型 GPU 前后向通过。TF32 默认设置时误差约 5.9e-4；关闭 TF32 后最大误差 2.46e-6。新三臂统一关闭 TF32，旧 run 不静默替换成新主表结果。
5. 完整 C4 模型 200 步标定完成：约 102.5 秒训练循环，峰值分配显存 9,584,441,344 bytes（约 8.93 GiB）。标定期间旧 FrameAvg 仍在训练，不能当作独占 GPU 吞吐。
6. 数据缓存/批量读取与原读取逐位一致，短训练权重和 RNG 流验证通过。本地每批数据准备约 139 ms -> 8.15 ms，预加载约 0.63 秒。
7. 新原生 C4 环境在 6 条留出示教完整动作序列、两种 legacy 初始化、四朝向的回放检查通过：位置误差 <=4.89e-11，coverage <=1.20e-12，像素旋转误差为 0。
8. 30 个留出帧检查：新 canonical 渲染与上游原版渲染逐像素相同。原版重渲染与保存的官方图像仍有差异（最大 MAE 0.000602、最大单像素差 0.443），已单独报告，未冒称完整示教重建通过。
9. 批量 C4 评估入口完成：25 测试基础初态 x 4 朝向、per-orientation、成功率、独立 episode RNG、逐批落盘与恢复。CPU 和 GPU smoke 通过。
10. C4 模型 100-step DDPM 推理标定：batch 1 / 4 / 8 每动作块总耗时约 1.86 / 1.49 / 1.52 秒；batch 8 每 episode 每动作块约 0.190 秒。仍为与旧训练共存时的测量，不是完整 rollout 耗时。
11. 新训练保留 16k / 18k / 20k EMA，并保持完整 `last.pt` 可恢复。配对 bootstrap 保留四朝向关联，对训练种子和基础初态分别重采样。
12. 19 项测试通过；调度器的失败后停止派发、源码被改后拒绝运行均通过 smoke。

证据均在 `development/proposal_v2/results/`，完整协议在开发副本及冻结版本的 `PROTOCOL.md`。

## 集群

- SICC 作业 **27690**：`robotic_cache_calib`，22:02 状态 PENDING / Priority。
- 独立远端版本：`${SICC_PROJECT}/releases/cache_calib_20261118/`。
- 1 A100 / 8 CPU / 32G / 1 小时，运行数据缓存对照与 200-step DP 标定。
- 旧远端源码未更新。提交回执在 `development/proposal_v2/results/cache_calibration_submission.json`。
- 仍需实测改进后的 A100 吞吐；当前没有承诺其可贡献多少训练量。

## 必须如实报告的协议修正

- 原场景像素/墙体中心为 255.5，原 goal 为 (256,256)。为保持官方 canonical 图像，将 goal 的**位置和角度一起旋转**；四个位置在 255/256 之间变化。不能声称完全遵循 proposal 中四方向始终同一个 (256,256) 中心的写法。
- 新 renderer 是“逆变换物理状态 -> 原版渲染 -> rot90”，并非证明原版 pygame 光栅化天生等变。物理过程确实在四个不同旋转坐标系的 pymunk space 中推进。
- 实际最大训练量仍为 200，因为保留 6 条验证示教。
- coverage 是交集/目标面积，不是 IoU；成功沿用上游 `>0.95`。
- 三个保留 checkpoint 的平均作为主统计，另报 final；有限三点最大值不冒充全训练历史最大值。
- 尚未完成官方 DP 完整配方复现、MimicGen sanity check、C8、RQ3、E4、PPT。C4 主队列启动不等于整个 proposal 已完成。

## 后续接管顺序

1. 读冻结队列状态和首个 C4 run 日志，确认 loss/保存/评估正常；不得修改冻结版本源码。
2. 查 SICC 27690，完成后回收日志、数据加载对照、训练 metrics/runtime/status。
3. 原 `codes/` 的 FrameAvg seed 2 完成后，汇总原 50-demo 9-run interim 表（与新主实验分开）。
4. 在开发副本继续 RQ3 状态版本、C8 的合理适配和统计汇总脚本；任何新长期队列另建冻结版本。
5. 根据实际训练和完整 rollout 成本重新估算 11/24 结果冻结范围，持续准备汇报材料。
