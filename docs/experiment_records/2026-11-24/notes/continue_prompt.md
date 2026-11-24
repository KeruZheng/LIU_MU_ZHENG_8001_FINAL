> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `notes/continue_prompt.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# 第二个对话的续接 prompt

把下面整段复制到新对话即可。

---

你接手一个已经进入"原版 EquiDiff-C4 + RQ3 + C8"阶段的项目。项目里有两个独立工作线：原 `codes/` 的 FrameAvg fallback 已跑完；新的主实验在 `development/proposal_v2/`，正在自动排队执行。请先读文档和实时状态，再动手。

## 项目路径

- 项目根：`${WORKSPACE}`
- 原 `codes/`（已冻结的 fallback 线）：`${WORKSPACE}/codes`
- **新主线**：`${WORKSPACE}/development/proposal_v2`
- 汇报日期：**2026-11-26（周四）pre**；结果冻结目标：**11/24 晚**
- 当前日期：2026-11-19

## 先读这些（按顺序）

1. `notes/progress.md` — 最新进度总览（两条线、当前队列、结果边界）
2. `notes/autonomous_20261118.md` — 原版 C4 / RQ3 / C8 的实施记录
3. `notes/live_status.md` — 实时队列状态
4. `notes/reproduction.md` — 复现证据
5. `notes/cluster.md` — SICC / 2080 Ti / A100 状态
6. `development/proposal_v2/PROTOCOL.md` — C4 主协议
7. `development/proposal_v2/RQ3_PROTOCOL.md`、`C8_PROTOCOL.md`
8. `03_Proposal_A_EquiDiff_final.md` — 原始 proposal
9. `codes/README.md` — 原 fallback 线的边界

## 两条线（不要混淆）

### A. 原 `codes/`：FrameAvg fallback（已完成）

- 50 demos × 3 seeds × 20k 步；10-seed interim 评估。
- 结果：
  - `dp`：0.3179 / 0.1957 / 0.1375 → mean **0.2171**
  - `dp_c4_aug`：0.6772 / 0.5286 / 0.5832 → mean **0.5963**（seed 2 success 0.10）
  - `dp_c4_frameavg`：0.2648 / 0.2444 / 0.4030 → mean **0.3041**
- aug 在三个训练种子上都最高；FrameAvg 与 DP 互有胜负，3-seed 均值 aug > FrameAvg > DP，但 FrameAvg/DP 差距在方差内。
- 这是 **fallback 表**，不是原版 EquiDiff 主表；PPT 必须分表。

### B. `development/proposal_v2/`：主实验（进行中）

- 原版 EquiDiff-C4 适配已打通：上游 `EquiResBlock` / `EquiDiffusionUNet`，96×96 图像、2D 位置/动作 ρ1、群元素权重共享；257,559,008 参数。不是 FrameAvg。
- 主矩阵：25/50/100/200 demos × {EquiDiff-C4, DP, DP+C4 aug} × 3 seeds = 36 次训练；20k updates、batch 64、`[512,1024,2048]`、TF32 关闭。
- 评估：25 基础 seed × 4 朝向 = 100 episodes；per-orientation、success、独立 RNG；保留 16k/18k/20k 三个 checkpoint，主统计取三点均值；配对 bootstrap。
- 原生四朝向示教回放通过：位置误差 ≤4.89e-11、coverage ≤1.2e-12、像素旋转误差 0。
- RQ3：fixed/rotated goal × state DP / EquiDiff / EquiDiff+goal / EquiDiff+gate × 3 seeds = 24 runs；SICC 27692 运行中。
- C8：架构消融 6 runs；SICC 27693 待预检。
- 27 项本地测试通过。

## 当前运行状态（2026-11-19 00:22）

- 本地 C4 主队列 `c4_main_20261118_v1`：145 jobs（36 训练 + 108 checkpoint 测试 + 1 验证）；当前 `equidiff_c4_n50_s0`，**step 16,135 / 20,000**，loss 0.00685，峰值 9.59 GB；预计约 00:40 训练完。
- Supervisor PID 2538616；当前训练 PID 2538617；监控 PID 2555086（只读，每 10 分钟同步，不传 checkpoint）。
- SICC **27692**：RQ3 24 runs / 96 jobs，RUNNING；当前 `fixed_state_equidiff_n50_s0_test_20000`，80 episodes，mean 0.5162，success 0.0375（未完成）。
- SICC **27691**：2080 Ti 缓存标定 COMPLETED，2m19s；200-step 43.8 s；数据准备 499 → 15 ms/batch。
- SICC **27690**：A100 缓存版 DP 标定 PENDING；完成后才能判断 A100 是否值得作为第二池。
- SICC **27693**：C8 6-run，PENDING，依赖 27690；作业内先做完整 GPU/采样等变预检。
- 已有 RQ3 状态结果（1 个训练 seed、state 输入、非图像）：`fixed_state_dp_n50_s0` mean **0.7688**，success **0.3433**，per-orientation `[0.818, 0.732, 0.708, 0.818]`。

## 关键决策与边界

- **原 `codes/` 已冻结**：不要修改其源码/队列；新工作只在 `development/proposal_v2/`。
- **不要修改运行中的 frozen release**：`releases/c4_main_20261118_v1/`、`rq3_state_20261118_v1/`、`c8_ablation_20261118_v1/` 的源码、配置、队列都不可改；新长期队列必须新建 release。
- **不要取消/重启训练**；不要同步 checkpoint；监控只读。
- **不要提前声称 C4 主结果**：第一条 run 还在训练，36 条 C4、24 条 RQ3、6 条 C8 都未完成。
- **Fallback 表 ≠ 主表**：`codes/` 的 FrameAvg 结果不能写成 EquiDiff 结果；`proposal_v2` 的 EquiDiff-C4 才是主方法。
- **RQ3 是 state 输入**：`fixed_state_dp` 的 0.7688 不能与图像主表直接比较；图像里 goal 可见，18 条图像 RQ3 实验尚未做。
- **C8 不是精确 45° 对称**：只能作为架构消融，不能声称环境/像素 C8 精确增强。
- **协议修正**：goal 位置和角度一起旋转（四个位置 255/256 变化）；renderer 是"逆变换→原版渲染→rot90"构造；coverage 是交集/目标面积，不是 IoU；success `>0.95`；最大 200 demos；三点均值 ≠ 完整 max-over-training。
- **组存储配额未核实**；checkpoint 4.6 GB/run，注意磁盘。

## 立即要做

1. 读 `notes/live_status.md` 和 `development/proposal_v2/results/monitor/latest.json`；确认 C4 第一条 run 是否完成，检查 loss、EMA 16k/18k/20k、validation 和 100-episode test 是否正常。
2. 检查 `equidiff_c4_n50_s0` 的 100-episode 测试结果；如果异常，先停队列并排查，不硬跑后面 35 条。
3. 查 SICC 27690、27692、27693；回收 27690 的 A100 缓存标定结果；确认 C8 预检是否通过。
4. 更新 `development/proposal_v2/results/summary/matrix_summary.json/.csv`；只读同步远端结果，不传 checkpoint。
5. 准备 PPT：先做方法/协议修正/验证/进度页；fallback 表和 C4 主表分开，结果页留占位。
6. 11/24 按实际完成 runs 冻结主表；未完成项明确写"进行中/未完成"。
7. 有余量再补：官方 DP 完整配方、MimicGen sanity、RQ3 图像实验、完整 max-over-training。

## 常用命令

```bash
cd ${WORKSPACE}/development/proposal_v2

# 实时状态
cat ../notes/live_status.md
cat results/monitor/latest.json

# 本地 C4 第一条 run 进度
tail -1 releases/c4_main_20261118_v1/runs/equidiff_c4_n50_s0/metrics.jsonl

# 汇总表
cat results/summary/matrix_summary.json
cat results/summary/matrix_summary.csv

# 本地 GPU / 进程
nvidia-smi
ps -eo pid,etime,pcpu,args | grep -E 'supervise|equidiff_c4|watch_progress' | grep -v grep

# 集群
squeue -u ${CLUSTER_USER}
sacct -j 27690,27691,27692,27693 --format=JobID,JobName,State,Elapsed,ExitCode,NodeList
```

## 汇报时的结论边界

- 可以说：原版 EquiDiff-C4 适配、原生 C4 回放、100-episode 评估、RQ3/C8 代码均已打通；fallback 表已完成；C4 主矩阵正在跑。
- 不能说：C4 主矩阵已经完成或已有性能结论；RQ3 的 state 结果可以直接代表图像任务；C8 是精确物理对称。
- 必须写：fallback 表与主表分开；C4 结果仍在进行；goal 位置/渲染器/coverage 定义与 proposal 的差异；完整 proposal 的官方 DP 配方、MimicGen/E4、RQ3 图像实验仍未完成。
