> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `notes/progress.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# 项目进度总览

> 2026-11-24 16:38：修正评估 v2 作业 **29935** 已通过 D0/D1/D2 全部场景配对、图像复现与真实模型 CUDA 预检，进入正式 D1 评估，后续自动 D2、D0，各 100 episodes。v1 的 ESCNN 缓存加载失败已修复并保留历史。当前 v2 状态 (`../codes/tasks/rq1_transfer/results/e4_corrected_eval_20261124/v2/STATUS.md`; local-only historical reference)；新科学分数仍待完整评估，旧 D1/D2 零成功不用于最终结论。

> 2026-11-24 16:24：E4 发现 D1/D2 软重置未逐 episode 重采样插柱，旧迁移分数暂不作为完整目标分布结论。独立修正版已部署；SICC 2080 Ti 重置预检 29923 通过，完整预检与 300-episode 重评估作业 29929 已运行。现有训练不变，主 RQ1/RQ2/RQ3 不受该问题影响。详见 [审计与修正](2026-11-24_e4_evaluation_audit.md)、修正评估实时状态 (`../codes/tasks/rq1_transfer/results/e4_corrected_eval_20261124/STATUS.md`; local-only historical reference)。

> 2026-11-23：已完成 [RQ1/RQ2/RQ3 主实验完整报告](../../../../reports/main_results_20261123/REPORT_ZH.md)，包含 background、research gap、三个问题、方法、结果、假设判定及限制。冻结了 78 个 test runs 和 36 个 validation runs 的完整三-checkpoint评估证据；C4 收敛最后一项的 60k 评估也已纳入。E4 在原 proposal 中明确是 Optional E4，独立于三个主问题，报告不使用未完成的 E4 迁移结果。

> 2026-11-22 10:40：按用户指令，原暂停的 25/100-demo C4 收敛六条曲线迁移到旧集群。232837 已在两张 A40 上训练；232838 等待 12k 断点上传和 SHA256 校验后自动释放，释放后仍可能排队。本地 C4 自动续跑已禁用，5090 保持 Stack-D1。详见 [迁移记录](migrate_c4_legacy_20261122.md)。

> 2026-11-22 02:35：按用户最高优先级指令，Stack-D1 seed 0 已在本地 5090 正式启动，保留 500 epochs / batch 128。C4 补充队列在已校验的 12,000-step checkpoint 暂停；Stack 完成并验证结果后自动续跑。详见 [优先级交接与启动记录](stack_d1_priority_20261122.md)。

> 2026-11-22：Stack-D1 独立环境与完整绝对动作数据已准备完成；完整模型 batch-1、仿真、官方训练/评估/保存工作流均通过预检。正式 batch 128 的资源预检和正式训练尚待执行。详见 [Stack-D1 准备记录](stack_d1_20261122.md)。现有 5090 队列持续运行。

> 2026-11-22：主矩阵 36/36、state RQ3 24/24、图像 RQ3 新增 12/12、DP/增强收敛各 12/12 均完成。官方 DP seed 42 完整 3050 epoch 已完成并归档。5090 已启动独立的 25/100-demo C4 收敛补充队列，详见 [本地追加与剩余范围](local_followup_20261122.md)。Stack-D1、可选 E4 和主实验完整 max-over-training 仍未覆盖。

> 2026-11-20 23:12：两个集群已追加 6 个作业。SICC 的 A100/H800/V100 各 2 个提交名额已满；旧集群两项双卡作业已排队。新增精确增强收敛、C4 剩余 seeds 和图像 RQ3。图像 RQ3 的 18 个比较单元复用 6 条已有基线，新增 12 次训练；详见 [本次排队与协议记录](fill_clusters_20261120.md)。下方较早进度为历史快照。

> 2026-11-19：已按 RQ1/RQ2/RQ3 整理 `codes/tasks/`，新增三平台统一看板与独立看门狗。入口：统一状态 (`../codes/operations/fleet/results/STATUS.md`; local-only historical reference)、[目录与监控交接](fleet_20261119.md)。原训练路径与冻结源码保持不变。

> 2026-11-19 22:43：旧集群已提交收敛检查作业 **232570、232571**，每项 2 GPU，共申请 4 GPU；提交后状态均为 `PENDING (Priority)`，尚未完成 GPU 预检。详情见 [追加实验部署记录](parallel_20261119.md)。现有实时监控页仍主要显示原三条主实验队列。

> 2026-11-19 21:00 更新：已部署逐实验归档关卡和集群恢复流程。RQ3 作业 27809 在 A100 上通过预检并续评估；C8 等损坏 checkpoint 备份校验后自动隔离旧运行并重新提交。详见 [恢复与归档记录](autonomous_20261119.md)，实时状态见 [live_status.md](live_status.md)。下方时间点为历史快照。

更新：**2026-11-19 00:22 CST**（按运行产物、`live_status.md`、`monitor/latest.json` 复核）
项目根：`${WORKSPACE}`
正式汇报：**2026-11-26（周四）pre**；结果冻结目标：**11/24 晚**
科学路线：`03_Proposal_A_EquiDiff_final.md`

> 当前主线已从 `codes/` 的 FrameAvg fallback 转为 `development/proposal_v2/` 的**原版 EquiDiff-C4 适配**。原 `codes/` 保持冻结；新训练在独立 release 中运行。
> 实时状态：[live_status.md](live_status.md)
> 详细实施记录：[autonomous_20261118.md](autonomous_20261118.md)
> 复现证据：[reproduction.md](reproduction.md)

---

## 1. 两条工作线

| 工作线 | 目录 | 内容 | 状态 |
|---|---|---|---|
| **Fallback 表** | `codes/` | DP / DP+C4 aug / DP-C4-FrameAvg；50 demos × 3 seeds；20k 步；10-seed interim | ✅ 已完成，独立于新主线 |
| **主实验** | `development/proposal_v2/` | 原版 EquiDiff-C4 / DP / DP+C4 aug；25/50/100/200 demos × 3 seeds；100-episode 评估（25×4） | 🔄 运行 / 排队 |
| RQ3 状态实验 | `development/proposal_v2/releases/rq3_state_20261118_v1/` | fixed/rotated goal × DP / EquiDiff / EquiDiff+goal / EquiDiff+gate × 3 seeds | 🔄 SICC 27692 运行 |
| C8 消融 | `development/proposal_v2/releases/c8_ablation_20261118_v1/` | C8 在 50/200 demos × 3 seeds | ⏳ 27693 待预检 |

**注意**：`codes/` 的 FrameAvg 表不能与 `proposal_v2` 的 EquiDiff-C4 主表混在一起；前者是明确标注的 fallback，后者才是 proposal 的主方法。

---

## 2. 当前运行状态（00:22）

### 本地 5090

- **C4 主队列** `c4_main_20261118_v1`：36 次训练 + 108 次保留 checkpoint 测试 + 1 次验证评估 = **145 jobs**；当前第 0 个任务 `equidiff_c4_n50_s0`，**step 16,135 / 20,000**，loss 0.00685，峰值 9.59 GB，update 0.304 s/step，data 7 ms/batch。预计约 00:40 训练完。
- 队列顺序：50 / 200 / 25 / 100 demos；方法 EquiDiff-C4 / DP / DP+C4 aug；3 seeds。
- Supervisor PID `2538616`；当前训练 PID `2538617`；监控 PID `2555086`（每 10 分钟同步，只读，不传 checkpoint）。
- GPU：约 19.2 GB / 32.6 GB，利用率约 97%。

### SICC 集群

| Job | 内容 | 状态 |
|---|---|---|
| **27692** | RQ3 状态矩阵，24 runs / 96 jobs | 🔄 RUNNING；当前 `fixed_state_equidiff_n50_s0_test_20000`，已完成 80 episodes，mean coverage 0.5162，success 0.0375 |
| **27691** | 2080 Ti 缓存标定 | ✅ COMPLETED，2m19s；200-step 循环 43.8 s；数据准备 499 → 15 ms/batch |
| **27690** | A100 缓存版 DP 标定 | ⏳ PENDING（Priority）；完成后才能判断 A100 是否值得作为第二池 |
| **27693** | C8 6-run 消融 | ⏳ PENDING，依赖 27690；作业内先做完整 GPU/采样等变预检 |

已有 RQ3 状态结果（1 个训练 seed，state 输入，非图像）：

- `fixed_state_dp_n50_s0`：mean coverage **0.7688**，success **0.3433**，per-orientation `[0.818, 0.732, 0.708, 0.818]`。
- 当前 `fixed_state_equidiff_n50_s0` 评估未完成；不能与上面的 DP 结果做最终比较。

---

## 3. 已完成结果

### 3.1 原 `codes/` FrameAvg fallback 50-demo × 3 seeds（10-seed interim）

| 方法 | seed 0 | seed 1 | seed 2 | 3-seed mean |
|---|---:|---:|---:|---:|
| `dp` | 0.3179 | 0.1957 | 0.1375 | **0.2171** |
| `dp_c4_aug` | 0.6772 | 0.5286 | 0.5832 | **0.5963** |
| `dp_c4_frameavg` | 0.2648 | 0.2444 | 0.4030 | **0.3041** |

- `dp_c4_aug` 在三个训练种子上都最高；seed 2 有一个 episode 成功（900007，0.9505），success 0.10。
- FrameAvg 与 DP 互有胜负：seed 0 是 DP 更高，seed 1/2 是 FrameAvg 更高；3-seed 均值 aug > FrameAvg > DP，但 FrameAvg 与 DP 的差距在方差范围内，**不能声称稳定排序**。
- 这是 10 个 canonical 验证 seed 的 **interim**，不是 proposal 的 100-episode 旋转正式评估。

### 3.2 早期 canonical / balanced pilot（供对照，不进入主表）

| 数据 | dp | dp_c4_aug | dp_c4_frameavg |
|---|---:|---:|---:|
| canonical-only 20k（10 seeds） | 0.4336 | 0.6810 | 0.3489 |
| balanced 20k（seed 0） | 0.3179 | 0.6772 | 0.2648 |

- canonical-only 违反三臂共享平衡朝向协议；FrameAvg 被强制错误等变。
- 2 episodes 会严重误导（20k DP 2 seeds 0.051 vs 10 seeds 0.4336）。

### 3.3 新主线 `proposal_v2` 的已完成工程与验证

- **原版 EquiDiff-C4 适配已打通**：复用上游 `EquiResBlock` / `EquiDiffusionUNet`，96×96 图像、二维位置与动作 ρ1、群元素维度 U-Net 权重共享；参数量 257,559,008（DP 288,378,562）。不是 FrameAvg。
- escnn fork 固定在 `fc4714cb6dc0d2a32f9fcea35771968b89911109`；C4 不需要 SO(3) py3nj 加速，未安装系统 gfortran。
- 小模型单步等变、反向传播、耦合噪声 DDPM 等变、EMA 保存/加载、仿真闭环通过；完整模型 GPU 前后向通过。TF32 关闭后最大等变误差 2.46e-6；新三臂统一关闭 TF32。
- **原生四朝向示教回放通过**：6 条留出示教 × 4 朝向 × 2 种 legacy 初始化；位置误差 ≤4.89e-11，coverage ≤1.2e-12，像素旋转误差 0。canonical 渲染与上游原版逐像素一致；存储录制重渲染仍有小差异（MAE 0.000602 / 单像素最大 0.443），已单独报告。
- **批量 100-episode 评估已实现**：25 基础 seed × 4 朝向、per-orientation、success、独立 episode RNG、逐批落盘/恢复；CPU 与 GPU smoke 通过。
- **保留三 checkpoint**：16k / 18k / 20k EMA；主统计取三点均值，另报 final 20k；配对 bootstrap 保留四朝向关联。
- **RQ3**：fixed/rotated goal、state 输入、显式 goal-input、零初始化 gate 已实现；八种配置 CPU smoke 与远端 GPU 预检通过。
- **C8**：作为架构消融实现，C4 子群单步/采样/梯度检查通过；**不能声称方形物理环境、像素旋转或 clipping 具有精确 45° 对称性**。
- **数据缓存**：本地每批 139 → 8.15 ms；2080 Ti 上 499 → 15 ms/batch；200-step 循环 43.8 s。
- **27 项本地测试通过**；原 `codes/` 基线与冻结版本 SHA256 审计无变更。

---

## 4. 协议修正（必须如实报告）

来自 `development/proposal_v2/PROTOCOL.md`：

1. 原场景像素/墙体中心为 255.5，原 goal 为 (256,256)。为保持 canonical 图像，goal 的**位置和角度一起旋转**；四个位置为 (256,256)/(256,255)/(255,255)/(255,256)，不是 proposal 写的固定 (256,256)。
2. 新 renderer 是"逆变换物理状态 → 原版渲染 → rot90"，不是证明原版 pygame 光栅化天生等变；物理在四个旋转坐标系中真实推进。
3. 实际最大训练量是 **200**（保留 6 条验证示教）。
4. coverage 是**交集 / 目标面积**，不是 IoU；success 沿用上游 `>0.95`。
5. 主统计用 16k/18k/20k 三个保留 checkpoint 的均值，另报 final；有限三点最大值不冒充完整 max-over-training。
6. 新训练统一关闭 TF32；旧 `codes/` run 不静默替换成新主表结果。

---

## 5. 未完成

- C4 主矩阵 36 runs 的实际结果（当前只有第一条在训练，尚无 coverage）。
- RQ3 24 runs 的完整结果（当前 1 条 state DP 完成，EquiDiff/goal/gate 未完成）。
- C8 6 runs（待 27693 预检）。
- 官方 DP 完整配方复现（当前仍是固定 20k 步的适配协议）。
- MimicGen sanity check / 可选 E4。
- RQ3 的 18 条图像实验（图像中 goal 可见，不能直接套用 state 的"看不到目标"假设）。
- 完整 max-over-training 指标。
- PPT 初稿/定稿/彩排；组存储配额未核实。

---

## 6. 风险

1. **队列时间**：C4 36 runs + RQ3 24 runs + C8 6 runs 仍在执行；11/24 冻结前可能跑不完，需要按实际完成度收窄主表。
2. **A100 吞吐未知**：27690 尚未完成；2080 Ti 缓存标定很好，但不能外推到 A100。
3. **RQ3 是 state 输入**：`fixed_state_dp` 的 0.7688 不能与图像主表直接比较；当前 EquiDiff state 评估还没完成。
4. **C8 不是精确 45° 物理对称**：只能作为架构消融，不能写成"环境 C8 精确增强"。
5. **两条线容易混淆**：`codes/` 的 FrameAvg 表是 fallback；`proposal_v2` 的 EquiDiff-C4 才是主方法。PPT 必须分表。
6. **存储**：checkpoint 4.6 GB/run；36+24+6 runs 的保留 checkpoint 和 3-checkpoint 评估会产生大量文件，组配额未核实。
7. **不要修改冻结 release**：运行中的队列源码被改动会破坏可复现性；新长期队列必须新建 release。

---

## 7. 下一步（按优先级）

1. **盯 C4 第一条 run**：00:40 左右训练完；检查 `runs/equidiff_c4_n50_s0/` 的 loss、EMA 16k/18k/20k、validation 和 100-episode test 是否正常；异常先停队列，不硬跑。
2. **检查 SICC**：27690 完成后回收 A100 缓存标定；27692 继续收集 RQ3 结果；27693 预检通过后才会进入 6 条训练。
3. **汇总脚本**：持续更新 `development/proposal_v2/results/summary/matrix_summary.json/csv`，只读同步远端结果，不传 checkpoint。
4. **PPT 素材**：把 fallback 表（已完成）和 C4 主表（进行中）分开；先做方法、协议修正、已完成验证、当前进度的页面，结果页留占位。
5. **11/24 结果冻结**：按实际完成的 runs 决定主表范围；未完成项写清"进行中/未完成"，不能用预期数字。
6. 有余量再补：官方 DP 配方、MimicGen sanity、RQ3 图像实验、完整 max-over-training。

---

## 8. 产物索引

| 内容 | 路径 |
|---|---|
| 原 `codes/` fallback 代码 | `codes/src/pusht_study/` |
| 原 `codes/` fallback 结果 | `codes/runs/pilot_*_20k_bal_local/`、`codes/runs/formal_*_s{1,2}_50/` |
| **新主线开发副本** | `development/proposal_v2/` |
| **新主线协议** | `development/proposal_v2/PROTOCOL.md`、`RQ3_PROTOCOL.md`、`C8_PROTOCOL.md` |
| 本地 C4 冻结 release | `development/proposal_v2/releases/c4_main_20261118_v1/` |
| RQ3 冻结 release | `development/proposal_v2/releases/rq3_state_20261118_v1/` |
| C8 冻结 release | `development/proposal_v2/releases/c8_ablation_20261118_v1/` |
| 监控 release | `development/proposal_v2/releases/monitor_20261118_v1/` |
| 实时状态 | `notes/live_status.md` |
| 监控 JSON | `development/proposal_v2/results/monitor/latest.json` |
| 汇总表 | `development/proposal_v2/results/summary/matrix_summary.json` / `.csv` |
| 远端结果镜像 | `development/proposal_v2/results/remote/` |
| 自动推进记录 | `notes/autonomous_20261118.md` |
| 复现证据 | `notes/reproduction.md` |
| 任务清单 | `notes/list.md` |
| 集群 | `notes/cluster.md` |
| 论文学习路线 | `notes/study_list.md` |
| 原始提案 | `03_Proposal_A_EquiDiff_final.md` |

---

## 9. 相关文档

- [任务清单](list.md)
- [复现情况](reproduction.md)
- [集群任务](cluster.md)
- [自动推进记录](autonomous_20261118.md)
- [实时状态](live_status.md)
- [论文学习路线](study_list.md)
- 汇报初始方案 (`../presentation/list.md`; local-only historical reference)
- [原始提案](../../../../reports/main_results_20261123/proposal_snapshot.md)
