> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `notes/list.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# 项目任务 list

> 2026-11-22 10:40：按用户指令，原暂停的 25/100-demo C4 收敛六条曲线迁移到旧集群。232837 已在两张 A40 上训练；232838 等待 12k 断点上传和 SHA256 校验后自动释放，释放后仍可能排队。本地 C4 自动续跑已禁用，5090 保持 Stack-D1。详见 [迁移记录](migrate_c4_legacy_20261122.md)。

> 2026-11-22 02:35：Stack-D1 正式训练已启动（5090，seed 0，500 epochs，batch 128）；C4 补充队列已安全暂停并设自动续跑。剩余 Stack 完整训练/评估/归档验证，详见 [启动记录](stack_d1_priority_20261122.md)。

> 2026-11-22：Stack-D1 环境与数据准备完成；剩余正式 batch/评估并行资源预检、冻结排队、完整训练与评估。预检不计入已完成实验，见 [准备记录](stack_d1_20261122.md)。

> 2026-11-19 21:00：已部署持续归档服务、每项实验归档成功后才继续的关卡，以及 RQ3/C8 恢复流程。最新接管步骤见 [恢复与归档记录](autonomous_20261119.md)。当前排队范围仍为 66 次训练，不代表完整 proposal 已全部安排。

> 2026-11-18 22:33：自动推进已新增 C4 原版架构适配、原生旋转验证、正式评估、RQ3 状态模型、C8 消融与 27 项测试；66-run 独立冻结队列正在执行/排队。最新完成项和未完成范围以 [自动推进记录](autonomous_20261118.md) 为准，动态状态见 [live_status.md](live_status.md)。下列早期勾选项尚未逐条迁移，不能据此判断实时阻塞。

更新：2026-11-18。用户已确认开始开发，正式 pre 日期为 **2026-11-26（下周四）**。本文件为当前执行清单；`presentation/list.md` 保留初始方案和实验设计细节。论文阅读路线见 [study_list](study_list.md)。当前进度总览见 [progress](progress.md)；新对话续接 prompt 见 [continue_prompt](continue_prompt.md)。

## 目标与时间线

核心：Push-T 上比较 DP、DP+C4 精确增强、C4 等变方法；优先原版 EquiDiff 移植，FrameAvg 为明确标注的备选。集群与本地共同推进，GPU 资源和排队时长仍待确认。

| 日期 | 目标 |
|---|---|
| 11/16–17 | 源码和数据准备、环境、CPU 正确性检查、GPU/集群接入与 DP pilot |
| 11/18–19 | PushT-Rot 数据/渲染对齐、原版 C4 移植决策、统一协议与成本标定 |
| 11/20 | 争取得到低数据量三方法的第一组有效结果 |
| 11/21–22 | 并行补种子与高数据量，完成 PPT 初稿 |
| 11/23–24 | 补齐核心矩阵；余量用于 RQ3；11/24 晚冻结正式汇报结果 |
| 11/25 | PPT 定稿、彩排、图表及引用核对 |
| 11/26 | pre |

正式矩阵目标：`{50, 200}` 条训练示教 × 3 方法 × 3 seeds = 18 次。200 条是从官方 206 条留出 6 条验证后的实际最大训练集。预算不足先保住 50 条下的 9 次训练。

## 已完成

- [x] 阅读并确认 EquiDiff / Push-T final proposal，阅读两套集群说明。
- [x] 创建 `codes/` 项目结构及 `notes/` 三份进度文档。
- [x] 下载两个官方仓库的固定 commit 源码，保留许可证和上游版本 manifest。
- [x] 下载并校验官方 Push-T 数据：206 episodes / 25,650 frames。
- [x] 建立项目局部虚拟环境，缺失依赖下载、校验并安装到项目内。
- [x] 固定 trajectory 级训练/验证划分、基础朝向和预留测试种子。
- [x] 实现 C4 图像/坐标/动作变换、各向同性归一化、数据读取。
- [x] 实现 DP 风格 pilot、Frame Averaging 模型、训练保存/恢复和评估入口。
- [x] 编写原生 PushT-Rot 环境原型与对称性诊断。
- [x] 编写两套集群配置、上传工具、预检/pilot Slurm 脚本及环境安装脚本。
- [x] 创建首次本地 Git 提交 `3550ad3`，配置用户指定 GitHub 仓库为 origin，补齐源码/数据恢复脚本和公开进度文档。
- [x] 验证干净本地克隆可以恢复官方源码与数据，SHA256 校验和 7 项测试通过。
- [x] 推送至 GitHub：用户在宿主机终端完成，已再次核实远端 main 为 `3550ad3c705b33e16504c9762dc55b161a5ccb2a`。
- [x] 基础单元测试通过：坐标/像素方向、群运算、数据分割、序列边界、FrameAvg 等变与反向传播。
- [x] 真实数据 CPU tiny-model 训练、checkpoint 加载与原版环境 rollout 打通；仅为工程冒烟，不是性能结果。
- [x] 断点恢复集成测试通过：连续 4 步与 2 步保存后恢复到 4 步的模型/EMA 权重逐位一致。
- [x] DDPM 多步采样在耦合随机噪声下的 C4 等变测试通过。

## 当前开发与验收

- [x] SICC robotic 代码/数据部署、独立 Python 3.10.19 + PyTorch 2.7.1/CUDA 12.8 环境安装完成；单卡 A100 预检作业 27159 COMPLETED，完整模型及 7 项测试通过。
- [x] 2026-11-17：提交 SICC 标定作业 27450（dp）和 27451（dp_c4_aug），队列预计 11/18 早晨启动；FrameAvg 标定在本地 RTX 5090 后台运行。账号 `MaxSubmitPA=2`，第三个集群作业待前两个完成后提交。
- [x] 2026-11-17：完成本地三臂 200 步标定（`dp` / `dp_c4_aug` / `dp_c4_frameavg`，batch 64，`[512,1024,2048]`）：吞吐 5.11 / 4.13 / 3.51 updates/s，峰值显存 6.54 / 6.54 / 8.13 GiB，checkpoint 4.61 GB；100-episode 评估外推约 42 / 33 / 144 min。A100 对照由 27450/27451 提供。
- [x] 2026-11-17：DP 20k/40k 学习曲线完成；10 个验证 seed 上 20k mean coverage 0.4336、40k 0.4033，loss 继续下降但 coverage 进入平台，**正式预算暂定 20k 步**。DP+C4 aug 20k 与 FrameAvg 20k 在本地顺序运行。
- [x] 2026-11-18：三臂 20k pilot（canonical，seed 0，10 seeds）：`dp_c4_aug` 0.681 > `dp` 0.434 > `dp_c4_frameavg` 0.349；但 canonical-only 与 proposal 的平衡朝向协议不一致，FrameAvg 被强制错误等变，**不能据此支持 H2**。下一步解锁 `rotated=True` 基础朝向后重跑同预算比较。
- [x] 2026-11-18：解锁 `rotated=True` 的 balanced 基础朝向（`allow_rotated` 显式开关；新增数据集旋转测试，8 项测试通过；CPU smoke 通过）；已启动 balanced 三臂 20k pilot，预计约 4.5 小时。
- [x] 2026-11-18：SICC 标定作业 27450/27451 完成：A100 训练约 0.46 updates/s、评估约 110 s/episode，比本地 5090 慢约 9×/4.5×。当前实现下 18-run 正式矩阵在 SICC 不可行；需 zarr 内存/节点本地缓存优化，或把正式训练放本地。
- [x] 2026-11-18：balanced DP seed 0 = 0.3179（canonical 0.4336）；普通 DP 在平衡朝向下变差。正式 50-demo 9-run 已启动：seed 0 复用 balanced pilot，seed 1/2 共 6 条在本地 5090 运行（`configs/matrix/`，`runs/formal_*`）。
- [x] 2026-11-18：`allow_formal` 显式开关已加；8 项测试 + CPU smoke 通过。正式 100-episode 旋转评估仍待实现，当前 interim 评估用 10 个验证 seed。
- [x] 2026-11-18：balanced 三臂 seed 0（10 seeds）：`dp_c4_aug` **0.6772**（min 0.456）> `dp` 0.3179 > `dp_c4_frameavg` 0.2648；balanced 数据未救回 FrameAvg。formal seed 1 的 DP 只有 0.1957，跨训练种子方差大，**必须等 3-seed 矩阵再下结论**。
- [x] 2026-11-18：formal 50-demo 部分结果：DP 3-seed mean **0.2170**（0.318/0.196/0.138）；aug 2-seed mean **0.6029**（0.677/0.529）；FrameAvg 2-seed mean **0.2546**（0.265/0.244）。排名稳定，等 seed 2 的 aug/FrameAvg。
- [x] 2026-11-18：A100 FrameAvg 标定 27582 完成（200 步 342 s，评估 225 s，112.5 s/episode）；三臂 A100 数据齐全。
- [x] 2026-11-19：原 `codes/` 的 50-demo 3-seed fallback 表完成：`dp` 0.2171 / `dp_c4_aug` 0.5963 / `dp_c4_frameavg` 0.3041；aug 三种子均最高，FrameAvg/DP 互有胜负。该表与 `development/proposal_v2/` 的 EquiDiff-C4 主表分开，不能混用。
- [x] 关键评估教训：2 episodes 会严重误导（20k DP 在 2 seeds 上 0.051，10 seeds 上 0.4336）；pilot 至少 10 seeds，正式 100 episodes。
- [x] 13:45 后重新验证本地 nvidia-smi、项目环境 CUDA 实际前后向和两套集群 SSH，当前会话访问已恢复。
- [x] 完整 pilot 模型 GPU 预检、100 步 DP 风格 pilot 和 2 个短 rollout 完成；峰值分配显存 2.09 GiB，详见 reproduction.md。原版 DP 配方复现仍待完成。
- [ ] 原版 DP 的完整 GPU pilot 和学习曲线检查；现有自编 pilot 不等同于上游完整训练配方复现。
- [ ] 原版 EquiDiff-C4 迁移到 2D 动作与 Push-T 图像；escnn/gfortran 仍未配置。
- [ ] 修复原生渲染像素误差，完成示教数据与新目标中心的对齐、demo open-loop replay。
- [x] 验证整个扩散采样器在耦合噪声下的等变性（CPU tiny 模型）；正式模型的 GPU 检查仍待运行。
- [ ] 冻结正式训练步数和模型规模；本地 200 步三臂标定已完成，待 A100 对照作业 27450/27451 完成后决定。
- [ ] 正式训练调度与统一统计报告；当前矩阵标记 blocked，不批量提交无效任务。
- [ ] 目标 18-run 完成；生成 coverage/成功率、每朝向指标、种子方差和失败案例。
- [ ] 有余量再做 RQ3 的 state + goal-input 三方法比较。
- [ ] PPT 初稿、定稿与彩排。

## 当前阻塞

1. GPU/SSH 与 SICC 基础环境阻塞已解除；原版 EquiDiff 的 escnn/gfortran 依赖尚待准备，组剩余存储配额仍未核实。当前环境已验证的范围为 DP 风格 pilot 和 FrameAvg，尚无正式训练结果（0/18）。
2. 原生环境物理旋转诊断误差很小，但像素并非严格等变；现阶段不能把旋转图像直接当作已验证的原生环境精确增强。
3. 原版 EquiDiff 尚未移植；FrameAvg 的可运行代码只是备选，不冒充原论文复现。

详情和证据见 [复现情况](reproduction.md)，集群状态见 [集群任务](cluster.md)。
