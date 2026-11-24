> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `notes/reproduction.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# 复现情况

> 2026-11-22 10:40：按用户指令，原暂停的 25/100-demo C4 收敛六条曲线迁移到旧集群。232837 已在两张 A40 上训练；232838 等待 12k 断点上传和 SHA256 校验后自动释放，释放后仍可能排队。本地 C4 自动续跑已禁用，5090 保持 Stack-D1。详见 [迁移记录](migrate_c4_legacy_20261122.md)。

> 2026-11-22 02:35：Stack-D1 完整 batch-128 预检通过，正式 500-epoch seed-0 训练已在本地 5090 启动。运行和证据入口见 [启动记录](stack_d1_priority_20261122.md)。

> 2026-11-22：Stack-D1 参考复现的独立环境、数据下载与完整动作转换完成；GPU 完整模型 batch-1 和官方 workspace 冒烟通过。正式训练未启动，运行环境兼容差异及证据见 [Stack-D1 准备记录](stack_d1_20261122.md)。

> 2026-11-19：新增逐实验备份校验/清理关卡（7 项保护与恢复测试通过）。节点异常和存储配额中断后的处理见 [恢复与归档记录](autonomous_20261119.md)。RQ3 后续任务改用 A100，需保留硬件差异说明；科学协议与冻结训练源码未改。

## 2026-11-18 22:33：独立版本自动推进

原版架构 EquiDiff-C4 的 Push-T 适配已通过小模型与完整 GPU 前后向、C4 数值等变、EMA 保存/加载和 rollout 检查。原生四朝向示教动作回放通过，批量 100-episode 评估、保留三 checkpoint、配对 bootstrap 已实现。RQ3 四种状态模型与 C8 架构消融也已实现；27 项本地测试通过。已安排的 36 + 24 + 6 次训练仍在执行/排队，不能当作已完成的性能结果。所有实现位于独立开发与冻结 release，原 `codes/` 未改。

证据、队列、明确的 proposal 修正与尚未完成范围见 [自动推进记录](autonomous_20261118.md)；实时状态见 [live_status.md](live_status.md)。以下保留早期复现历史。

## 2026-11-17：本地三臂标定（工程标定，不是性能结果）

在 RTX 5090 上用同一份 50 demos、batch 64、U-Net `[512,1024,2048]` 配置各跑 200 步，评估 2 episodes × 300 步：

| 方法 | 训练耗时（200 步） | updates/s | 峰值显存（GiB） | loss 前10/后10 | 2×300 评估 | 100 episodes 外推 | checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dp` | 47 s | 5.11 | 6.54 | 1.694 / 0.085 | 50 s | ~42 min | 4.61 GB |
| `dp_c4_aug` | 55 s | 4.13 | 6.54 | 1.719 / 0.091 | 39 s | ~33 min | 4.61 GB |
| `dp_c4_frameavg` | 66 s | 3.51 | 8.13 | 0.429 / 0.059 | 172 s | ~2.4 h | 4.61 GB |

- 200 步后的 coverage（`dp` 0.086 / `dp_c4_aug` 0.117 / `dp_c4_frameavg` 0.525）只是工程信号，**不能作为性能结果**。
- 模型参数量 288.4 M；`[512,1024,2048]` 的 checkpoint 约 4.6 GB。
- 评估是主要成本：FrameAvg 每个 100-episode checkpoint 约 2.4 h（5090），正式评估需要并行或先做子集筛选。
- 18 个 run 的 checkpoint 约 83 GB，组配额尚未核实；正式矩阵前应考虑只保留 EMA/最终 checkpoint。
- 同日提交 SICC 标定作业 27450（`dp`）/ 27451（`dp_c4_aug`），队列预计 11/18 早晨启动；FrameAvg 因账号 `MaxSubmitPA=2` 暂未提交集群。

### 10-seed 复评（关键：2 episodes 会严重误导）

用保留验证 seed 900000–900009 重评四个 checkpoint（300 步）：

| checkpoint | mean max coverage | min | max | success |
|---|---:|---:|---:|---:|
| DP, 20k 步 | **0.4336** | 0.000 | 0.949 | 0 |
| DP, 200 步 | 0.1366 | 0.000 | 0.496 | 0 |
| DP+C4 aug, 200 步 | 0.1441 | 0.000 | 0.496 | 0 |
| FrameAvg, 200 步 | 0.3179 | 0.000 | 0.590 | 0 |

- 20k DP 在最初 2 个 seed（900000/900001）上只有 0.051，但在 10 个 seed 上是 0.4336，最高 0.949。**2-episode 评估不能用于方法比较或预算决策。**
- 200 步时 FrameAvg 的平均 coverage 高于 DP/Aug，但这仍不是同预算比较；需要 20k 步的 FrameAvg/Aug 结果。
- 所有方法在 10 个验证 seed 上 success 都是 0（阈值 `>0.95`），最高 coverage 0.949 非常接近阈值；正式测试的 25×4 episodes 仍需观察。
- 结论：当前 pilot 配方**能学到东西**，可以继续；但 pilot 评估至少用 10 个 seed，正式评估用 100 episodes。
- 40k 续训（同 seed/数据）：mean 0.4033，success 0.10，max 0.960；对比 20k 的 0.4336 / 0 / 0.949。loss 从 20k 的 0.0124 继续降到 40k 的 0.0096，但 coverage 没有提升。**正式预算暂定 20k 步**，40k 不作为默认预算；这个平台期检查会写进报告。

### 三臂 20k pilot（seed 0，10 个验证 seed，canonical 数据）

| 方法 | mean max coverage | min | max | success | 20k loss |
|---|---:|---:|---:|---:|---:|
| `dp` | 0.4336 | 0.000 | 0.949 | 0 | 0.0124 |
| `dp` 40k | 0.4033 | 0.000 | 0.960 | 0.10 | 0.0096 |
| `dp_c4_aug` | **0.6810** | **0.515** | 0.941 | 0 | 0.0207 |
| `dp_c4_frameavg` | 0.3489 | 0.000 | 0.863 | 0 | 0.0123 |

- 表面结果：`dp_c4_aug` > `dp` > `dp_c4_frameavg`。aug 是唯一所有 10 个 seed 都 ≥0.515 的方法，说明在线精确 C4 增强显著提高了跨初态的稳定性。
- **但这个比较有 confound，不能据此支持 H2**：当前所有 run 都是 `rotated=False`，即三臂只看到 canonical 朝向的数据；而 proposal 要求每条示教分配固定基础朝向、四方向均衡、三臂共享。aug 通过在线增强获得了 C4 对称分布，DP/FrameAvg 没有；FrameAvg 还被强制 C4 等变，但 canonical 数据分布本身不是 C4 对称的——这属于"错误等变"场景，按理论就应该受伤。
- `low loss ≠ good policy`：aug 的 20k loss 最高（0.0207）但 coverage 最好；FrameAvg loss 最低（0.0123）但 coverage 最差。
- 下一步必须先解锁 `rotated=True` 的基础朝向数据，让三臂共享同一个平衡朝向数据集，再做同预算比较；否则 RQ2 的结论无效。

### 2026-11-18：解锁 balanced 基础朝向

- `train.py`：`rotated=True` 仍需显式 `allow_rotated=true`；默认保持锁定。`rotated` 的含义是"每条示教固定基础朝向、四方向均衡"，与原生旋转评估不同。
- `evaluate.py`：允许把 balanced-trained checkpoint 放回原版 canonical 环境评估；原生旋转评估仍由 `native_rotated_eval` 单独控制并保持未实现。
- 新增测试 `test_rotated_dataset_applies_fixed_base_orientation`：验证 `rotated=True` 的数据项等于 canonical 数据项按该 episode 的基础朝向旋转；8 项测试通过。
- CPU smoke（2 步 tiny + 1×8 rollout）通过，确认 rotated 训练 → checkpoint → 原版环境评估闭环可用。
- 已启动 balanced 三臂 20k pilot（seed 0，10 个验证 seed）：`runs/pilot_dp_20k_bal_local`、`runs/pilot_aug_20k_bal_local`、`runs/pilot_frameavg_20k_bal_local`。目的：消除 canonical-only confound，让三臂共享同一份平衡朝向数据，再做 RQ2 比较。预计约 4.5 小时完成。

### 2026-11-18：balanced DP seed 0 与正式 9-run 启动

- balanced DP 20k（seed 0，10 seeds）：mean **0.3179**，max 0.588；低于 canonical-only 的 0.4336。普通 DP 在四朝向平衡数据上需要从图像推断目标朝向，难度更高。
- balanced `dp_c4_aug` / `dp_c4_frameavg` seed 0 仍在运行（aug 已接近 20k，FrameAvg 排队）。
- 正式 50-demo 9-run：seed 0 复用 balanced pilot 的三条 run；seed 1/2 共 6 条已启动，配置在 `configs/matrix/`，输出 `runs/formal_*_s{1,2}_50/`。
- 协议：50 demos、20k 步、batch 64、`[512,1024,2048]`、`rotated=true` 平衡朝向、3 seeds；当前 interim 评估 10 个验证 seed，正式 100-episode 旋转评估待实现。
- `train.py` 增加 `allow_formal` 显式开关；`formal=true` 必须同时 `allow_formal=true`。8 项测试与 CPU smoke 均通过。
- 预计 6 条 seed 1/2 训练 + interim 评估约 9–10 小时，与 balanced FrameAvg seed 0 并发。

### 2026-11-18：balanced 三臂 seed 0 结果（10 个验证 seed）

| 方法 | mean max coverage | min | max | 20k loss |
|---|---:|---:|---:|---:|
| `dp` | 0.3179 | 0.000 | 0.588 | — |
| `dp_c4_aug` | **0.6772** | **0.456** | 0.949 | 0.0210 |
| `dp_c4_frameavg` | 0.2648 | 0.000 | 0.949 | 0.0120 |

- augmented 仍然最强，且 10 个 seed 全部 ≥0.456；balanced 数据没有救回 FrameAvg（0.349 → 0.265），普通 DP 也从 0.434 降到 0.318。aug 相对 DP 的优势从 canonical 的 +0.247 扩大到 +0.359。
- **不能把 seed 0 当成结论**：formal seed 1 的 balanced DP 只有 **0.1957**，跨训练种子方差很大。必须等 3-seed 矩阵。
- FrameAvg 的 loss 最低但 coverage 最低，说明优化目标与闭环表现脱节；在把它当作第三臂结论前，需要排查实现/推理 averaging 的影响，或回到原版 EquiDiff 移植。
- 200 步 A100 FrameAvg 标定（27582）完成：训练 342 s、评估 225 s（112.5 s/episode）、checkpoint 4.61 GB；A100 节点间有波动，但仍远慢于本地 5090。

### 2026-11-18：formal 50-demo fallback 表（seed 0/1/2，最终）

| 方法 | seed 0 | seed 1 | seed 2 | 3-seed mean |
|---|---:|---:|---:|---:|
| `dp` | 0.3179 | 0.1957 | 0.1375 | **0.2171** |
| `dp_c4_aug` | 0.6772 | 0.5286 | 0.5832 | **0.5963** |
| `dp_c4_frameavg` | 0.2648 | 0.2444 | 0.4030 | **0.3041** |

- seed 0 为 balanced pilot 的三条 run；seed 1/2 为 `configs/matrix/` 的正式 run。协议相同。
- `dp_c4_aug` 在三个训练种子上都最高；seed 2 有一个 episode 成功（900007，0.9505），success 0.10；其余 run success 0。
- FrameAvg 与 DP 互有胜负：seed 0 是 DP 更高，seed 1/2 是 FrameAvg 更高；3-seed 均值 aug > FrameAvg > DP，但 FrameAvg/DP 的差距在方差范围内，不能声称稳定排序。
- `dp_c4_frameavg` 的 loss 最低但 coverage 最低，保留"实现/优化待排查"标记；它不是 EquiDiff 的等价替代。
- 这是 10 个 canonical 验证 seed 的 interim，**不是** proposal 的 100-episode 旋转正式评估；与 `development/proposal_v2/` 的 EquiDiff-C4 主表分开。

### 2026-11-18：SICC A100 标定结果（27450/27451 完成）

| 环境 | 方法 | 200 步训练 | updates/s | 2×300 评估 | 每 episode | checkpoint |
|---|---|---:|---:|---:|---:|---:|
| SICC A100-40GB | `dp` | 431 s | 0.46 | 226 s | 113 s | 4.61 GB |
| SICC A100-40GB | `dp_c4_aug` | 428 s | 0.47 | 221 s | 110 s | 4.61 GB |
| RTX 5090 | `dp` | 47 s | 4.25 | 50 s | 25 s | 4.61 GB |
| RTX 5090 | `dp_c4_aug` | 55 s | 3.64 | 39 s | 19.5 s | 4.61 GB |

- **A100 训练约慢 9×，评估约慢 4.5×**。GPU 本身的算力不足以解释这个差距；更可能是 `/data` NFS 上 zarr 的随机读取（819 个 chunk 文件）、共享节点 CPU 竞争，以及 CPU-bound 的 pygame/pymunk 环境。
- 按 A100 的 0.46 updates/s，20k 步单次训练约 12 小时；18-run 正式矩阵在 SICC 当前实现下不可行。
- 需要先优化数据加载（例如把 zarr 预加载进内存或复制到 `$SLURM_TMPDIR`），或者把正式训练放到本地 5090；A100 不适合直接跑当前的自编 pilot。
- 27450/27451 的 200 步 coverage（`dp` 0.078 / `dp_c4_aug` 0.156，2 seeds）只是工程信号，不能与本地 10-seed 结果比较。
- FrameAvg 的 A100 标定尚未提交；集群队列已空，可以补提交。

产物：`codes/runs/calib_dp_b64_d512_local/`、`codes/runs/calib_aug_b64_d512_local/`、`codes/runs/calib_frameavg_b64_d512_local/`；配置为 `codes/configs/calib_*_b64_d512.json`。

## 2026-11-16：开始开发

### SICC 独立环境部署与 A100 预检

代码与官方数据已上传至 `${SICC_PROJECT}`，522 个源码文件、819 个数据文件校验通过。项目独立 Python 3.10.19 和全部新增依赖均位于 `robotic/envs/`，最终 venv 的 `include-system-site-packages = false`。固定 PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128；依赖一致性检查通过，完整清单已保存。没有改动其他环境或停止其他任务。

预检作业 **27159** 在 `um-a100-02` 的 NVIDIA A100-SXM4-40GB 上运行，状态 COMPLETED、退出码 0:0、耗时 51 秒。完整 pilot 模型前后向耗时约 2.315 秒，单步 C4 等变最大误差 1.490e-8，七项测试全部通过。单元测试仍以现有 CPU 测试为主，不将此记录解释为完整 DDPM 采样器的 GPU 等变检查。

安装日志、环境清单、预检 JSON、Slurm 日志、提交时的源码清单及部署索引均回收至 `codes/results/sicc_deployment_20261116/`。工作站使用开发版 PyTorch，集群使用稳定版，两者结果明确分开；尚未运行集群 pilot 训练，更没有正式三方法比较结果。当前就绪范围为 DP 风格 pilot / FrameAvg，原版 EquiDiff 的 escnn/gfortran 与模型移植仍未完成。

### 13:45 后：访问恢复与 GPU pilot

重新阅读 PATHS.md 和三份进度文件后，当前会话已成功运行 `nvidia-smi`、项目 Python CUDA 实际前后向，以及两套集群的只读 SSH 检查。RTX 5090 / 驱动 580.178.04 正常；本地 Python 为 `codes/envs/local/bin/python`，继续只读继承 PyTorch `2.12.0.dev20260407+cu128`，CUDA runtime 12.8。本轮未安装任何依赖，未修改其他项目环境，未停止已有训练进程；GPU 进程 PID 4079267 在 pilot 完成后仍存在。

- 完整 ResNet18 + `[256, 512, 1024]` U-Net 的 FrameAvg GPU 前后向通过，耗时约 1.107 秒，单步 C4 等变最大误差 `2.235e-8`。报告：`codes/results/preflight_full_gpu_20261116.json`。这不是完整 DDPM 采样器的 GPU 等变检查。
- 使用原有 `configs/pilot_dp.json` 完成 50 demos / seed 0 / batch 16 / 100 更新步；输出 `codes/runs/dp_gpu_pilot_20261116/`，状态 complete，checkpoint 为 `last.pt`。
- 前 10 步平均 loss `1.168895`，后 10 步 `0.117876`；峰值 PyTorch 分配显存 `2.093 GiB`，不含其他进程和 CUDA 上下文占用。第 11–49 步约 `15.96 updates/s`。到最后一次更新耗时 `8.110 s`，含中途 checkpoint、不含最终 checkpoint；这只是短 pilot 标定，不能外推为完整训练收敛预算。汇总见该运行目录的 `calibration.json`。
- EMA checkpoint 加载后完成原版环境 2 episodes × 30 步短 rollout，使用预留 validation seeds 900000/900001；平均最大 coverage `0.069664`，成功率 0。结果为 `evaluation_smoke.json`，仅证明 GPU 训练、保存、加载、采样、仿真闭环可运行，不作为方法性能结果。
- 两套集群 SSH 与队列读取成功；远端尚未创建 robotic、安装环境或提交 Slurm 作业。SICC 默认 PATH 的 Python 为 3.12.3，需解决项目专属 Python 3.10 环境。具体证据及剩余配额检查见 cluster.md。

原版 DP 完整配方、原版 EquiDiff 移植、原生渲染/数据对齐和正式矩阵仍未完成。以下 CPU 与访问失败记录保留为历史证据。

### 13:12 后的资源更正与发布准备

用户提供的 `nvidia-smi` 确认宿主机 RTX 5090 和驱动正常，当前占用 5223/32607 MiB，已有其他项目 Python 进程。此前 CUDA/SSH 失败是当前执行会话的访问限制，不应继续描述为显卡或驱动故障。保留现有进程及环境。

用户指定 GitHub 仓库：`https://github.com/KeruZheng/LIU_MU_ZHENG_8001_FINAL.git`。已生成首次本地提交 `3550ad3`，包含 41 个文件；环境、数据、checkpoint 不入 Git，真实集群配置迁移到忽略文件 `configs/clusters.local.json`，公开仓库使用示例配置。新增恢复官方固定版本源码和数据的 `scripts/fetch_artifacts.py`，以及 `docs/notes/` 的公开进度快照。

首次发布状态：用户在宿主机终端执行推送成功；随后用 `git ls-remote origin refs/heads/main` 核实远端为 `3550ad3c705b33e16504c9762dc55b161a5ccb2a`，与本地一致，**GitHub 上传已完成**。此前 agent 的推送受执行环境限制，记录为已由用户终端解决；集群 SSH 访问仍未解决。只配置本项目 Git 作者为 GitHub 用户名及公开 noreply 邮箱，没有改动全局 Git 配置。干净本地克隆已完成官方源码/数据恢复、逐文件 SHA256 校验及全部 7 项测试，记录在 `codes/results/clean_clone_validation.txt`。

环境隔离已再次核实：新增 `pymunk` 位于 `codes/envs/local/lib/python3.10/site-packages`，原 `modelscope_5090` 环境查询仍不存在该包；PyTorch 从原环境只读复用。

本轮目标是把项目落实为保存于 `codes/` 的可检查实现、真实数据与运行记录，并准备集群迁移。**尚无正式训练结果，不能报告已复现论文性能。**

早期 `codes/` 初始化后曾因缺少 Git 作者身份未能提交；此问题后来通过仅配置本项目身份解决，首次提交和推送状态以上述 `3550ad3` 记录为准。

### 上游与数据

| 项目 | 固定版本 / 实测情况 |
|---|---|
| Diffusion Policy | `5ba07ac6661db573af695b419a7947ecb704690f` |
| EquiDiff | `e40abb003b25071d4e5b01bfa9933dc16cd32c67` |
| 源码形式 | 官方固定 commit 的 tar.gz，解压到 `codes/upstream/`；保留许可证，未更改上游文件；没有 Git 历史 |
| 官方 Push-T | 206 episodes，25,650 frames；图像 96×96×3，动作 2D |
| 数据包 SHA256 | `63d52a114a3f010861f0181309d165b7d69133ccae426ece2fc94caed147bdf9` |
| 划分 | 200 条训练、6 条验证；50 条训练集合是 200 条的固定前缀 |
| Manifest | `codes/manifests/upstream.json`、`data.json`、`split.json`、`evaluation_seeds.json` |

### 实现边界

- `src/pusht_study/geometry.py`：图像 `rot90` 与 y 向下坐标一致，世界中心 255.5，xy 共同尺度归一化。
- `data.py`：按轨迹采样序列，边界复制不跨 episode；基础朝向由 manifest 固定。
- `models.py`：直接复用官方 DP `ConditionalUnet1D`；普通 ResNet18 + GroupNorm 编码；显式 C4 Frame Averaging 备选。
- `train.py` / `evaluate.py`：固定更新步数的 pilot 训练，EMA、优化器与随机状态保存，显式恢复；评估原版 Push-T 的 coverage。
- `env.py`：原生 PushT-Rot 原型，目标/初态旋转中心对齐；仍需示教和渲染验证。

当前 pilot 使用简化的训练实现：无图像随机裁剪，ResNet 全局池化，较小 U-Net 宽度、固定学习率、带放回序列采样。**这是用于打通流程和标定的 DP 风格实现，不是官方 DP 完整配方或原版 EquiDiff 的性能复现。** 正式实验协议尚未冻结，代码主动阻止 formal 和 rotated 训练。

### 环境

本地 `codes/envs/local` 是 `--system-site-packages` 虚拟环境，继承现有 `modelscope_5090` 的 PyTorch `2.12.0.dev20260407+cu128`，不修改原环境；新增 Push-T 依赖仅安装在项目环境。该环境依赖本机基座，不能直接复制到集群。

pip 在线依赖解析失败后，使用 PyPI 元数据定位当前 Python/平台的安装包，逐个下载并核验 SHA256，再离线安装。安装包及元数据在 `codes/envs/`；最终版本清单记录在 `codes/manifests/local_environment.txt`。

### 已有验证证据

`PYTHONPATH=src envs/local/bin/python -m unittest discover -s tests -v`：7 个测试通过，覆盖像素/坐标旋转方向、C4 组合与归一化、轨迹划分、序列边界、FrameAvg 等变性和梯度、真实数据断点恢复一致性，以及耦合随机噪声下的 DDPM 多步采样等变性。

CPU tiny 模型完整闭环已跑通：`codes/runs/smoke_cpu/` 保存 2 步真实数据训练日志、checkpoint 和 2 个 episode × 8 步的仿真评估。成功率为 0；如此短的工程测试没有性能解释意义，不能放入正式方法比较表。独立恢复测试比较连续 4 步和 2+2 步，模型/EMA 权重逐位相同。

`codes/results/preflight_cpu.json` 记录 CPU 前后向检查通过，FrameAvg 单步最大等变误差约 `2.98e-8`；GPU 可用性仍为 False。

额外的非 tiny 模型 CPU 检查也已通过：完整 ResNet18 + `[256, 512, 1024]` temporal U-Net 的 FrameAvg 前后向可运行，最大等变误差约 `1.49e-8`，报告为 `codes/results/preflight_full_cpu.json`。该宽度仍是当前 pilot 配方，不能等同于官方完整基线。

`codes/manifests/data.sha256` 记录 819 个解压后数据文件的校验值；`code.sha256` 记录 521 个项目及上游源码文件，逐项校验已通过。验证索引在 `codes/manifests/verification.json`。

`scripts/check_symmetry.py`：5 个随机初态，每个 20 个控制步，逐个比较 3 个旋转方向：

| 指标 | 最大误差 |
|---|---:|
| T 形物体位置 | `3.41e-13` |
| coverage | `6.94e-16` |
| 图像平均绝对误差（0–1） | `0.001399` |
| 图像单像素最大误差（0–1） | `0.72549` |

结果文件：`codes/results/symmetry_native.json`。这是短随机控制诊断，不是全部示教回放或对精确物理对称性的证明。像素误差说明渲染路径仍需处理，不能由物理状态误差小推断精确图像增强成立。

### 资源状态与下一步

本地 CUDA 和两端 SSH 均验证通过；本地完整 pilot 模型 GPU 预检及 100 步训练/短 rollout 已完成。SICC 已完成项目部署、独立环境安装和 A100 预检作业 27159。集群相关进度单独维护在 `cluster.md`。

下一步顺序：集群 DP pilot 成本标定与原版 DP 配方验证 → 原版 EquiDiff 移植和原生图像数据对齐 → 冻结正式矩阵 → 集群批量实验。本地 CPU/GPU 工程闭环及 SICC 基础环境验证已完成。
