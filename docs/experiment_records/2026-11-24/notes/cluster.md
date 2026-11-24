> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `notes/cluster.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# 集群任务

> 2026-11-22 10:40：按用户指令，原暂停的 25/100-demo C4 收敛六条曲线迁移到旧集群。232837 已在两张 A40 上训练；232838 等待 12k 断点上传和 SHA256 校验后自动释放，释放后仍可能排队。本地 C4 自动续跑已禁用，5090 保持 Stack-D1。详见 [迁移记录](migrate_c4_legacy_20261122.md)。

> 2026-11-19 21:00：旧 RQ3 27692/C8 27693 已失败；RQ3 新作业 27809 在 um-a100-02 通过 CUDA/存储预检并续评估。um-gpu03 报告 GPU 数 0 < 7，被标记 DRAIN+INVALID_REG；um-gpu04 的恢复预检也失败，均已避开。归档关卡、C8 延迟恢复及新监控说明见 [最新恢复记录](autonomous_20261119.md)。以下旧作业状态为历史快照。

## 2026-11-18 22:33 自动推进更新

- **27691 / gpu_batch / RTX 2080 Ti**：缓存标定完成，ExitCode 0:0。200-step DP 循环 43.8 秒，原读取/缓存读取分别约 499/15 ms 每批。
- **27692 / gpu_batch**：RQ3 状态矩阵 24 runs 正在运行。独立冻结路径 `robotic/releases/rq3_state_20261118_v1/`，独立新增依赖环境 `robotic/envs/rq3/`，只读引用原 PyTorch 环境。Slurm 内预检已通过。
- **27690 / a100_batch**：缓存 DP 标定排队。独立冻结路径 `robotic/releases/cache_calib_20261118/`。
- **27693 / a100_batch**：6-run C8 消融已提交，依赖 27690 成功，作业内先做完整 GPU/采样等变检查和 200-step 标定。路径 `robotic/releases/c8_ablation_20261118_v1/`。
- 原远端源码没有更新；新增任务仅使用独立 release。没有取消其他作业。组剩余存储配额仍未核实。
- 自动同步的无 checkpoint 结果在 `development/proposal_v2/results/remote/`；实时状态见 [live_status.md](live_status.md)，完整记录见 [autonomous_20261118.md](autonomous_20261118.md)。以下为原历史记录。

## 当前状态（2026-11-16）

### 集群环境准备（已完成基础环境及预检）

SICC 的 `${SICC_PROJECT}` 已创建，522 个源码文件、819 个数据文件上传及 SHA256 校验通过。安装脚本增加项目内 Python 3.10 引导流程：仅在 `robotic/envs/bootstrap` 安装 uv 0.9.18，再下载解释器到 `robotic/envs/python`，最终环境位于 `robotic/envs/robotic`，不继承系统或其他项目 site-packages；缓存及临时目录也限制在项目内。

环境：Python 3.10.19、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、NumPy 1.26.4；`pip check` 通过。首次安装因引导 Python 自带旧 pip 的包名元数据解析失败，已在项目环境内固定 pip 25.3 / setuptools 75.8.2 / wheel 0.45.1 后解决。没有修改系统或其他项目环境。

**Slurm 预检 27159：COMPLETED，ExitCode 0:0，耗时 51 秒。** 分区 a100_batch，节点 um-a100-02，NVIDIA A100-SXM4-40GB；申请 1 GPU / 8 CPU / 40G 主机内存。完整模型 CUDA 前后向通过，单步 C4 等变误差 1.490e-8，7 项单元/集成测试全部通过。作业完成后个人队列为空。本轮未提交 pilot 训练或正式矩阵，原版 EquiDiff 的额外依赖仍待准备。

环境清单位于远端 `manifests/cluster_environment.{json,txt}`；预检输出 `results/preflight_27159.json`、`logs/slurm/preflight_27159.out`。本地证据汇总在 `codes/results/sicc_deployment_20261116/`，包含提交时的源码清单快照、部署记录、安装日志、环境清单与预检输出。

实时查询：`squeue -A ${CLUSTER_ACCOUNT}` 为空；`a100_qos` 显示 MaxJobsPU=3、MaxSubmitPU=3、MaxTRESPU=gres/gpu=4。仅申请一个预检作业，不将这些上限解释为独享资源。`xfs_quota` 无法查询 NFS `/data`，因此组剩余存储配额仍未核实；官方文档记载 `/data` 为组共享 1 TB。不得把文件系统剩余空间当成组配额。

### 访问恢复检查

**13:45 后复核：当前会话已可访问本地 GPU 和两套集群 SSH。** 以下早期访问限制仅为历史记录，不再是当前阻塞。两端 `squeue -u ${CLUSTER_USER}` 均为空，两个计划的 `robotic/` 目录均不存在；本轮仅做远端只读检查，未上传或提交作业。

- SICC：登录节点 `login01`；用户属于 `group_${CLUSTER_ACCOUNT}`；`/data/${CLUSTER_ACCOUNT}/${CLUSTER_USER}` 可写，组归属正确。account/QOS 查询为 `${CLUSTER_ACCOUNT}/normal`，`a100_batch` 在线。`/data` 文件系统可用约 29T，此数值不是个人配额，个人/组存储配额尚未核实。
- SICC：默认 `python3` 为 `/usr/bin/python3`、Python 3.12.3；默认 PATH 未找到 `python3.10` 或 `quota`，现有安装脚本的 Python 3.10 前提尚未满足。
- 旧集群：登录节点 `${LEGACY_LOGIN}`；项目父目录存在，`gbunchQ2` 在线；存在 `/usr/bin/python3.10` 和 `/usr/bin/quota`。本轮未核实存储配额。
- 本地：RTX 5090、驱动 580.178.04；项目 CUDA 12.8 实际前后向通过。其他项目 GPU 进程 PID 4079267 保持运行；本轮未安装依赖。

### 早期访问限制（已解除）

13:12 用户提供宿主机 `nvidia-smi`：RTX 5090、驱动 580.178.04 正常；5223/32607 MiB 已用，其中现有 Python 任务约占 3378 MiB。确认的是宿主机 GPU 正常，agent 当前会话的 GPU/SSH 访问限制仍然存在。不改驱动，不结束其他项目进程。

用户已同意使用集群，并要求代码与数据共同保存到 `robotic/`。当前**没有本项目已提交的 Slurm 作业，也没有已确认的远端目录**。

本次只读尝试两个 SSH 地址均返回 `socket: Operation not permitted`。申请在沙箱外重试被自动审批拒绝，原因是当前配置禁止 sandbox approval。未进入远端账号，不能把此结果解释为密码错误、集群离线或账号无权限。

## 路径与连接

| 集群 | 登录地址 | 项目根目录 | 计划分区 |
|---|---|---|---|
| SICC，优先核实 | `${CLUSTER_USER}@${PRIVATE_IP}` | `${SICC_PROJECT}` | `a100_batch` |
| 旧集群，备用 | `${CLUSTER_USER}@${PRIVATE_IP}` | `${LEGACY_PROJECT}` | `gbunchQ2`，确认后可考虑 A40 |

SICC 连接保留 `HostKeyAlias=${SICC_LOGIN}` 与 `StrictHostKeyChecking=yes`，不绕过主机密钥校验。SICC 的 account 为 `${CLUSTER_ACCOUNT}`、QOS 为 `normal`，`/data` 文件组应为 `group_${CLUSTER_ACCOUNT}`。这些来自既有记录，仍需实时核实。

本地完整路径及映射见 PATHS.md (`../PATHS.md`; local-only historical reference)。两套真实参数保存在 Git 忽略的 `codes/configs/clusters.local.json`；`clusters.json` 改为可公开的示例。上传映射为本地整个 `codes/` → 远端整个 `robotic/`，其中数据在 `robotic/data/`，不会散落到其他目录。

## 资源与提交规则

- 旧集群文档：每用户最多 2 个运行作业、4 个队列作业、4 张 GPU；组配额另计。不能默认开 18 个 array task。
- SICC 文档：A100 为 40GB；单卡最多 8 CPU/40G 主机内存；组与用户还有作业数/排队限制。
- 每个实验先采用单作业单卡，不手动设置 `CUDA_VISIBLE_DEVICES`，GPU 由 Slurm 分配。
- 登录节点只用于环境安装、文件传输、队列管理和日志读取；模型、仿真和 GPU 测试均通过 Slurm。
- 首先提交 20 分钟预检，确认 CUDA、完整模型前后向和测试后，再运行 DP pilot；正式矩阵等待成本标定和协议通过。
- 首批并发建议 1–2 个作业，查询实际组内占用后再扩展；不取消或挤占已有其他项目任务。

## 已准备的工具

在本地 `codes/` 下：

```bash
python3 scripts/cluster.py check --cluster sicc
python3 scripts/cluster.py upload --cluster sicc
python3 scripts/cluster.py preflight --cluster sicc
```

默认只打印命令。添加 `--execute` 才执行，执行意向与返回结果追加到 `manifests/cluster_actions.jsonl`。旧集群使用 `--cluster legacy`。

上传前运行 `python3 scripts/source_manifest.py`，上传后校验源码 SHA256。上传不使用 `--delete`，排除本地虚拟环境、运行结果与临时下载；依赖安装包保留用于离线准备。发现活动 `robotic_` 作业时拒绝更新代码。

远端环境准备命令（上传后，在目标集群的 robotic 根目录执行；缺少 Python 3.10 时自动在项目内引导）：

```bash
bash scripts/setup_cluster_env.sh ${SICC_PROJECT} python3.10
```

该安装脚本的稳定版 PyTorch 2.7.1/CUDA 12.8 已通过 SICC A100 预检。不能直接复制本地 `envs/local`，因为其解释器路径和继承环境依赖是本机专用。依赖版本已保存；正式比较仍需统一模型、训练和评估协议。

## 作业登记

| 日期 | 集群 | 阶段 | Job ID | 状态 | 输出 |
|---|---|---|---|---|---|
| 2026-11-16 | SICC / 旧集群 | 连接检查 | 无 | 本地运行环境限制 SSH，尚未提交 | 本笔记 |
| 2026-11-16 13:45 后 | SICC / 旧集群 | 重新连接检查 | 无 | 两端 SSH 成功，队列为空，robotic 尚未创建 | 本笔记 |
| 2026-11-16 14:25 左右 | SICC | CUDA / 完整模型 / 测试预检 | 27159 | COMPLETED，0:0，51 秒 | robotic/results/preflight_27159.json |

后续每个任务必须记录：集群、job ID、配置、源码 hash、数据 hash、seed、GPU 型号、checkpoint 与评估路径。SSH 提交返回不明确时先查队列，不直接重复提交。

## 接下来

- [x] 验证两套集群 SSH、当前个人队列、分区及 SICC 父目录写权限。
- [ ] 核实组剩余存储配额；当前 Slurm 资源限制已查询。
- [x] 创建并核实 `robotic/`，上传代码与数据，检查 SICC 组归属。
- [x] 准备项目专属 Python 3.10 环境并提交预检，作业 27159 已通过。
- [ ] DP pilot 测量显存和吞吐，选取统一训练预算。
- [ ] 对称性与数据对齐通过后，逐批提交 18-run 正式矩阵。

来源：`${LOCAL_STORAGE}/DR_ZHAO/notes/cluster/tutorial.md`、`slurm2/README.md`、`slurm2/WORKFLOW.md`、`slurm2/SICC-Cluster_README_2026-11-04.md`。
