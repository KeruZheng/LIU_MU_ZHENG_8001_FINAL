> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `notes/study_list.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# 论文学习路线（今晚冲刺版）

更新：2026-11-17
目标：今晚读完核心论文，明天能参与三臂实验、集群标定结果和结果分析的讨论。
正式汇报：2026-11-26（还有 9 天）。

---

## 0. 今晚的验收标准

读完下面内容后，你应该能**不看笔记**回答：

- [ ] EquiDiff 约束的是哪个函数？为什么这个约束能省数据？
- [ ] 为什么 MimicGen 上的"旋转增强"基线不公平？Push-T 上为什么公平？
- [ ] DP、DP + exact C4 aug、EquiDiff-C4 三者的归纳偏置和计算代价差在哪？
- [ ] RQ2 的三种可能结果分别说明什么？哪种结果会推翻我们的假设？
- [ ] FrameAvg 为什么是 fallback？4× 成本具体来自哪里？
- [ ] 我们的 coverage 定义、success 阈值、100-episode 评估协议是什么？

---

## 1. 一张图看懂 proposal 的论证链

```text
原论文主张
  EquiDiff 在 MimicGen 上 +21.9pp；App. I 说等变架构 53.7 > 旋转增强 46.2
        │
        ▼
但是：MimicGen 是斜视角 6-DoF
  旋转 RGB 图像 ≠ 旋转场景 → 增强样本物理上错误
  "架构 vs 增强"的比较被混淆
        │
        ▼
换任务：Push-T
  俯视、平面、方形场地、无重力、目标在中心
  90° 旋转是动力学 + 像素的精确对称（96×96 全图，rot90 无损）
        │
        ▼
核心问题 RQ2
  当 DP 也能用精确 C4 增强时，EquiDiff 的等变结构还有多少额外收益？
        │
        ├── EquiDiff 显著更好 → 等变结构有独立价值
        ├── 打平             → 原论文增益主要来自增强质量/任务几何（H2）
        └── EquiDiff 更差     → 强约束限制拟合，或实现/预算有问题
        │
        ▼
数据效率 RQ1：50 vs 200 demos；错误等变 RQ3：固定目标 + 目标方向输入
        │
        ▼
有效性边界
  FrameAvg ≠ EquiDiff；4× 旋转副本 ≠ 4× 独立示教；
  训练不足 ≠ 方法差异；rot90 无损 ≠ 环境严格等变
```

**一句话记住：我们不是在"复现 EquiDiff 并希望它赢"，而是在做一个受控的证伪实验，把"等变架构"和"精确增强"分开。**

---

## 2. 阅读顺序和时间盒

### 路线 A：今晚 3 小时（最低要求）

| 时间 | 内容 | 读完能做什么 |
|---|---|---|
| 0:00–0:20 | 本地材料：`01_EquiDiff_学习报告.md` + `03_Proposal_A_EquiDiff_final.md` | 知道我们要回答什么、为什么换 Push-T |
| 0:20–1:40 | **EquiDiff 论文** | 解释等变约束、+21.9pp、App. I 的混淆 |
| 1:40–2:30 | **Diffusion Policy** Sec. 3 + Push-T 实验 | 解释 DP 如何生成动作、Push-T 的观测/指标 |
| 2:30–3:00 | **Frame Averaging** Sec. 2–3 | 解释 4 次前向如何得到精确 C4 等变 |

### 路线 B：今晚 4.5 小时（推荐）

在路线 A 基础上加：

- 30 min：proposal 提到的 NeurIPS 2025 follow-up（FrameAvg / invariant representations）
- 30 min：Push-T 源码 + 我们的 `geometry.py` / `evaluate.py`
- 30 min：整理笔记，写下 5 个自测问题的答案

### 路线 C：5–6 小时（完整）

再加（可选）：

- Implicit Behavioral Cloning（Push-T 的来源）
- Group Equivariant CNN / Steerable CNN 的 C4 特例
- Approximately Equivariant Networks（RQ3 的 gate 思路）

---

## 3. 每篇论文怎么读

### 3.1 本地材料（20–30 min，先读）

| 文件 | 读什么 | 读完回答 |
|---|---|---|
| `01_EquiDiff_学习报告.md` | 全文，中文摘要 | EquiDiff 的核心机制和实验结论 |
| `03_Proposal_A_EquiDiff_final.md` | Sec. 1–3、5–6 | RQ1/RQ2/RQ3、H1/H2/H3、三臂设计 |
| `presentation/list.md` | Sec. 3、5、6、8 | 数据划分、指标、18-run 矩阵、算力门槛 |
| `notes/reproduction.md` | "实现边界""已有验证证据" | 哪些是正式结果，哪些只是工程冒烟 |
| `codes/README.md` | "Current scope" | pilot 不是官方 DP 配方；EquiDiff 尚未移植 |

**注意**：`presentation/list.md` 里"待确认"和旧日期是历史内容，当前进度以 `notes/list.md` 为准。

### 3.2 EquiDiff（核心，60–90 min）

- 论文：Wang et al., *Equivariant Diffusion Policy*, CoRL 2024
- 链接：https://arxiv.org/abs/2407.01812

**读这些**：

1. **Abstract + Intro**：用三句话记下——主张、方法、主要结果。
2. **Method**：找到等变公式 `ε(g·a, t, g·o) = g·ε(a, t, o)`；理解动作/观测如何被赋予 SO(2) 表示；理解 weight tying 在做什么。
3. **Experiments**：MimicGen 12 个任务、100 demos、+21.9pp；数据效率曲线。
4. **App. I**：53.7 vs 46.2 的实验设置；为什么旋转增强样本在斜视角下物理错误。
5. **Limitations**：作者自己承认的相机视角问题。

**先跳过**：escnn 的实现细节、完整证明、其他任务的细节。

**读完要能回答**：

- [ ] 等变约束作用在去噪器上，而不是策略输出上——为什么这样设计？
- [ ] 为什么等变能省数据？（参数共享 vs 从数据中学习旋转副本）
- [ ] App. I 的增强基线到底哪里错了？
- [ ] 原论文的结论在什么条件下才成立？

**笔记模板**：

```text
论文：
一句话主张：
关键证据（数字/图表）：
作者承认的局限：
与我们 RQ1/RQ2/RQ3 的关系：
我还不懂的地方：
```

### 3.3 Diffusion Policy（背景，45–60 min）

- 论文：Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023
- 链接：https://arxiv.org/abs/2303.04137

**读这些**：

1. **Sec. 3 Method**：DDPM/DDIM 如何生成动作序列；观测历史、动作 horizon、receding horizon。
2. **Push-T 实验**：观测（96×96 图像 + 2D agent_pos）、动作（2D 目标位置）、评估（coverage、success）。
3. **Appendix 中 Push-T 的实现细节**（如果时间允许）。

**先跳过**：其他机器人任务、与 Push-T 无关的消融。

**读完要能回答**：

- [ ] 一个扩散策略如何从观测历史生成一段动作？
- [ ] 为什么 Push-T 是好的对称性测试平台？
- [ ] 官方 coverage 和 success 的定义是什么？
- [ ] 我们的 pilot 和官方 DP 配方差在哪？（`README.md` 已列出）

### 3.4 Frame Averaging（30–45 min）

- 论文：Puny et al., *Frame Averaging for Invariant and Equivariant Network Design*, ICLR 2022
- 链接：https://arxiv.org/abs/2110.03336

**读这些**：

1. **Sec. 2–3**：群平均公式 `Φ_FA(x) = (1/|G|) Σ_g g^{-1} f(g·x)`。
2. 为什么对任意网络都能得到**精确**等变，而不需要改网络结构。
3. 代价：C4 需要 4 次前向/反向。

**读完要能回答**：

- [ ] 为什么平均之后一定等变？
- [ ] 4× 成本具体花在哪里？训练和推理都贵吗？
- [ ] 我们的 `C4FrameAverage` 如何同时变换图像、agent、动作和噪声？
- [ ] 为什么它只能叫 fallback，不能叫 EquiDiff？

### 3.5 NeurIPS 2025 follow-up（20–30 min，可选）

- 引用以 `03_Proposal_A_EquiDiff_final.md` 为准；在 arXiv/Google Scholar 搜索 "equivariant diffusion policy frame averaging invariant representations"。
- **只读** Abstract + Intro + Method + 实验设置。
- 读完回答：
  - [ ] 他们为什么从 escnn 转向 FrameAvg / invariant representations？
  - [ ] 成本/性能 trade-off 是什么？
  - [ ] 如果我们的第三臂用 FrameAvg，和这篇工作的关系是什么？

### 3.6 Push-T 源码（30 min，推荐）

本地路径：

- `codes/upstream/diffusion_policy/diffusion_policy/env/pusht/pusht_env.py`
- `codes/upstream/diffusion_policy/diffusion_policy/env/pusht/pusht_image_env.py`
- `codes/upstream/diffusion_policy/diffusion_policy/dataset/pusht_image_dataset.py`
- `codes/upstream/diffusion_policy/image_pusht_diffusion_policy_cnn.yaml`

**读的时候只回答**：

- [ ] 为什么 90° 旋转是精确对称？（俯视、方形、中心、rot90）
- [ ] 观测为什么是 96×96 全图，而不是以 agent 为中心的 crop？
- [ ] 目标中心 256 和场地中心 255.5 的 0.5 px 差异在哪里？
- [ ] coverage = `area(block ∩ goal) / area(goal)`，success > 0.95。
- [ ] 官方训练配方：batch 64、3050 epochs、`down_dims [512,1024,2048]`、crop 84、cosine LR；我们的 pilot 是简化版。

---

## 4. 论文概念 ↔ 代码映射

| 论文概念 | 我们的代码 |
|---|---|
| C4 图像/坐标/动作旋转 | `codes/src/pusht_study/geometry.py` |
| 数据读取、序列采样、划分 | `codes/src/pusht_study/data.py`、`codes/manifests/split.json` |
| DP 去噪器（ResNet18 + U-Net） | `codes/src/pusht_study/models.py` 的 `Denoiser` |
| Frame Averaging | `codes/src/pusht_study/models.py` 的 `C4FrameAverage` |
| 噪声预测训练、EMA、断点恢复 | `codes/src/pusht_study/train.py` |
| rollout、coverage、success | `codes/src/pusht_study/evaluate.py`、`env.py` |
| 对称性诊断（物理 vs 像素） | `codes/scripts/check_symmetry.py`、`codes/results/symmetry_native.json` |
| 官方 DP 配方 | `codes/upstream/diffusion_policy/image_pusht_diffusion_policy_cnn.yaml` |
| 官方 Push-T 环境 | `codes/upstream/diffusion_policy/diffusion_policy/env/pusht/` |

---

## 5. 五个自测问题

读完论文后，合上笔记，尝试回答：

1. **EquiDiff 的等变约束具体作用在哪个函数上？为什么它和"数据增强"不是一回事？**
2. **为什么 MimicGen 上旋转图像 ≠ 旋转场景？Push-T 上为什么成立？**
3. **DP + exact aug 和 EquiDiff 的归纳偏置差在哪？各自的计算代价？**
4. **RQ2 的三种结果分别支持什么结论？哪种结果会推翻 proposal 的假设？**
5. **我们的实验哪里可能不公平或无效？**（数据朝向、训练预算、评估协议、渲染误差）

如果这五个问题能答出来，明天的实验讨论你就能直接参与。

---

## 6. 如果今晚只有 2 小时

优先级：

1. 20 min：`01_EquiDiff_学习报告.md` + `03_Proposal_A_EquiDiff_final.md` 的 Sec. 1–2。
2. 70 min：EquiDiff 的 Abstract / Intro / Method / App. I / Limitations。
3. 30 min：Diffusion Policy 的 Sec. 3 + Push-T 实验。

Frame Averaging、follow-up 和 Push-T 源码留到明早 1 小时。

---

## 7. 今晚不要读什么

- escnn 源码和完整数学推导（只有移植原版 EquiDiff 时才需要）
- EquiDiff 附录里的完整证明
- MimicGen 的其他任务细节
- `02_TD-MPC2_学习报告.md`、`04_Proposal_B_TD-MPC2.md`（不同方向）
- 旧版相机/光照扰动 proposal
- proposal 里 200 GPU-h 的完整计划（本周只做核心 RQ1/RQ2）

---

## 8. 读完之后的下一步

- [ ] 明早补齐 FrameAvg / follow-up / Push-T 源码（如果今晚跳过）
- [ ] 读 `codes/scripts/check_symmetry.py` 和 `codes/results/symmetry_native.json`，理解"物理误差小、像素误差不为零"
- [ ] 读 `codes/manifests/experiment_matrix.json`，知道 18-run 矩阵和当前 blocked 状态
- [ ] 读 `notes/cluster.md`，知道 SICC 预检 27159 和标定作业状态
- [ ] 参与讨论：第三臂选 EquiDiff 还是 FrameAvg；50 demos 的 9-run 何时提交

---

## 9. 参考链接

**核心**

- EquiDiff: https://arxiv.org/abs/2407.01812
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Frame Averaging: https://arxiv.org/abs/2110.03336

**可选**

- Implicit Behavioral Cloning（Push-T 来源）: https://arxiv.org/abs/2109.00137
- Approximately Equivariant Networks: https://arxiv.org/abs/2201.11969
- Group Equivariant Convolutional Networks: https://arxiv.org/abs/1602.07576

**本地**

- proposal: `${WORKSPACE}/03_Proposal_A_EquiDiff_final.md`
- 学习报告: `${WORKSPACE}/01_EquiDiff_学习报告.md`
- 进度: `${WORKSPACE}/notes/list.md`
- 复现证据: `${WORKSPACE}/notes/reproduction.md`
- 代码: `${WORKSPACE}/codes/`
