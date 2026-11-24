# 实验结果与画图数据地图

本文件所在目录就是主报告的数据入口。GitHub 路径为
`reports/main_results_20261123/`；本地路径为
`/mnt/hdd1/keru/homework/group_work/wangdian/codes/reports/main_results_20261123/`。

**修改图形样式直接使用 `plot_data/` 的 CSV 或 `summary.json` 即可，不需要重新训练。**
这里的主实验数值冻结于 2026-11-23，包含 114 次独立训练、342 次 checkpoint 评估、
34,200 条 checkpoint-episode 记录。它们对应 78 个 test runs 和 36 个独立 validation runs。
这些不是 34,200 次独立训练，也不是 34,200 个互不重复的测试初始状态。
E4 修正评估另行保存，未混入这份主报告。

## 一、已有图分别用哪个文件

下表中的文件、CSV、JSON 和脚本均纳入 GitHub。

| 图 / 想展示的结果 | 直接画图的 CSV | 对应原图 | 数据含义 |
|---|---|---|---|
| RQ1 / RQ2：25、50、100、200 条示教，三种方法 | [rq1_rq2.csv](plot_data/rq1_rq2.csv) | [PNG](rq1_rq2_test.png)、[PDF](rq1_rq2_test.pdf) | 12 组，三训练 seed 的均值及各 seed 值 |
| RQ3：state 输入，fixed / rotated goal，四种方法 | [rq3_state.csv](plot_data/rq3_state.csv) | [PNG](rq3_state_test.png)、[PDF](rq3_state_test.pdf) | 8 组；柱高为三 seed 均值，小点为各 seed |
| 收敛检查：20k / 40k / 60k updates | [convergence.csv](plot_data/convergence.csv) | [PNG](convergence_validation.png)、[PDF](convergence_validation.pdf) | 36 个曲线点，独立 validation 实验 |
| RQ3：图像输入下的 fixed / rotated 对照 | [rq3_image.csv](plot_data/rq3_image.csv) | 当前报告主要用表格，可据此另画图 | 6 组、18 个比较单元，复用 6 个主矩阵基线 |
| C8 消融 | [c8.csv](plot_data/c8.csv) | 当前报告主要用表格，可据此另画图 | 50 / 200 demos；C4 对照取 rq1_rq2.csv |
| 方法差异及 95% 配对区间 | [paired_comparisons.csv](plot_data/paired_comparisons.csv) | 可画差值图 / forest plot | 37 个比较 × coverage / success 两种指标 |

目前三张原图的绘图实现均在 [build_report_data.py](build_report_data.py) 的 `figures()` 函数。
图像插入 LaTeX 使用 `latex/figures/` 中的 PDF 副本；优化图后，需同步更新相应副本，
否则根目录新图与 LaTeX 使用的旧图可能不一致。

## 二、从汇总到逐 episode 的完整层级

| 文件 | 内容 | 用途 |
|---|---|---|
| [summary.json](summary.json) | 38 个统计组；各 seed、各 checkpoint 均值；37 个配对比较 | 原图与报告统计的直接来源 |
| [evidence.json](evidence.json) | 114 个 run，逐 checkpoint 的 coverage / success 数组、episode keys、checkpoint 哈希 | 重算统计、bootstrap、检查某个 seed |
| [tables.md](tables.md) | 已格式化的主要统计表 | 快速查看数值，不建议从四位小数表格反推精确数据 |
| [test_summary.csv](plot_data/test_summary.csv) | 26 个 test 统计组 | 主结果、RQ3 和 C8 的统一绘图表 |
| [test_seeds.csv](plot_data/test_seeds.csv) | 78 个 test run 的三-checkpoint均值 | 散点、箱线等 seed 层级展示；只有三个训练 seed |
| [checkpoint_summary.csv](plot_data/checkpoint_summary.csv) | 38 组 × 3 checkpoint，共 114 行 | 所有保留 checkpoint 的三 seed 汇总 |
| [run_checkpoints.csv](plot_data/run_checkpoints.csv) | 114 runs × 3 checkpoint，共 342 行 | 查看每个训练 seed 在每个 checkpoint 的表现 |
| [episodes.csv](plot_data/episodes.csv) | 34,200 行，包含 run、训练 seed、checkpoint、初态 seed、朝向、coverage、success | 自定义分布图、按朝向拆分、重算统计 |
| [run_catalog.csv](plot_data/run_catalog.csv) | 114 个 run 的方法、数据量、输入类型、源码版本、GPU、原始证据路径 | 从图中的点追溯到实验与源码 |
| [source_hashes.json](source_hashes.json) | 原始 config / evaluation 文件及采集索引的 SHA256 | 验证冻结证据的来源 |
| [plot_data/manifest.json](plot_data/manifest.json) | CSV 行数、输入与输出 SHA256、导出校验记录 | 确认 CSV 与冻结 JSON 对齐 |

`evidence.json` 中一个 run 的 `episode_keys[i]` 对应同一 run 各 checkpoint 中
`coverage[i]`、`success[i]` 的同一个初始条件。键是 `(base_seed, orientation)`。
`run_id` 形如 `c4_main_20261118_v1/dp_n50_s0`，不以运行机器命名。

## 三、画图时指标怎么读

- `mean_coverage`：每个 episode 在控制过程中达到的最大覆盖率，再按协议求均值。
  coverage = 交集面积 / 目标面积；不是 IoU，也不是成功率。
- test 主结果：每个训练 seed 先对 16k / 18k / 20k 三个 checkpoint 和各自 100 episodes
  求均值，再汇总三个训练 seed。原图中线或柱为三 seed 均值，小点为各 seed。
- `seed_std`：三个训练 seed 的样本标准差（ddof=1），不是标准误，也不是 95% CI。
- `coverage_ci95_low/high`：冻结统计中的 bootstrap 区间，重采样训练 seed 和基础初态，
  四个朝向保留为一组。只有三个训练 seed；区间未做多重比较校正。
- `success_rate`：0--1 的比例；展示百分比时乘 100。Push-T 成功条件是 coverage > 0.95。
- `final_20k_coverage` 只用最终 20k；`limited_grid_max_mean` 是每个 seed 在三个保留
  checkpoint 中取最好的得分再平均，不能标成完整训练过程中最优值。
- 收敛曲线只使用 validation。其 20k 点来自独立训练，不能拼到主 test 曲线上。
- `paired_comparisons.csv` 的差值始终为 `left - right`，换比较方向时须同时翻转区间。

RQ1、RQ2 共用主矩阵；RQ3 图像版复用部分主矩阵基线。按图表相加会重复计算实验数，
需要用 `run_id` 去重。不同 checkpoint 是重复测量，不能当作新的独立训练 seed。

## 四、本地原始文件在哪里，哪些没有上 GitHub

下面的 `codes/` 相对于完整工作区 `wangdian/`。

| 本地目录 | 保存内容 | GitHub 是否收录 |
|---|---|---|
| `codes/reports/main_results_20261123/` | 冻结 JSON、CSV、图表、报告和 LaTeX | **是** |
| `codes/tasks/rq1_transfer/results/` | 主矩阵、收敛等原始 config、metrics、每个 checkpoint 的评估 JSON | 原始目录不上传；画图所需逐 episode 证据已提取到公开 evidence.json / CSV |
| `codes/tasks/rq2_exact_aug/results/` | C8 等任务证据；共享主矩阵由 RQ1 保存 | 同上 |
| `codes/tasks/rq3_symmetry_breaking/results/` | state / image RQ3 原始证据 | 同上 |
| `development/proposal_v2/releases/<release>/runs/<run>/` | 原始训练工作目录、checkpoint、日志、评估输出 | 大文件和运行目录不上传 |
| `development/proposal_v2/results/` | 集群回传、验证结果、模型备份及归档收据 | 本次精选轻量记录发布到 docs/experiment_records；不是完整镜像 |

查某一行的原始 JSON，先从 `run_catalog.csv` 的 `evidence` 列定位；该列相对 Git 仓库根：

```text
codes/tasks/rq1_transfer/results/c4_main_20261118_v1/runs/dp_n50_s0/
  config.json
  runtime.json
  metrics.jsonl
  test_step_16000.json
  test_step_18000.json
  test_step_20000.json
```

validation 文件名相应为 `validation_step_20000.json`、`validation_step_40000.json`、
`validation_step_60000.json`。`metrics.jsonl` 保存训练过程指标；本次 CSV 是报告评估数据，
不包含每一步 loss、轨迹视频或所有中间 checkpoint。

## 五、E4 单独放在哪里

本地修正评估的自动回传目录：

```text
codes/tasks/rq1_transfer/results/e4_corrected_eval_20261124/v2/
  STATUS.md
  monitor.json
  results/equidiff_s0/
    status.json
    preflight_passed.json
    d1/<首个episode seed>/scores.json
    d1/<首个episode seed>/scenes/<episode seed>.json
    d1/scores.json                 # 该域100次完成后才生成
    d2/scores.json                 # 同上
    d0/scores.json                 # 同上
```

这部分实时结果**不自动上传 GitHub**。GitHub 已有
[修正评估代码、协议和预检证据](../../operations/e4_corrected_eval_20261124/README.md)，
以及本次实验记录中的带时间戳状态快照。待完整结果校验后，再发布独立 E4 结果快照。
不要用 `smoke_d*/scores.json` 的八步工程测试得分画科学结果，也不要将修正前的 D1/D2
零成功率当作已经验证的完整目标分布结果。

## 六、建议的重绘方式

用自己偏好的 matplotlib、seaborn、Origin 或 Excel 读取对应 CSV，把新图输出到新的目录。
保持 JSON 和 CSV 数据不变即可独立修改配色、字号、布局、误差线和图例。
例如从 Git 仓库根运行：

```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

root = Path('reports/main_results_20261123')
data = pd.read_csv(root / 'plot_data/rq1_rq2.csv')
fig, ax = plt.subplots()
for method, group in data.groupby('method'):
    group = group.sort_values('demos')
    ax.plot(group['demos'], group['mean_coverage'], marker='o', label=method)
ax.set(xlabel='Training demonstrations', ylabel='Mean episode-max coverage', ylim=(0, 1))
ax.legend()
output = root / 'redrawn_figures'
output.mkdir(exist_ok=True)
fig.savefig(output / 'rq1_rq2.pdf', bbox_inches='tight')
```

重新生成 CSV（仅标准库，无需 GPU）：

```bash
python3 reports/main_results_20261123/export_plot_data.py
```

原脚本 `build_report_data.py` 会重算汇总并覆盖三张原图，需 NumPy / Matplotlib。
其 `--capture` 会重新读取当前本地实验目录并改写冻结证据；**仅调整图形时不需要此参数**。

实验过程、协议版本及失败恢复记录见 [实验记录入口](../../docs/experiment_records/README.md)。
