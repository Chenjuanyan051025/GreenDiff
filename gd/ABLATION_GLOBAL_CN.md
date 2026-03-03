# GreenDiff 全局消融实验设计

## 1. 目标

这份文档用于把 GreenDiff 的**全局消融实验**固定下来，回答以下核心问题：

1. 为什么要在 `latent space` 做逆推断，而不是直接在原空间做回归？
2. 为什么必须有 `LatentGreen` 物理代理，而不是只靠条件扩散？
3. 为什么代理必须做**双域一致性 + 频谱/统计校准**，而不是普通 MSE surrogate？
4. 为什么推理阶段必须有 `TeacherSampler / 流形校正 / 高保真正`，而不是只在训练期加 physics loss？
5. 在相同高保真预算下，主动 teacher 机制是否优于固定或随机策略？

这不是局部组件消融，而是整篇项目的系统级消融设计。

## 2. 指标体系（全局统一）

为了避免每张表口径不一致，建议全篇统一使用以下三层指标。

### 2.1 主指标（每张表尽量都要有）

- `Rel L2 (Linear)`
- `Residual`
- `PSD error`

这三项分别对应：

- 整体精度
- 物理一致性
- 频谱/结构一致性

### 2.2 物理分布指标（代理与采样相关表必须有）

- `MSE (Linear)`
- `MAE (Linear)`
- `Peak Ratio mean`
- `Peak Ratio p95`
- `Peak Ratio p99`
- `Pred/Obs P99 ratio`
- `Pred/Obs Std ratio`
- `Pred/Obs Mean ratio`

这些指标主要用于证明：

- 不是只对上均值
- 没有过平滑
- 尖峰与尾部分布没有塌

### 2.3 成本与采样指标（teacher / sampling 表必须有）

- `num_corrections`
- `num_query_requested`
- `num_teacher_rejected`
- 推理时间（每样本或每 batch）
- 每样本高保真调用数

## 3. Table 1：系统级递进消融（主表）

这是整篇最重要的一张表。它用“逐层加模块”的方式证明每一层都有必要。

### 3.1 实验组

1. `R-Direct`
   直接从 `g_obs -> V` 做监督回归。
2. `R-Latent`
   `g_obs -> z -> decode -> V`，有 latent 表示，但没有 diffusion、没有物理代理。
3. `D-Latent`
   条件潜空间扩散：`g_obs -> diffusion -> z -> decode -> V`，无物理代理、无推理校正。
4. `D-LG-Train`
   扩散训练时加入 `LatentGreen` 物理代理约束，但推理时不做 guidance。
5. `D-LG-Legacy`
   使用旧的 heuristic guidance。
6. `D-LG-PMD1`
   使用新的流形校正采样（1-step）。
7. `D-LG-PMD2-ActHF-B1`
   在 PMD 基础上加入主动高保真正（预算 `B=1`）。

### 3.2 表格模板

| Method | Representation | Train Phys | Inference Guidance | High-Fidelity Teacher | Rel L2 (Linear) | Residual | PSD error | Pred/Obs Std ratio | #Teacher Queries |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| R-Direct | Pixel | No | No | No |  |  |  |  |  |
| R-Latent | Latent AE | No | No | No |  |  |  |  |  |
| D-Latent | Latent Diffusion | No | No | No |  |  |  |  |  |
| D-LG-Train | Latent Diffusion | Yes | No | No |  |  |  |  |  |
| D-LG-Legacy | Latent Diffusion | Yes | Legacy | No |  |  |  |  |  |
| D-LG-PMD1 | Latent Diffusion | Yes | PMD-1 | No |  |  |  |  |  |
| D-LG-PMD2-ActHF-B1 | Latent Diffusion | Yes | PMD-2 | Active |  |  |  |  |  |

### 3.3 结论目标

这张表要支撑的核心结论是：

- `latent` 表示比直接回归更适合病态逆问题；
- diffusion 比 regression 更适合表达多解后验；
- `LatentGreen` 不是可选辅助件，而是系统核心；
- 新的 PMD 采样优于旧 heuristic guidance；
- 主动高保真正在有限预算下仍有额外收益。

## 4. Table 2：表示层消融（Representation Ablation）

这张表回答：为什么要用 `VAE + latent diffusion`。

### 4.1 实验组

1. `R-Direct`
2. `R-Latent`
3. `D-Latent`

如果算力允许，再加：

4. `D-Pixel`
   直接在原空间 `V` 上做 diffusion（哪怕是小分辨率版本）。

### 4.2 表格模板

| Method | Space | Generative? | Rel L2 (Linear) | Residual | PSD error | Runtime / sample | GPU Memory |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| R-Direct | Pixel | No |  |  |  |  |  |
| R-Latent | Latent | No |  |  |  |  |  |
| D-Latent | Latent | Yes |  |  |  |  |  |
| D-Pixel (optional) | Pixel | Yes |  |  |  |  |  |

### 4.3 结论目标

- `latent` 表示更稳定、更高效；
- `latent diffusion` 更适合表达后验多样性；
- 原空间方法不是不能做，而是成本和稳定性更差。

## 5. Table 3：LatentGreen 全局消融（代理层消融）

这张表专门回答：为什么物理代理存在、且必须被“正确训练”。

### 5.1 子表 A：代理是否必要

| Method | LatentGreen | Proxy Training | Rel L2 (Linear) | Residual | PSD error | Pred/Obs Std ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| D-Latent | No | - |  |  |  |  |
| D-LG-Plain | Yes | MSE / obs only |  |  |  |  |
| D-LG-Dual | Yes | Dual-domain |  |  |  |  |
| D-LG-Full | Yes | Dual + spectral + stats + peak |  |  |  |  |

### 5.2 子表 B：代理主干消融

| Backbone | Rel L2 (Linear) | Residual | PSD error | Peak Ratio p95 | Pred/Obs Std ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| cnn |  |  |  |  |  |
| fno |  |  |  |  |  |
| hybrid_fno |  |  |  |  |  |

### 5.3 子表 C：代理损失消融

建议在 `LatentGreen Full` 上逐个去掉：

- `- fft_loss`
- `- psd_loss`
- `- stats_loss`
- `- multiscale_loss`
- `- peak_control`
- `- residual_loss`

表格模板：

| Proxy Variant | Rel L2 (Linear) | Residual | PSD error | Peak Ratio p99 | Pred/Obs Std ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full |  |  |  |  |  |
| - fft_loss |  |  |  |  |  |
| - psd_loss |  |  |  |  |  |
| - stats_loss |  |  |  |  |  |
| - multiscale_loss |  |  |  |  |  |
| - peak_control |  |  |  |  |  |
| - residual_loss |  |  |  |  |  |

### 5.4 结论目标

- 代理是必要的；
- 代理训练方式本身是创新点，不是“堆 loss”；
- `cnn/fno/hybrid_fno` 是实现归纳偏置，不是整篇主方法，但会影响最终效果。

## 6. Table 4：采样层消融（Inference Ablation）

这张表专门回答：推理阶段的结构化校正为什么是必要的。

### 6.1 主组

1. `No Guidance`
2. `Legacy Guidance`
3. `PMD-1`
4. `PMD-2`
5. `PMD-3`

### 6.2 表格模板

| Method | Step Schedule | Grad Steps | Normalize Grad | Grad Clip | Max Step Norm | Rel L2 (Linear) | Residual | PSD error | Runtime / sample |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| No Guidance | - | 0 | - | - | - |  |  |  |  |
| Legacy Guidance | legacy | 1 | No | - | - |  |  |  |  |
| PMD-1 | constant/sigma2 | 1 | No | - | - |  |  |  |  |
| PMD-2 | constant/sigma2 | 2 | No | - | - |  |  |  |  |
| PMD-3 | constant/sigma2 | 3 | No | - | - |  |  |  |  |

### 6.3 细分消融（推荐单独子表）

#### A. 步长调度
- `legacy`
- `constant`
- `sigma2`
- `late_strong`

#### B. 梯度处理
- `normalize_grad = false / true`
- `grad_clip = null / 1.0 / 2.0`

#### C. 位移约束
- `max_step_norm = null / 0.5 / 1.0`

### 6.4 结论目标

- 新流形校正优于旧 heuristic guidance；
- 多步小校正优于单步大拉回；
- 梯度规范化与步长约束影响的是稳定性而不是简单提分。

## 7. Table 5：高保真预算消融（Budgeted Teacher Ablation）

这是最像“算法论文亮点”的一组。

### 7.1 固定预算

建议至少测试：

- `B = 0`
- `B = 1`
- `B = 2`
- `B = 4`

其中 `B` 定义为：**每个样本最多允许的高保真查询次数**。

### 7.2 比较策略

1. `No Teacher`
2. `Fixed-Step Teacher`
3. `Random Teacher`
4. `Active Teacher`
5. 可选：`Always Teacher`（仅作上界参考）

### 7.3 表格模板

| Budget B | Strategy | Query Rule | Rel L2 (Linear) | Residual | PSD error | #Teacher Queries | #Teacher Rejections | Runtime / sample |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | No Teacher | - |  |  |  | 0 | 0 |  |
| 1 | Fixed-Step | fixed t |  |  |  |  |  |  |
| 1 | Random | random t |  |  |  |  |  |  |
| 1 | Active | risk >= tau |  |  |  |  |  |  |
| 2 | Fixed-Step | fixed t |  |  |  |  |  |  |
| 2 | Random | random t |  |  |  |  |  |  |
| 2 | Active | risk >= tau |  |  |  |  |  |  |
| 4 | Fixed-Step | fixed t |  |  |  |  |  |  |
| 4 | Random | random t |  |  |  |  |  |  |
| 4 | Active | risk >= tau |  |  |  |  |  |  |

### 7.4 必配图

1. `性能-预算 Pareto 曲线`
2. `查询时刻分布图`
3. `teacher_reject_frac vs t`

### 7.5 结论目标

- 在相同 teacher 预算下，主动策略优于固定和随机策略；
- 在相同目标精度下，主动策略更省昂贵求解预算；
- 你的方法不是“多查几次”取胜，而是“更聪明地查”。

## 8. Table 6：后验/多解性消融（Posterior Ablation）

这张表和配图用于支撑“零空间感知后验生成”的叙事。

### 8.1 实验组

固定同一个 `g_obs`，对比：

1. `R-Direct`
2. `D-Latent (single sample)`
3. `D-Latent (multi-sample)`
4. `PMD (multi-sample)`
5. `PMD + Active Teacher (multi-sample)`

### 8.2 统计量

需要同时度量：

- `V-space diversity`
- `g-space consistency`

### 8.3 表格模板

| Method | #Samples / cond | V-space Diversity | g-space Consistency | Best-of-N Rel L2 | Mean-of-N Residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| R-Direct | 1 | 0 |  |  |  |
| D-Latent (single) | 1 | 0 |  |  |  |
| D-Latent (multi) | 10 |  |  |  |  |
| PMD (multi) | 10 |  |  |  |  |
| PMD + Active Teacher (multi) | 10 |  |  |  |  |

### 8.4 必配图

对同一个 `g_obs`：

- 展示多个 `V` 样本
- 再展示它们前向得到的 `g_pred`

希望看到：

- `V` 有明显差异
- `g_pred` 仍然高度一致

## 9. Table 7：泛化消融（Generalization Ablation）

这张表用来支撑“模型学到的是可迁移的逆推断机制，而不是死记训练分布”。

### 9.1 推荐维度

#### A. 参数泛化

- `hopping` 留区间测试
- `eta` 留区间测试

#### B. 结构泛化

- 不同 defect 模式
- 不同 disorder 强度

#### C. 代理主干泛化（可选）

- `cnn`
- `fno`
- `hybrid_fno`

### 9.2 表格模板

| Split | Setting | Method | Rel L2 (Linear) | Residual | PSD error |
| --- | --- | --- | ---: | ---: | ---: |
| IID | in-range | PMD |  |  |  |
| OOD | held-out hopping | PMD |  |  |  |
| OOD | held-out eta | PMD |  |  |  |
| OOD | unseen defect regime | PMD |  |  |  |

## 10. 图表清单（建议最少）

除了表，建议最少配以下 4 张图：

1. `Figure 1: 系统递进柱状图`
   - 横轴：系统版本
   - 纵轴：`Rel L2` / `Residual`

2. `Figure 2: 性能-预算曲线`
   - 横轴：teacher budget
   - 纵轴：`Residual` 或综合分数

3. `Figure 3: 风险分数可靠性图`
   - 把样本按 `risk_score` 分桶
   - 看每桶的 `Residual` / `teacher_reject_rate`

4. `Figure 4: 后验多样性图`
   - 同一 `g_obs` 的多个 `V` 样本及对应 `g_pred`

## 11. 实验配置矩阵（按当前仓库能力拆分）

## 11.1 当前仓库可直接运行的实验组

这些不需要新建模型脚本，靠现有代码即可完成。

### A. LatentGreen 代理层

训练入口：

```bash
python -m gd.train.train_latent_green --config gd/configs/default.yaml
```

评估入口：

```bash
python -m gd.test.test_latent_green --config gd/configs/default.yaml
```

可直接覆盖：

- `cnn / fno / hybrid_fno`
- 代理损失消融
- `Peak Ratio / Std Ratio / PSD error / Residual`

关键配置项：

- `latent_green.model.backbone`
- `latent_green.loss.*`
- `latent_green.peak_control.*`
- `latent_green.training.*`

### B. Diffusion + 推理层

训练入口：

```bash
python -m gd.train.train_diffusion --config gd/configs/default.yaml
```

评估入口：

```bash
python -m gd.test.test_diffusion --config gd/configs/default.yaml
```

可直接覆盖：

- `No Guidance`
- `Legacy Guidance`
- `PMD-1/2/3`
- `Active Teacher / Fixed Teacher / Random Teacher`（需要在当前 `TeacherSampler` 策略上通过配置和少量测试脚本控制）

关键配置项：

- `guidance.enabled`
- `guidance.lambda.*`
- `guidance.manifold.*`
- `guidance.risk.*`
- `guidance.budget_hooks.*`
- `validation.restart.*`
- `validation.kpm_check.*`
- `diffusion.sampler.*`

## 11.2 当前仓库需要额外补基线脚本的实验组

这些是全局消融需要，但当前仓库没有现成“一键可跑”的对应基线。

### A. `R-Direct`

需要一个直接从 `g_obs -> V` 的监督回归训练脚本。

可选做法：

- 复用 `StudentModel` 改成直接监督 `V` 的 baseline
- 或新建一个最小 `train_direct_regression.py`

### B. `R-Latent`

需要一个：

- `g_obs -> z`
- 再 `vae.decode(z) -> V`

的 latent 回归基线。

这同样需要补一个专门 baseline runner，当前仓库没有直接对应脚本。

### C. `D-Pixel`（如果你要做）

需要新建原空间 diffusion baseline，不属于当前主线代码，可选放弃。

## 12. 配置覆盖矩阵（推荐直接做成 run sheet）

下面给的是建议的“每组实验至少改哪些配置项”。

### 12.1 系统级递进

#### `D-Latent`

- `guidance.enabled = false`
- `validation.enabled = false`
- `diffusion.training.phys_supervision.enabled = false`（如果有该开关）

#### `D-LG-Train`

- 打开 diffusion 训练期 physics supervision
- `guidance.enabled = false`
- `validation.enabled = false`

#### `D-LG-Legacy`

- `guidance.enabled = true`
- `guidance.manifold.enabled = true`
- `guidance.manifold.step_schedule = legacy`
- `guidance.manifold.grad_steps_per_iter = 1`
- `guidance.budget_hooks.enabled = false`
- `validation.enabled = false`

#### `D-LG-PMD1`

- `guidance.enabled = true`
- `guidance.manifold.step_schedule = constant` 或 `sigma2`
- `guidance.manifold.grad_steps_per_iter = 1`
- `guidance.budget_hooks.enabled = false`

#### `D-LG-PMD2-ActHF-B1`

- `guidance.enabled = true`
- `guidance.manifold.grad_steps_per_iter = 2`
- `guidance.budget_hooks.enabled = true`
- `guidance.budget_hooks.policy = threshold`
- `guidance.budget_hooks.threshold = <固定阈值>`
- `guidance.budget_hooks.max_queries_per_sample = 1`
- `guidance.budget_hooks.dry_run = false`
- `validation.kpm_check.enabled = true`
- `validation.restart.enabled = true`

### 12.2 预算表

#### `No Teacher`

- `guidance.budget_hooks.enabled = false`

#### `Fixed-Step Teacher`

当前仓库还没有单独的 fixed-step 策略开关。

建议做法：

- 在测试脚本或临时评估脚本中，手动指定固定时间步 query mask
- 或在 `TeacherSampler` 中增加一个固定策略分支（如果后续要固化成正式基线）

#### `Random Teacher`

同样建议通过测试脚本实现：

- 固定随机种子
- 从所有可采样步里随机选查询步

#### `Active Teacher`

- `guidance.budget_hooks.enabled = true`
- `policy = threshold`
- `threshold = tau`
- `max_queries_per_sample = B`
- `dry_run = false`

## 13. 推荐运行顺序（避免先跑最贵的）

建议按下面顺序推进：

1. **先跑 LatentGreen 全套**
   - 先把代理 backbone 和 surrogate loss 消融做完
   - 先确定最强 `LatentGreen` 再进入大规模 diffusion 消融

2. **再跑采样层消融**
   - `No Guidance`
   - `Legacy`
   - `PMD-1`
   - `PMD-2`

3. **再跑预算表**
   - 先 `B=1`
   - 再 `B=2`
   - 最后 `B=4`

4. **最后补后验多样性与泛化**
   - 这两组最适合在主方法已稳定后做

## 14. 最小可行全局消融（如果算力有限）

如果算力不够，不要一口气做满全部 7 张表。最少做这 4 张：

1. `系统级递进表`
2. `LatentGreen 代理表`
3. `采样层表`
4. `预算表`

这四张已经足够支撑：

- 这是一个完整算法而不是工程拼装
- 物理代理和采样校正都是必要的
- 主动高保真在有限预算下更优

## 15. 最终建议的论文主结论对应关系

为了让全文逻辑最硬，建议主结论和图表一一对应：

1. **系统级递进表**
   证明：这是一个完整算法，而不是组件堆叠。

2. **代理层消融表**
   证明：`LatentGreen` 及其校准训练原则是必要的。

3. **采样层消融表**
   证明：PMD 不是 heuristic，而是结构性改进。

4. **预算表**
   证明：主动高保真正在有限昂贵计算预算下更有效。

5. **后验多样性图**
   证明：方法在做后验生成，不是单点回归。
