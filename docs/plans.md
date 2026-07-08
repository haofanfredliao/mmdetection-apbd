## 三个问题的定性与优先级


| 问题              | 性质              | 在哪一层解              | 数据现状                       | 优先级          |
| --------------- | --------------- | ------------------ | -------------------------- | ------------ |
| **过/欠分割 OS/US** | 实例分解错误（与边界形状正交） | 推理后处理 + 剖分先验注入     | OS=0.69（巨大）/ US=0.056（已很低） | **最高**（数字最大） |
| **边界密合**        | 栅格定位精度          | 网络 loss（你已有基建）     | 密合度随简化流失（IoU 0.99→0.83）    | 中高（mAP75 敏感） |
| **边界规整**        | 边界形状光滑/直        | 一半矢量化（简化即规整）+ 一半网络 | eps=1 就能 11 顶点/IoU0.97     | 中低（增量）       |


关键判断：**OS=0.69 是压倒性瓶颈**，而且它跟"边界"无关——是 Mask2Former 输出彼此独立可重叠的 mask、不利用"农田=平面剖分（互不重叠）"这个强先验导致的。US 已经很低（0.056），不用急。所以主力应放在 OS/US，边界密合次之，规整最后做增量。

---

## E0 — 统一度量套件（前置，必做，不训练）

没有可比的度量，后面所有实验无法证伪。扩展 `FieldSegmentationMetric` 或新建一个 metric，**每次实验统一输出**：

- 已有：`segm_mAP / mAP50 / mAP75`、`OS-Rate / US-Rate`
- **新增边界密合**：Boundary-IoU、Boundary-F@{1,3,5}px（Cheng et al. 的 boundary metric）
- **新增规整度**：`vertices@IoU0.95`（达到 0.95 IoU 所需顶点数，越少越规整——直接复用我刚写的 `si_diagnostic` 逻辑）、平均边界曲率能量
- **新增实例计数**：预测/GT 实例数比、重复预测率（duplicate rate）

先在 baseline / v1 / v2 三个 checkpoint 上跑一遍建 baseline 表。这一步会告诉我们 OS 里有多少是"重复 mask"（可后处理消除）vs 真实碎块。

---



## Track 1：OS/US（最高优先，三级递进）

**E1 — 推理端过滤（最便宜，先做）**

- 假设：OS=0.69 里很大一部分是低置信重复 mask。
- 动作：扫 `test_cfg` 的 score 阈值；加 **mask-NMS/去重**（高 IoU 的预测 mask 合并）；试 Mask2Former 的 **panoptic/argmax 式推理**（每像素 argmax 归属唯一实例，结构上强制不重叠）替代当前 `instance_on=True` 的独立 mask。
- 决策门：若 OS 从 0.69 大幅下降 → 说明大半是重复，问题降级；残余才是真碎块。

**E2 — 非重叠先验注入 loss（结构轻量，核心创新点）**

- 把"田块互不重叠"直接写进 Mask2Former 训练，无需改架构：


\mathcal{L}*{\text{overlap}}=\frac{1}{|X|}\sum*{x}\text{ReLU}\Big(\sum_{q\in \text{pos}} p_q(x)-1\Big)
\quad\text{或}\quad
\sum_{q\neq q'}\sum_x p_q(x)p_{q'}(x)


- matched 正样本 query 一一对应 GT 实例，而 GT 田块互不重叠，所以惩罚正样本间重叠是与 GT 一致的强监督，低风险。可在 V2 已有的 2D 边界分支上算。
- 消融：仅正样本 vs top-k 高分 query（后者才压推理期重复）；`loss_weight` 网格。

**E3 —（可选/并行，高天花板）边界检测+分水岭重构**

- 农田边界提取文献主流（Waldner & Diakogiannis：extent+boundary+distance 三任务 → seeded watershed）。结构上保证不重叠、无自交，OS/US 由 seed 数控制。
- 明确标注为高风险大改，作为 Mask2Former 触顶后的 Plan B，不首发。

---



## Track 2：边界密合（次优先，复用你的 V2 分支）

**E4 — 强化边界监督**

- (a) **边界带加权采样**：把 dice/CE 的点采样偏向 GT 边界带（当前是 uncertainty 采样），零显存代价提升边界锐度。
- (b) **距离变换加权 loss** 或 **Active Boundary Loss / Kervadec Boundary Loss**：用 GT 边界的距离场惩罚预测边界的位移，直接优化边界定位（比 Dice 更对边界敏感）。
- (c) 提升 mask 解码分辨率 / 边界 loss 分辨率的权重与 `boundary_max_res` 调参。
- 目标指标：Boundary-IoU / F@1px、mAP75。消融 a/b/c 单项与组合。



## Track 3：边界规整（最后做增量）

**E5 — 保角法向曲率正则**

- \mathcal{L}*{\text{curv}}=\frac{1}{|B|}\sum b^**{xy}\max(0,|\kappa_{xy}|-\tau)^2，\kappa=\nabla\cdot(\nabla p/\nabla p)，\tau 取 GT 拐角曲率的高分位数以**保护 90° 直角**。
- 目标指标：`vertices@IoU0.95` 下降、曲率能量下降，且 mAP 不掉。
- 消融：全曲率 vs 仅法向反号 vs hinge——验证保角必要性。
- 注意：规整很大程度矢量化时（eps≥1）已免费获得，E5 只处理"简化后仍残留的锯齿"，预期是小增量，别投太多。

---



## 实验规范（避免上次 v1 对比不公平的坑）

- 固定 seed、**对齐** `max_iters` **与 LR schedule**（v1 曾多训 34%）；每配置 ≥2 seed 看方差。
- 所有实验用 E0 统一度量套件输出，进同一张对比表。
- 一次只动一个变量；组合实验放在单项验证之后。



## 建议执行顺序与决策流

```
E0 度量套件(必做)
        │
        ▼
E1 推理过滤 ──OS大降?──是──► OS 问题降级，转 E4 边界密合
        │否
        ▼
E2 非重叠 loss ──► 攻 OS/US 主体
        │
        ▼
E4 边界密合 ──► E5 规整(增量) ──► (触顶再考虑 E3 分水岭重构)
```



## 代码落点

- 度量：`mmdet/evaluation/metrics/field_metric.py`（加 Boundary-IoU/F、vertices@IoU、计数）
- loss：`mmdet/models/losses/boundary_loss.py` 加 `NonOverlapLoss`/`CurvatureLoss`/`ActiveBoundaryLoss` → `__init__.py` 导出
- 接入：`mmdet/models/dense_heads/mask2former_head_v2.py` 的 `_loss_by_feat_single` / `loss_by_feat`
- 推理：`test_cfg` + panoptic fusion head 的 argmax 路径
- 配置：`configs/ai4boundary/` 逐个新建 `_v3_*.py`

---

我的建议是**从 E0 度量套件开始**——它是所有实验的前置，且能立刻用现有 3 个 checkpoint 量化出"OS 里多少是重复 mask、边界密合差距多大"，直接决定 E1 之后往哪打。

要不要我现在就把 **E0 度量套件**实现出来（在 `FieldSegmentationMetric` 里加 Boundary-IoU/F@px、[vertices@IoU0.95](mailto:vertices@IoU0.95)、实例计数/重复率）？如果你能给一个 checkpoint 或已有预测 json，我顺手把"在预测上跑全套度量"的脚本也搭好。