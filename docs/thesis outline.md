# Thesis Outline

**Title:** Field Parcel Instance Segmentation from Remote Sensing Imagery Using Deep Learning
**Subtitle:** Injecting Geometric and Topological Inductive Biases into a Query-based Segmentation Model

**Target length:** 8,000–10,000 words

**Organizing principle (核心主线):**

- **Topological bias** → Planar partition (non-overlap / connectivity)
- **Geometric bias** → Boundary adherence + Boundary regularity
- **Narrative spine (诊断驱动):** 度量诊断推翻表象 (OS 假象) → 重新定位真问题 (边界密合) → 针对性注入偏置 → 验证

**Suggested word budget:**


| Section                 | Words |
| ----------------------- | ----- |
| Abstract                | 250   |
| 1. Introduction         | 1,200 |
| 2. Literature Review    | 2,000 |
| 3. Methodology          | 2,500 |
| 4. Experiments          | 1,200 |
| 5. Results & Case Study | 1,800 |
| 6. Discussion           | 800   |
| 7. Conclusion           | 400   |


---



## Abstract ★

- 一段式：问题 → 方法 (三偏置注入) → 关键发现 (度量诊断 + 各偏置贡献) → 结论。



## 1. Introduction



### 1.1 Background and Rationale

- 农田地块提取的应用价值 (补贴核查、产量估计、地块级管理)；遥感影像获取现状。



### 1.2 Problem Statement

- 农田的领域特性：单类、平面剖分 (互不重叠、铺满)、边界规则密合、内部纹理不重要。
- 通用实例分割 (Mask2Former) 未编码这些先验 → 具体失效模式。
- ★ **Pilot study:** 多边形自回归头 (polyseq/polyformer) 连小样本都无法过拟合 → 论证放弃"从零生成顶点"、回归 dense-raster query-based 模型的依据。



### 1.3 Contributions & Research Questions ★

- 显式列 3–4 条贡献 (含"度量发现"作为一级贡献)。
- 显式列 RQ / 假设。



## 2. Literature Review (two scopes)



### 2.1 Taxonomy of parcel extraction (RS/GIS scope)

- semantic / instance / boundary-based 三类范式；分水岭/extent-boundary-distance 多任务谱系。



### 2.2 Instance segmentation in computer vision (DL scope)

- query-based 实例分割 (DETR → MaskFormer → Mask2Former)。
- boundary-aware losses (Kervadec Boundary Loss / Active Boundary Loss / Boundary IoU 谱系)。
- ★ polygon / contour representations (Polygon-RNN, Curve-GCN, Deep Snake, PolyWorld, BoundaryFormer)；点明"从零自回归生成难训"这一已知难题，呼应 pilot。



## 3. Methodology — Injecting Inductive Biases



### 3.0 Definition & mapping ★

- 广义界定 "inductive bias" = 任何编码领域知识的机制 (架构 / 损失 / 推理)，预防"loss 不算 bias"的质疑。
- 三偏置 ↔ 几何/拓扑映射图。



### 3.1 Planar partition (non-overlap) — topological

- 非重叠先验注入 loss；推理端去重 (score 阈值 + mask-NMS)。



### 3.2 Boundary adherence — geometric

- 边界带加权采样 / 距离变换加权 loss / (Active) Boundary Loss；复用 V2 边界分支。



### 3.3 Boundary regularity — geometric

- 保角法向曲率正则 (hinge 阈值保护 90° 直角)。



### 3.4 Evaluation criteria design

- ★ 动机：朴素 OS/US 无工作点 → 极大高估过分割 (度量发现)。
- 度量套件：mAP/50/75、Boundary-IoU、Boundary-F@{1,3,5}px、[vertices@IoU0.95](mailto:vertices@IoU0.95)、curvature energy、OS/US (带工作点)。



## 4. Experiments



### 4.1 Dataset and Explanatory Data Analysis

- 数据来源、划分、质量分层 (good/lazy/extreme)。
- ★ 诊断性 EDA：自交 = 多边形化产物 (SI-rate vs eps)、顶点数/eps/密合度权衡。



### 4.2 Architecture of model

- Mask2Former + 自定义 head (V2) 结构。



### 4.3 Experimental settings

- ★ 公平对比协议：对齐 max_iters/LR schedule、固定 seed、≥2 seed 报方差。



## 5. Results and Case Study



### 5.1 Quantitative comparison

- ★ 表格按 baseline → +partition → +adherence → +regularity 阶梯组织。
- ★ OS 分解结果 (阈值/NMS 扫描)，呼应 3.4 度量发现。



### 5.2 Visual comparison



### 5.3 Case study

- 典型成功/失败样例 (碎块、粘连、锯齿边界)。



## 6. Discussion



### 6.1 Strengths



### 6.2 Limitations ★

- 单数据集、单类别；OS 改善部分来自后处理/度量修正而非模型本身。



### 6.3 Future Improvements

- 边界检测 + 分水岭重构 (E3) 作为高天花板方向。



## 7. Conclusion



## References



## Appendices

