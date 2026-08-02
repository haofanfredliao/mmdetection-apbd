术语沿用大纲：**planar partition (non-overlap) — topological bias**、**boundary adherence — geometric bias**、**boundary regularity — geometric bias**，对应 §3.1/3.2/3.3。正文是英文的，appendix 按英文写。

这一版对应 v5（`configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v5.py`），即正在训练的最终配置。与上一稿的实质性差异见文末「本次修正」。

---

# Appendix A. Implementation Details of Inductive-Bias Injection

## A.1 Base architecture and available attachment points

All experiments build on Mask2Former with a ResNet-50 backbone. The backbone emits four feature levels at strides 4/8/16/32 (256/512/1024/2048 channels) which are consumed directly by the multi-scale deformable-attention pixel decoder; no FPN neck is inserted. The transformer decoder has nine layers operating on 100 object queries, and the classification head is configured for a single thing class (`field`) with no stuff classes. Following the standard Mask2Former recipe, deep supervision is applied to all ten prediction sets (the query embedding before the first decoder layer, plus the output of each of the nine layers).

Two structural properties of this architecture determine where each inductive bias can be attached, and both are easy to overlook.

**Property 1 — the mask losses are not computed on dense maps.** Stock Mask2Former evaluates `loss_dice` and `loss_mask` on 12,544 coordinates drawn per instance by uncertainty-based importance sampling (oversample ratio 3.0, importance ratio 0.75), not on the dense mask logits. Any loss whose definition depends on spatial adjacency — a morphological boundary band, a curvature operator — is therefore meaningless if it is simply substituted for `loss_dice`. We established this empirically: an early version of this work replaced `loss_dice` with a boundary-band Dice loss and observed a silent no-op, because the boundary extraction operated on a scattered point set with no spatial layout. Losses of this kind must instead be attached as *separate dense-map auxiliary branches*, whereas losses that are pointwise in the prediction can be folded into the existing sampled path at almost no cost. Both routes are used below.

**Property 2 — Hungarian matching permutes but does not change the ground-truth set.** Bipartite matching is performed independently at every decoder layer, so the correspondence between queries and ground-truth instances differs from layer to layer. The *set* of ground-truth masks in a batch, however, is layer-invariant, and each ground-truth instance is matched exactly once. Any quantity derived purely from the ground truth — most importantly the signed distance transform of §A.3 — can therefore be computed once per iteration and re-indexed per layer by the matching permutation, rather than recomputed ten times.

All auxiliary losses operate on matched positive queries only, i.e. the subset of the 100 queries that bipartite matching assigned to a ground-truth instance. Each auxiliary term is additionally scaled by a per-image quality weight propagated through `img_meta['loss_weight']`, which down-weights annotations from the `2_lazy` quality stratum. This weighting is applied to `loss_dice` and to every auxiliary loss, but deliberately not to the cross-entropy `loss_mask`, which we found destabilising to reweight.

Table A.1 summarises where each bias attaches.

| Inductive bias             | Mechanism                   | Attachment point                                       | Stage     |
| -------------------------- | --------------------------- | ------------------------------------------------------ | --------- |
| Planar partition (§3.1)    | Boundary-band Dice loss     | Dense auxiliary branch, last decoder layer, ≤256 px    | Training  |
| Planar partition (§3.1)    | Non-overlap loss            | Dense auxiliary branch, last decoder layer, ≤128 px    | Training  |
| Planar partition (§3.1)    | Argmax instance assignment  | Panoptic fusion head                                    | Inference |
| Boundary adherence (§3.2)  | Kervadec surface loss       | Folded into the point-sampled path, all decoder layers | Training  |
| Boundary regularity (§3.3) | Polygon simplification      | Raster-to-vector post-processing                        | Inference |

## A.2 Topological bias: planar partition

The topological prior — agricultural parcels tessellate the plane and do not overlap — is injected at both stages: two dense auxiliary losses encourage disjointness during training, and the fusion head enforces it structurally at inference.

**Boundary-band Dice loss.** For each matched positive query, the dense mask logits at decoder resolution are bilinearly downsampled to at most 256 px per side and the ground-truth mask is nearest-downsampled to match. A soft boundary band is extracted from the predicted probability map and a hard band from the ground truth using a differentiable morphological difference, dilation minus erosion, realised with `max_pool2d` on the map and its complement respectively (kernel size 3). The Dice coefficient is then computed between the two bands. Sharpening the shared edge between adjacent parcels is what makes them separable as distinct instances, which is why this term is grouped with the partition bias rather than with boundary adherence, although it contributes to both. The term carries weight 2.0.

**Non-overlap loss.** Let $p_q(x)$ denote the predicted foreground probability of matched positive query $q$ at pixel $x$, and let the positives be grouped by source image. Within each group we penalise the excess of the summed probabilities over unity,

$$\mathcal{L}_{\text{overlap}}=\frac{1}{|X|}\sum_{x\in X}\operatorname{ReLU}\Big(\sum_{q\in\text{pos}}p_q(x)-1\Big)^{2},$$

averaged over the images in the batch that contain at least two positives. The formulation is deliberately restricted to matched positives: these are in one-to-one correspondence with ground-truth instances, and since ground-truth parcels are mutually disjoint, penalising their overlap is strictly consistent with the annotation and introduces no supervision conflict. Extending the penalty to unmatched or top-scoring queries would target inference-time duplication more directly but is no longer guaranteed to agree with the ground truth, and is not used here. The term is evaluated at ≤128 px, which is sufficient because it measures area overlap rather than boundary placement. It carries weight 6.5, ramped from zero as described in §A.5.

Both dense terms are computed on the last decoder layer only. Replicating them across all ten prediction sets would multiply the dense intermediate tensors accordingly; restricting them to the final layer keeps the memory overhead of the auxiliary branch below that of the point-sampled losses it accompanies.

**Argmax instance assignment.** The two losses above shape the prediction but cannot guarantee a partition, so disjointness is finally imposed at inference by a modified panoptic fusion head. Each pixel is assigned to the single query with the highest score-weighted probability, and a query's output mask is the set of pixels assigned to it; overlap is thereby structurally impossible rather than merely penalised. Queries retaining less than a fraction $\tau_{\text{IoU}}=0.7$ of their pre-assignment area are discarded as duplicates, and queries scoring below 0.2 are dropped.

One detail of this operator is not optional and is easily got wrong. The argmax runs over the query axis only, and that axis has no background row — every pixel is necessarily claimed by some query, so the resulting masks tile the entire image. A second condition, retaining a pixel only where that query's own probability also exceeds 0.5, is the only mechanism that carves background back out. Measured on the test set with identical weights, enabling it improves every metric simultaneously: segm mAP 0.246→0.298, Boundary-IoU 0.458→0.520, Boundary-F$_{\text{1px}}$ 0.335→0.411, vertices@IoU0.95 26.6→12.3, and curvature energy 0.600→0.373. Without it the model produces the characteristic full-image tessellation in which every pixel belongs to some parcel.

The training-time and inference-time mechanisms are complementary rather than redundant. Argmax alone yields a hard partition of whatever the network produces, but it must break ties between queries that were never trained to disagree, and those ties are where spurious slivers originate; the non-overlap loss reduces the number of such ties in the first place.

## A.3 Geometric bias I: boundary adherence

Boundary adherence is injected with the surface loss of Kervadec et al., which multiplies the predicted foreground probability by a signed Euclidean distance map $\phi$ derived from the ground truth — negative inside the object, zero on the contour, positive outside — so that minimising $\mathbb{E}[p(x)\phi(x)]$ optimises the *location* of the boundary rather than the *area* of the region, complementing the region-based Dice and cross-entropy terms.

Rather than adding a further dense auxiliary branch, we fold this term into the existing point-sampled path. The signed distance map is computed once per iteration on the ground-truth masks of the batch (Euclidean distance transform on CPU, under `no_grad`, at 128 px and clipped to ±32 cells to bound gradient magnitude over large uniform backgrounds), and is then sampled at exactly the coordinates already drawn for `loss_dice` and `loss_mask` at that decoder layer:

$$\mathcal{L}_{\text{surface}}=\frac{1}{|S|}\sum_{i\in S}\sigma(z_i)\,\phi(x_i),$$

where $S$ is the set of sampled coordinates and $z_i$ the corresponding mask logit. Three properties follow, and together they are the reason we prefer this formulation to a dense one.

First, the sampling distribution is *aligned with the region where the loss has gradient*. Differentiating with respect to the logit gives $\partial\mathcal{L}/\partial z(x)=\phi(x)\sigma(z)(1-\sigma(z))/|S|$, which vanishes at both extremes: far outside the object the prediction is saturated and $\sigma(1-\sigma)\to0$, while on the contour itself $\phi\to0$. Effective gradient exists only in a thin annulus around the decision boundary. Uncertainty-based importance sampling concentrates coordinates precisely on that annulus, whereas a dense formulation averages the same signal over the entire crop.

The consequence is measurable, and it runs opposite to what the reported loss value suggests. Because the sampled band has $\phi\approx0$ by construction, the point-sampled term *prints a smaller number* than the dense one; but per unit of loss weight it delivers a gradient norm of 84.4 against the dense variant's negligible contribution (§A.5). The dense formulation is not weak because its weight is too small — it is weak because most of the pixels it averages over are saturated and contribute value without gradient. Raising the dense weight does not fix this: in a preliminary run at a tenfold weight the term still accounted for 0.4% of the objective and had converged, by iteration 5,000 of 16,500, to exactly the value attained by an exact prediction at that resolution, after which it carried no signal at all.

Second, because the distance map depends only on the ground truth, the layer-invariance noted in §A.1 applies. A naive implementation that recomputes the transform inside each layer's loss costs 2.82 s/iteration against a 1.64 s baseline; computing it once per iteration and re-indexing it with each layer's matching permutation brings this to 1.48 s, i.e. below the baseline, and the term is then supervised at every decoder layer for the cost of one additional point-sampling operation per layer. This restores consistency with Mask2Former's deep-supervision design, which the dense auxiliary branches of §A.2 must forgo for memory reasons. Since a mis-indexed permutation would silently optimise each mask against another parcel's distance field, the reordering is verified against a direct recomputation from each layer's own matched targets (`scripts/check_surface_cache.py`); the two agree bit-exactly on all ten layers.

Third, no dense intermediate tensor is materialised, so the memory cost is negligible.

## A.4 Geometric bias II: boundary regularity

Regularity is not injected during training. It is obtained at the raster-to-vector stage, which the downstream cadastral use case requires in any case, by simplifying the external contour of each predicted instance with the Douglas–Peucker algorithm. For each instance the tolerance $\varepsilon$ is selected as the smallest value on a fixed ladder for which the polygon rasterised back to the mask grid retains an IoU of at least 0.95 with the predicted raster mask, so that simplification is bounded by an explicit fidelity constraint rather than a global tolerance. This is the same procedure used by the `vertices@IoU0.95` metric of §3.4, which makes the reported regularity figures a direct measurement of the operator actually applied rather than a proxy.

Placing regularity entirely in post-processing is a deliberate choice supported by the exploratory analysis of §4.1, which showed that the vertex-count/fidelity trade-off is dominated by the simplification tolerance, and that a tolerance as small as one pixel already reduces a typical parcel to roughly eleven vertices at IoU 0.97. A training-time curvature regulariser was implemented and evaluated during development and is not part of the final model; Appendix B reports that negative result and its diagnosis.

## A.5 Calibrating the auxiliary loss weights

Three auxiliary terms accompany the two stock mask losses, and their relative weighting is the one design choice with no principled default. We set it by gradient norm rather than by loss magnitude, for a reason that the development history makes concrete: the dense surface loss of §A.3 was tuned by inspecting its printed value, and the resulting term was inert for the entire run. Loss magnitude and influence on the optimiser are different quantities, and for a term whose gradient vanishes wherever the sigmoid saturates they can differ by more than an order of magnitude.

The procedure (`scripts/probe_loss_gradients.py`) takes a single forward pass, backpropagates each term separately, and records the $L_2$ norm of the resulting gradient over all trainable parameters. Because a loss weight enters as a scalar factor, the gradient norm is exactly linear in it, and the weight achieving any target share follows in closed form. Two choices matter. The measurement is taken on a trained checkpoint (the §5 baseline at 16,500 iterations) rather than at initialisation, because these terms are designed to act on an already roughly-correct mask and their relative scale before that is not representative. And the reference is `loss_dice` summed over all ten prediction sets, since that is what the optimiser actually sees.

We adopt as the budget the share already occupied by the boundary-band Dice term at its inherited weight of 2.0, and scale the other two to match:

| Term              | Weight | Gradient norm, as share of `loss_dice` |
| ----------------- | ------ | -------------------------------------- |
| `loss_boundary`   | 2.0    | 9.5% (inherited, unchanged)            |
| `loss_nonoverlap` | 6.5    | 10% (3.1% at the initial weight of 2.0) |
| `loss_surface`    | 0.09   | 10% (84.4 per unit weight)             |

For the surface term this calibration reverses the direction of the intended correction. Read by loss value, the term looked far too weak and the weight was to be raised towards 0.5; read by gradient, 0.5 would have given it 57% of the dice gradient — no longer an auxiliary term at all — and the correct value is 0.09. The point-sampled formulation had already supplied the missing influence; the weight had to come down to accommodate it.

Both of the new terms are ramped linearly over the first 10% of iterations and held constant thereafter, for different reasons.

The surface term is ramped from $10^{-3}$ following the annealing schedule of the original formulation, which is motivated by instability early in training when predictions are far from the ground truth and gradients scale with the distance map. The end point of the ramp is placed early deliberately: in a preliminary run in which the ramp extended over the first 30% of training, the term reached full weight only at the point where it had already saturated, so that its full strength coincided with the interval in which it carried no remaining signal.

The non-overlap term is ramped from zero to avoid a large transient at initialisation. Before training, every matched query predicts approximately 0.5 everywhere, so with around twelve instances per image the summed probability is about 6 and the squared excess starts near 4 before weighting — 26% of the total objective, driving the gradient norm to 3,318 against a steady-state value near 150. The transient resolves itself within roughly sixty iterations, but while the masks are still noise the cheapest way to satisfy the constraint is to predict nothing anywhere, and there is no reason to expose the model to that gradient before its masks mean anything. With the ramp in place the gradient norm at the start of training is 240.

## A.6 Training and inference configuration

Training uses AdamW (learning rate $5\times10^{-5}$, weight decay 0.05, $\beta=(0.9,0.999)$) with a batch size of 12 on a single GPU, for 16,500 iterations with the learning rate decayed by a factor of ten at 90% and 95% of the schedule. Augmentation follows the large-scale jittering recipe: random horizontal flip, random resize with ratio range $[0.1,2.0]$, and random crop to $1024\times1024$. The training split contains 3,293 parcel-bearing tiles of the `1_good` quality stratum plus a deterministic 20% sample of empty-annotation tiles retained as negatives (3,952 tiles in total, approximately 330 iterations per epoch); the `3_extreme` stratum is excluded entirely and `2_lazy` tiles are down-weighted as described in §A.1. Validation and test splits are restricted to the `1_good` stratum so that reported figures are not confounded by annotation quality.

At inference, images are resized with aspect ratio preserved to a long side of 1,333 and short side of 800. Instance predictions are produced by the argmax fusion head of §A.2 with score threshold 0.2, duplicate-suppression threshold 0.7, per-pixel probability threshold 0.5, and at most 100 instances retained per image.

---

## 本次修正

改了四处，第一处是实质性的：

**A.3 / A.5：surface loss 的权重从 0.5 改为 0.09。** 上一稿把「提高权重」写成了结论，方向是反的。梯度标定显示点采样版每单位权重的梯度范数是 84.4，而十层 dice 加起来是 74.5——按 0.5 跑，这个辅助项会占到 dice 梯度的 57%。稠密版之所以无效不是权重不够，而是它平均的像素大多处于 sigmoid 饱和区，有数值没梯度；点采样已经补上了缺的影响力，权重反而要降下来。

**新增 A.5「权重标定」。** 三个辅助项的相对权重原本是全文唯一没有依据的设计选择，现在统一按「梯度范数占 dice 的 10%」标定，基准取自 BoundaryDiceLoss 在继承权重 2.0 下本来就落在的 9.5%。这一节同时把两个 ramp 的动机说清楚了，NonOverlapLoss 的爬坡是新加的（初期瞬态会把 grad_norm 顶到 3318）。

**A.2 补上 argmax 推理端。** 上一稿把它漏在了 Track 1 之外，但它是唯一真正保证剖分的机制，且 `filter_low_score` 那个坑必须写进论文——不写的话别人复现只会得到布满全图的分割。同时说明了它与 NonOverlapLoss 为什么互补而不是重复。

**A.3 补上距离图缓存与其验证。** 每层重算会让训练从 1.64 慢到 2.82 s/iter，改成每轮算一次后是 1.48 s/iter；索引错位会静默训错，所以写明了逐位比对的验证。

还剩两个待办：**Appendix B**（曲率正则的 negative result）我在 A.4 结尾引用了但还没写；**§5 的数字**要等 v5 训完（约 6 小时 55 分）才能填。另外 BoundaryDiceLoss 目前按你的指定归在 Track 1，理由写在 A.2，如果你想挪到 §3.2 只需移动那一段。
