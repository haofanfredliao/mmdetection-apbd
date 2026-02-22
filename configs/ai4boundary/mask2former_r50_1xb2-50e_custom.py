# 继承官方的 Mask2Former ResNet-50 基础配置
# 该基础配置已经包含了 50 Epoch 训练策略和 LSJ 强数据增强
_base_ = ['../mask2former/mask2former_r50_8xb2-lsj-50e_coco.py']

# ================= 1. 数据集与类别信息配置 =================
data_root = 'data/data_coco/'
# 你的单类别配置
metainfo = dict(
    classes=('field', ),
    palette=[(220, 20, 60)]
)

# ================= 2. 模型结构配置 (适配单类别) =================
# Mask2Former 将类别分为 Things(实例) 和 Stuff(背景/语义)
# 我们的任务是纯实例分割，因此 Things=1, Stuff=0
num_things_classes = 1
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes

model = dict(
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        # 类别权重，背景类(no object)权重通常设为 0.1
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
    # 关闭全景分割和语义分割，只输出实例分割结果
    test_cfg=dict(panoptic_on=False, instance_on=True, semantic_on=False)
)

# ================= 3. 数据加载器配置 (适配显存) =================
# NVIDIA L4 有 24GB 显存，Mask2Former + LSJ(1024x1024) 比较吃显存
# 建议单卡 batch_size 设为 2
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/')))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/')))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test/')))

# ================= 4. 评估器配置 =================
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric=['bbox', 'segm'],
    format_only=False)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test.json',
    metric=['bbox', 'segm'],
    format_only=False)

# ================= 5. 优化器与训练策略调整 =================
# 官方默认是 8张卡 x batch_size 2 = 16 的学习率 (0.0001)
# 我们是单卡 batch_size 2，为了防止梯度爆炸，将学习率适当下调到 0.00005
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00005,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)))

# 训练 Epoch 保持基础配置的 50 Epoch 不变
# 增加 Checkpoint 钩子，自动保存验证集上 segm_mAP 最高的模型
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,              # 每 5 个 epoch 保存一次常规权重
        max_keep_ckpts=3,        # 最多保留 3 个最新权重，节省硬盘
        save_best='coco/segm_mAP', # 自动追踪并保存最佳的分割权重
        rule='greater'))

# 将最大迭代次数修改为适合你数据集的次数 (假设跑50个Epoch，大约132500次)
max_iters = 132500
train_cfg = dict(
    type='IterBasedTrainLoop', 
    max_iters=max_iters, 
    val_interval=2650) # 每隔多少次(大约1个Epoch)验证一次

# 学习率衰减策略也需要跟着缩放 (通常在 0.9 倍和 0.95 倍时衰减)
param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[int(max_iters * 0.9), int(max_iters * 0.95)],
    gamma=0.1)