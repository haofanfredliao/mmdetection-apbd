# 继承官方的 Mask2Former ResNet-50 基础配置
_base_ = ['../mask2former/mask2former_r50_8xb2-lsj-50e_coco.py']

# ================= 1. 数据集与类别信息配置 =================
data_root = 'data/data/ai4b_coco/' 
metainfo = dict(
    classes=('field', ),
    palette=[(220, 20, 60)]
)

# ================= 2. 模型结构配置 =================
num_things_classes = 1
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes

model = dict(
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1]),
        # 覆写默认的 DiceLoss，使用自定义边界损失
        loss_dice=dict(
            type='BoundaryDiceLoss', 
            loss_weight=5.0,        # 保持与 mask2former 默认的 dice loss 权重一致
            kernel_size=3,
            eps=1e-5)
    ),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
    test_cfg=dict(panoptic_on=False, instance_on=True, semantic_on=False)
)

# ================= 3. 数据加载器配置 (适配 H100 80GB) =================
# 【修改说明】: H100 有 80GB 显存。
# 如果你的输入是 1024x1024，单卡 batch_size 可以开到 16。
# 如果你的输入是 512x512，单卡 batch_size 甚至可以开到 32 或 64。
# 这里以 batch_size=16 为例。同时增加 num_workers 防止 CPU 读图成为瓶颈。
train_dataloader = dict(
    batch_size=12,       # 从 2 提升到 16
    num_workers=8,       # 从 4 提升到 8 (如果 CPU 核心多，可以设为 16)
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/')))

val_dataloader = dict(
    batch_size=4,        # 验证集也可以适当增大 BS 加快验证速度
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/')))

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test/')))

# ================= 4. 评估器配置 =================
# 将原来的单一评估器改为列表，同时评估标准的 COCO 指标和自定义农田指标
val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/instances_val.json',
        metric=['bbox', 'segm'],
        format_only=False),
    dict(
        type='FieldSegmentationMetric',
        iou_thr=0.5)
]

test_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/instances_test.json',
        metric=['bbox', 'segm'],
        format_only=False),
    dict(
        type='FieldSegmentationMetric',
        iou_thr=0.5)
]

# ================= 5. 优化器与训练策略调整 (核心修改区) =================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00005,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,              
        max_keep_ckpts=3,        
        save_best='coco/segm_mAP', 
        rule='greater'))

max_iters = 22163
val_interval_iters = 444

train_cfg = dict(
    type='IterBasedTrainLoop', 
    max_iters=max_iters, 
    val_interval=val_interval_iters) # 每 444 次 (约1个Epoch) 验证一次

param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[int(max_iters * 0.9), int(max_iters * 0.95)],
    gamma=0.1)