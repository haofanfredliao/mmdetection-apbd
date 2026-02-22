# 继承基础配置
_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 1. 修改模型类别数 (我们只有 1 类：field)
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)
    )
)

# 2. 修改数据集配置
dataset_type = 'CocoDataset'
data_root = 'data/data_coco/' 

# 定义类别名称，必须与 JSON 中的一致
metainfo = {
    'classes': ('field', ),
    'palette': [
        (220, 20, 60),
    ]
}

train_dataloader = dict(
    batch_size=4, # 根据你的 GPU 显存调整 (512x512 图像 4-8 应该没问题)
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/')
    )
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/')
    )
)

test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test/')
    )
)

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val.json')
test_evaluator = dict(ann_file=data_root + 'annotations/instances_test.json')

# 3. 训练策略微调 (可选)
# 默认的 schedule_1x 是 12 个 epoch。
# 如果你的数据集较小，可以适当减小学习率
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)

# 设置工作目录，保存权重和日志
work_dir = './work_dirs/mask-rcnn_r50_fpn_1x_ai4b'

# 加载 COCO 预训练权重以加速收敛
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'