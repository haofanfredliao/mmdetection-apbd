import os
import glob
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS

def visualize_val_results(config_path, checkpoint_path, val_img_dir, out_dir, score_thr=0.5, num_images=20):
    """
    对验证集进行推理并保存可视化结果
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. 初始化模型
    print("正在初始化模型...")
    model = init_detector(config_path, checkpoint_path, device='cuda:0')
    
    # 2. 初始化可视化工具 (会自动读取 config 中的 visualizer 配置)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # 将模型的数据集元信息（如类别名称）传给可视化工具
    visualizer.dataset_meta = model.dataset_meta
    
    # 3. 获取验证集图片路径
    img_paths = glob.glob(os.path.join(val_img_dir, '*.png')) # 之前预处理保存的是 png
    
    # 为了快速查看，默认只取前 num_images 张图，如果你想看全部，可以注释掉下面这行
    img_paths = img_paths[:num_images] 
    
    print(f"找到 {len(img_paths)} 张图片，开始推理...")
    
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        out_path = os.path.join(out_dir, img_name)
        
        # 读取图像 (MMDet 3.x 推荐使用 RGB 顺序进行可视化)
        img = mmcv.imread(img_path, channel_order='rgb')
        
        # 4. 执行推理
        result = inference_detector(model, img)
        
        # 5. 可视化并保存
        visualizer.add_datasample(
            name=img_name,
            image=img,
            data_sample=result,
            draw_gt=False,        # 是否同时画出真实标注(这里设为False只看预测)
            show=False,           # 是否弹窗显示(服务器上通常设为False)
            out_file=out_path,    # 保存路径
            pred_score_thr=score_thr # 置信度阈值，低于此分数的预测将被过滤
        )
        print(f"已保存可视化结果: {out_path}")

if __name__ == '__main__':
    # ================= 请修改以下路径 =================
    # 1. 你训练时使用的配置文件路径
    CONFIG = 'configs/ai4boundaries/mask2former_r50_1xb2-50e_custom.py' 
    
    # 2. 训练生成的权重文件路径 (通常在 work_dirs/你的模型名/ 下，找 best_xxx.pth 或 epoch_xx.pth)
    CHECKPOINT = 'work_dirs/mask2former_r50_1xb2-50e_custom/best_coco_segm_mAP_iter_129850.pth' 
    
    # 3. 验证集图片的路径
    VAL_IMG_DIR = 'data/data_coco/images/val' 
    
    # 4. 可视化结果保存的文件夹名称
    OUT_DIR = 'vis_results_mask2former' 
    # ==================================================
    
    visualize_val_results(CONFIG, CHECKPOINT, VAL_IMG_DIR, OUT_DIR, score_thr=0.5)