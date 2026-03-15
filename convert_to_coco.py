import os
import glob
import json
import numpy as np
import tifffile
import cv2
from tqdm import tqdm

def create_coco_format(image_dir, mask_dir, output_img_dir, output_json_path):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "field"}] # 假设只有一类：地块
    }
    
    annotation_id = 1
    
    # 获取所有图像文件
    image_paths = glob.glob(os.path.join(image_dir, '*.tif'))
    
    for img_id, img_path in enumerate(tqdm(image_paths, desc=f"Processing {os.path.basename(image_dir)}")):
        filename = os.path.basename(img_path)
        # 根据你的命名规则推断 mask 文件名
        mask_filename = filename.replace('_ortho_1m_512.tif', '_ortholabel_1m_512.tif')
        mask_path = os.path.join(mask_dir, mask_filename)
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {filename}")
            continue
            
        # 1. 读取图像并保存为 PNG (丢弃地理信息，仅用于视觉训练)
        img_data = tifffile.imread(img_path)
        # 确保是 512x512x3 的 uint8
        if img_data.dtype != np.uint8:
            img_data = img_data.astype(np.uint8)
        
        png_filename = filename.replace('.tif', '.png')
        png_path = os.path.join(output_img_dir, png_filename)
        # tifffile 读入通常是 RGB，cv2 保存需要 BGR
        cv2.imwrite(png_path, cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))
        
        height, width = img_data.shape[:2]
        coco_dict["images"].append({
            "id": img_id,
            "file_name": png_filename,
            "width": width,
            "height": height
        })
        
        # 2. 读取 Mask 并提取实例
        mask_data = tifffile.imread(mask_path)
        # 提取第4个通道 (索引为3)
        instance_mask = mask_data[:, :, 3]
        
        # 获取所有唯一的实例 ID，排除背景 -10000
        instance_ids = np.unique(instance_mask)
        instance_ids = instance_ids[instance_ids != -10000]
        
        for inst_id in instance_ids:
            # 创建当前实例的二值掩码
            binary_mask = (instance_mask == inst_id).astype(np.uint8)
            
            # 提取轮廓
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            segmentation = []
            for contour in contours:
                contour = contour.flatten().tolist()
                # 只有包含至少3个点（6个坐标）的多边形才是有效的
                if len(contour) > 4:
                    segmentation.append(contour)
            
            if len(segmentation) == 0:
                continue
                
            # 计算边界框 [x, y, width, height]
            pos = np.where(binary_mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            bbox = [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]
            area = int((xmax - xmin) * (ymax - ymin)) # 简化面积计算，或使用 cv2.contourArea
            
            coco_dict["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            })
            annotation_id += 1

    with open(output_json_path, 'w') as f:
        json.dump(coco_dict, f)
    print(f"Saved COCO json to {output_json_path}")

# 假设你的原始数据在当前目录的 data 文件夹下
base_dir = os.path.expanduser("~/data/ai4b")
out_base_dir = os.path.expanduser("~/data/ai4b_coco")

for split in ['train', 'val', 'test']:
    print(f"Converting {split} set...")
    create_coco_format(
        image_dir=os.path.join(base_dir, f'images/{split}'),
        mask_dir=os.path.join(base_dir, f'masks/{split}'),
        output_img_dir=os.path.join(out_base_dir, f'images/{split}'),
        output_json_path=os.path.join(out_base_dir, f'annotations/instances_{split}.json')
    )