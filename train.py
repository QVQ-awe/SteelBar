# train.py
from ultralytics import YOLO
import torch
from siou_loss import bbox_iou

# 替换YOLOv8默认的CIoU为SIoU
from ultralytics.utils.loss import BboxLoss
original_bbox_iou = BboxLoss.bbox_iou # type: ignore
BboxLoss.bbox_iou = lambda self, *args, **kwargs: bbox_iou(*args, SIoU=True, **kwargs) # type: ignore

def main():
    # 加载基础模型（论文使用YOLOv8n）
    model = YOLO('yolov8n.yaml')  # 使用修改后的C2f模块的配置文件
    
    # 训练参数（完全匹配论文）
    model.train(
        data='rebar.yaml',  # 数据集配置文件
        epochs=100,
        batch_size=16,
        imgsz=640,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # 使用论文生成的Anchor（替换默认值）
        anchors=[[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]],
        device='0',  # 使用GPU
        workers=8,
        project='rebar_count',
        name='yolov8n_cbam_siou',
        exist_ok=True
    )

if __name__ == '__main__':
    main()