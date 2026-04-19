# ==============================================
# 修复 PyTorch 2.7 权重加载报错（必须放在最最顶部）
# ==============================================
import torch
from ultralytics.nn.tasks import DetectionModel
torch.serialization.add_safe_globals([DetectionModel])
# ✅ 添加更多安全全局对象
import torch.nn as nn
torch.serialization.add_safe_globals([
    nn.Sequential,
    nn.ModuleList,
    nn.ModuleDict,
    # 如果还有其他错误，逐步添加
])
# ==============================================
# 正常训练代码 + SIoU 损失
# ==============================================
from ultralytics import YOLO
from ultralytics.utils import metrics
from siou_loss import bbox_iou

# 替换 SIoU
original_iou = metrics.bbox_iou
metrics.bbox_iou = lambda *args, **kwargs: bbox_iou(*args, SIoU=True, **kwargs)

def main():
    # 加载模型
    model = YOLO("yolov8n.yaml")  # 纯模型结构

    # 训练（关闭AMP检查，避免二次加载权重，彻底解决报错）
    model.train(
        data="rebar.yaml",
        epochs=100,
        batch=4,       # RTX3050 4GB 最稳
        imgsz=640,
        device=0,
        amp=False,     # 关键：关闭AMP检查，彻底不触发报错
        workers=4,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        box=7.5,
        cls=0.5,
        single_cls=True,
        dfl=1.5,
        patience=20,
        resume=True,
    )

if __name__ == "__main__":
    main()