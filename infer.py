# ==============================================
# 🔥 终极修复：PyTorch 2.6 + weights_only=False
# ==============================================
import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, Bottleneck

# ✅ 修补 torch_safe_load 函数
from ultralytics.nn import tasks
original_torch_safe_load = tasks.torch_safe_load

def patched_torch_safe_load(file):
    """修补版本：加载权重时允许 weights_only=False"""
    return torch.load(file, map_location='cpu', weights_only=False), file

tasks.torch_safe_load = patched_torch_safe_load

# 白名单
torch.serialization.add_safe_globals([
    DetectionModel,
    Conv,
    Conv2d,
    BatchNorm2d,
    SiLU,
    C2f,
    Bottleneck,
    nn.Sequential,
    nn.ModuleList,
    nn.ModuleDict,
])

# ==============================================
# 推理代码
# ==============================================
from ultralytics import YOLO
import cv2
import numpy as np

def count_rebar(image_path, model_path='runs/detect/train5/weights/best.pt', conf_thres=0.25):
    """成捆钢筋计数主函数"""
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    results = model(img, conf=conf_thres, iou=0.45)
    rebar_count = len(results[0].boxes)
    annotated_img = results[0].plot(
        conf=False,      # 隐藏置信度
        labels=False,    # 隐藏标签
        boxes=True       # 保留检测框
    )
    #annotated_img = results[0].plot(boxes=True, conf=False)
    #annotated_img = results[0].plot()
    output_path = image_path.replace('.jpg', '_result.jpg')
    cv2.imwrite(output_path, annotated_img)
    print(f"检测完成，钢筋数量：{rebar_count}")
    print(f"结果已保存至：{output_path}")
    return rebar_count, annotated_img

if __name__ == '__main__':
    count_rebar('test_rebar.jpg')