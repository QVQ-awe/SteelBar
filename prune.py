# prune.py
from ultralytics import YOLO
import torch
import torch.nn as nn

def structured_prune(model, prune_ratio=0.7):
    """基于L1范数的结构化剪枝（移除卷积核）"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.weight.requires_grad:
            # 计算每个卷积核的L1范数
            weight = module.weight.data.abs().mean(dim=(1,2,3))
            # 确定剪枝阈值
            threshold = torch.quantile(weight, prune_ratio)
            # 保留重要性高于阈值的卷积核
            keep_indices = torch.where(weight > threshold)[0]
            # 剪枝卷积层
            module.weight.data = module.weight.data[keep_indices, :, :, :]
            if module.bias is not None:
                module.bias.data = module.bias.data[keep_indices]
            # 更新输出通道数
            module.out_channels = len(keep_indices)
    return model

if __name__ == '__main__':
    # 加载训练好的模型
    model = YOLO('runs/detect/yolov8n_cbam_siou/weights/best.pt')
    # 结构化剪枝（论文70%剪枝比例）
    pruned_model = structured_prune(model.model, prune_ratio=0.7)
    # 保存剪枝后的模型
    torch.save(pruned_model.state_dict(), 'runs/detect/yolov8n_cbam_siou/weights/pruned_best.pt')
    print("剪枝完成，模型已保存")