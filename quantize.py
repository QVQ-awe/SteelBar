# infer.py
from ultralytics import YOLO
import cv2
import numpy as np

def count_rebar(image_path, model_path='runs/detect/yolov8n_cbam_siou/weights/best.pt', conf_thres=0.25):
    """成捆钢筋计数主函数"""
    # 加载模型
    model = YOLO(model_path)
    # 读取图像
    img = cv2.imread(image_path)
    # 推理
    results = model(img, conf=conf_thres, iou=0.45)
    # 统计钢筋数量
    rebar_count = len(results[0].boxes)
    # 绘制结果
    annotated_img = results[0].plot()
    # 保存结果
    output_path = image_path.replace('.jpg', '_result.jpg')
    cv2.imwrite(output_path, annotated_img)
    print(f"检测完成，钢筋数量：{rebar_count}")
    print(f"结果已保存至：{output_path}")
    return rebar_count, annotated_img

if __name__ == '__main__':
    count_rebar('test_rebar.jpg')