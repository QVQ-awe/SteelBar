# anchor_clustering.py
import numpy as np
from sklearn.cluster import KMeans
import os
import xml.etree.ElementTree as ET

def load_annotations(anno_dir):
    """加载YOLO格式标注文件，提取所有bbox的宽高"""
    boxes = []
    for file in os.listdir(anno_dir):
        if file.endswith('.txt'):
            with open(os.path.join(anno_dir, file), 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        w = float(parts[3])
                        h = float(parts[4])
                        boxes.append([w, h])
    return np.array(boxes)

def kmeans_plus_plus_anchors(boxes, num_anchors=9):
    """使用K-means++聚类生成Anchor尺寸"""
    kmeans = KMeans(n_clusters=num_anchors, init='k-means++', random_state=42)
    kmeans.fit(boxes)
    anchors = kmeans.cluster_centers_
    # 按面积排序
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(areas)
    anchors = anchors[sorted_indices]
    return anchors

if __name__ == '__main__':
    anno_dir = 'datasets/rebar/labels/train'  # 你的训练集标注路径
    boxes = load_annotations(anno_dir)
    anchors = kmeans_plus_plus_anchors(boxes, num_anchors=9)
    print("生成的Anchor尺寸（归一化）：")
    print(anchors)
    # 转换为YOLO格式（乘以输入尺寸640）
    print("\nYOLO格式Anchor（输入尺寸640）：")
    yolo_anchors = (anchors * 640).astype(int)
    print(yolo_anchors.flatten().tolist())