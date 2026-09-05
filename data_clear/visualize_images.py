import json
import os
import cv2
import numpy as np

img_dir = './images'
affordance_json = './affordance.json'
output_img_dir = './visualized_images'

count = 0
with open(affordance_json, 'r') as f:
    data = json.load(f)
    for item in data:
        filepath = os.path.join(img_dir, item['image_path'])
        
        image = cv2.imread(filepath)
        color = (255, 0, 0)
        thickness = 2

        x_min,y_min = item['affordance']['x'], item['affordance']['y']
        x_max,y_max = item['affordance']['x']+item['affordance']['width'], item['affordance']['y']+item['affordance']['height']

        # 定义矩形的四个顶点坐标
        pts = np.array([
            [x_min, y_min],  # 左上角
            [x_max, y_min],  # 右上角
            [x_max, y_max],  # 右下角
            [x_min, y_max]   # 左下角
        ], dtype=np.float32)

        # 绘制矩形框
        cv2.polylines(image, [pts.astype(int)], isClosed=True, color=color, thickness=thickness)

        # 获取相对路径并拼接目标路径
        relative_path = os.path.relpath(filepath, img_dir)  # 获取相对于 img_dir 的相对路径
        output_img_path = os.path.join(output_img_dir, relative_path)  # 拼接目标路径

        # 创建目标文件夹
        output_directory = os.path.dirname(output_img_path)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # 打印调试信息
        print(f"Input filepath: {filepath}")
        print(f"Output image path: {output_img_path}")
        print(f"Output directory: {output_directory}")

        # 保存图像
        count += 1
        cv2.imwrite(output_img_path, image)

print(count)