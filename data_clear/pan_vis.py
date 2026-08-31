import json
import os
import cv2
import numpy as np

img_dir = '/home/jin/6t/item/auto-labeling-Robot/data_clear/image/pan.png'
affordance_json = '/home/jin/6t/item/auto-labeling-Robot/data_clear/json/pan.json'
output_img_dir = '/home/jin/6t/item/auto-labeling-Robot/data_clear/res'

os.makedirs(output_img_dir, exist_ok=True)

with open(affordance_json, 'r', encoding='utf-8') as f:
    data = json.load(f)
if isinstance(data, dict):
    data = [data]

for item in data:
    filepath = img_dir if os.path.isfile(img_dir) else os.path.join(img_dir, str(item['id']))

    image = cv2.imread(filepath)
    if image is None:
        raise FileNotFoundError(f"读不到图像: {filepath}")
    color = (255, 0, 0)
    thickness = 2

    x_min = item['affordance']['x']
    y_min = item['affordance']['y']
    x_max = x_min + item['affordance']['width']
    y_max = y_min + item['affordance']['height']

    pts = np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=np.float32,
    )
    cv2.polylines(image, [pts.astype(int)], isClosed=True, color=color, thickness=thickness)

    output_img_path = os.path.join(output_img_dir, os.path.basename(filepath))
    print(f"Input filepath: {filepath}")
    print(f"Output image path: {output_img_path}")
    cv2.imwrite(output_img_path, image)
