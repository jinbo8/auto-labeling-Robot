import json
import os
from PIL import Image, ImageDraw

trajectory_final = "/home/jin/6t/item/auto-labeling-Robot/data_clear/json/carrot.json"
img_dir = "/home/jin/6t/item/auto-labeling-Robot/data_clear/image/carrot.png"
output_img_dir = "/home/jin/6t/item/auto-labeling-Robot/data_clear/res"

os.makedirs(output_img_dir, exist_ok=True)

with open(trajectory_final, "r", encoding="utf-8") as f:
    data = json.load(f)
if isinstance(data, dict):
    data = [data]

for item in data:
    filepath = img_dir if os.path.isfile(img_dir) else os.path.join(img_dir, str(item["id"]))
    points = item["points"]

    image = Image.open(filepath).convert("RGB")
    draw = ImageDraw.Draw(image)
    color = (255, 0, 0)
    thickness = 2

    scaled_points = [(point[0], point[1]) for point in points]
    for i in range(len(scaled_points) - 1):
        draw.line([scaled_points[i], scaled_points[i + 1]], fill=color, width=thickness)
    for x, y in scaled_points:
        r = 3
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color)

    output_img_path = os.path.join(output_img_dir, os.path.basename(filepath))
    print(f"Input filepath: {filepath}")
    print(f"Output image path: {output_img_path}")
    image.save(output_img_path)
