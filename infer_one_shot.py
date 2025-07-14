import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from tqdm import tqdm

# 设置路径
image_dir = "../data/SMN/LR/image"
output_dir = "result_one_shot_0507/mask"
model_path = "result_one_shot_0507/best_model.pt"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()

# 定义预处理
transform = transforms.Compose([
    transforms.ToTensor()
])

# 遍历所有 PNG 图像
image_paths = [f for f in os.listdir(image_dir) if f.endswith(".png")]

for filename in tqdm(image_paths, desc="Predicting"):
    img_path = os.path.join(image_dir, filename)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Warning: Failed to read {filename}, skipping.")
        continue

    # Resize to 1024x1024
    image_resized = cv2.resize(image, (512, 512))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # 转为Tensor并预测
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy().astype(np.uint8)

    # 将 mask 映射为 0 和 255（黑白图）
    binary_mask = (pred_mask == 1).astype(np.uint8) * 255

    # 保存mask图像
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, binary_mask)


# for filename in tqdm(image_paths, desc="Predicting"):
#     img_path = os.path.join(image_dir, filename)
#     image = cv2.imread(img_path)
#
#     if image is None:
#         print(f"Warning: Failed to read {filename}, skipping.")
#         continue
#
#     # Resize to 1024x1024
#     image_resized = cv2.resize(image, (1024, 1024))
#     image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
#
#     # 转为Tensor并预测
#     input_tensor = transform(image_rgb).unsqueeze(0).to(device)
#     with torch.no_grad():
#         output = model(input_tensor)
#         pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy().astype(np.uint8)
#
#     # 创建叠加图像（红色区域）
#     overlay = image_resized.copy()
#     overlay[pred_mask == 1] = [0, 0, 255]  # 红色
#     blended = cv2.addWeighted(image_resized, 0.6, overlay, 0.4, 0)
#
#     # 保存图像
#     save_path = os.path.join(output_dir, filename)
#     cv2.imwrite(save_path, blended)

print("Batch inference completed.")
