import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import OrderedDict

from dask.array import map_overlap
from segmentation_models_pytorch import create_model

# ========== 参数 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_PATH = "8.png"  # 你的输入图像
MODEL_PATH = "best_model.pt"  # 模型路径
SAVE_DIR = "feature_maps"  # 保存特征图的目录
RESIZE_TO = 512  # 每张输出图大小（px）

# ========== 特征提取 ==========
feature_maps = OrderedDict()

def hook_fn(name):
    def hook(module, input, output):
        feature_maps[name] = output.detach().cpu()
    return hook

def register_hooks(model):
    # 注册 encoder 的 P2–P5
    encoder_layers = {
        "P2": model.encoder.layer1,  # 1/4
        "P3": model.encoder.layer2,  # 1/8
        "P4": model.encoder.layer3,  # 1/16
        "P5": model.encoder.layer4,  # 1/32
    }
    for name, layer in encoder_layers.items():
        layer.register_forward_hook(hook_fn(name))

    # 注册 decoder 的 D0–D4（U-Net有5层）
    for i, block in enumerate(model.decoder.blocks):
        name = f"D{i}"  # D0 ~ D4
        block.register_forward_hook(hook_fn(name))

# ========== 图像预处理 ==========
def preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# ========== 保存特征图 ==========
def save_feature_maps(feature_maps, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)

    for name, fmap in feature_maps.items():
        fmap_avg = torch.mean(fmap[0], dim=0)  # 通道平均后 [H, W]
        fmap_norm = (fmap_avg - fmap_avg.min()) / (fmap_avg.max() - fmap_avg.min() + 1e-8)
        fmap_np = (fmap_norm.numpy() * 255).astype(np.uint8)
        fmap_resized = cv2.resize(fmap_np, (RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_NEAREST)

        save_path = os.path.join(save_dir, f"{name}.png")
        cv2.imwrite(save_path, fmap_resized)
        print(f"[✓] Saved: {save_path}")

# ========== 主程序 ==========
if __name__ == "__main__":
    print("🚀 Loading model...")
    model = torch.load("best_model.pt", map_location=DEVICE)
    model.to(DEVICE).eval()

    map_overlap(DEVICE).TOP

    print("🔧 Registering hooks...")
    register_hooks(model)

    print("📷 Preprocessing image...")
    input_tensor = preprocess_image(IMAGE_PATH)

    print("🔍 Running inference to capture features...")
    with torch.no_grad():
        _ = model(input_tensor)

    print("💾 Saving feature maps...")
    save_feature_maps(feature_maps)

    print("✅ Done! All feature maps saved to:", SAVE_DIR)
