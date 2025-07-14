import os
import cv2
import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from torchvision import transforms
import segmentation_models_pytorch as smp

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载并准备模型
model = torch.load("result_one_shot/50/best_model.pt", map_location=device)
model = model.float()  # ⭐ 关键：转成float32
model.to(device)
model.eval()


# 正确选择最后的特征层
target_layer = model.decoder.blocks[-1].conv1  # or conv2 if you want


# 初始化 Grad-CAM
cam = GradCAM(model=model, target_layers=[target_layer])

# 输入图像路径和输出路径
input_folder = "../data/SAC/image"
output_folder = "result_one_shot/50/vis_cam/SAC"
os.makedirs(output_folder, exist_ok=True)

# 定义预处理
preprocess = transforms.Compose([
    transforms.ToTensor(),
])

# 遍历处理所有图片
image_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]

for img_name in image_files:
    # 读取并resize图像
    img_path = os.path.join(input_folder, img_name)
    image_bgr = cv2.imread(img_path)
    image_bgr = cv2.resize(image_bgr, (1024, 1024))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) / 255.0  # 归一化到 0-1

    input_tensor = preprocess(image_rgb).unsqueeze(0).to(device).float()

    # 模型前向，拿到预测
    output = model(input_tensor)
    pred_mask = output.argmax(1).squeeze().cpu().numpy()

    # 只可视化前景（假设类别是1）
    target_category = 1
    mask = np.zeros(pred_mask.shape, dtype=np.uint8)
    mask[pred_mask == target_category] = 1

    # GradCAM
    targets = [SemanticSegmentationTarget(target_category, mask)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # 叠加热力图
    cam_image = show_cam_on_image(image_rgb, grayscale_cam, use_rgb=True)

    # 保存
    save_path = os.path.join(output_folder, img_name)
    cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

    print(f"Saved Grad-CAM visualization to {save_path}")

print("All Grad-CAM visualizations have been saved.")
