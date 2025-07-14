import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from tqdm import tqdm

image_dir = "..."
output_dir = "..."
model_path = "..."

os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

image_paths = [f for f in os.listdir(image_dir) if f.endswith(".png")]

for filename in tqdm(image_paths, desc="Predicting"):
    img_path = os.path.join(image_dir, filename)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Warning: Failed to read {filename}, skipping.")
        continue

    image_resized = cv2.resize(image, (512, 512))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    input_tensor = transform(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy().astype(np.uint8)

    binary_mask = (pred_mask == 1).astype(np.uint8) * 255

    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, binary_mask)

print("Batch inference completed.")
