import os
import torch
from glob import glob
from segmentation_models_pytorch import create_model
from datasets import get_data
from util import train_model, set_seed

if __name__ == "__main__":
    EPOCHS = 50
    BATCH_SIZE = 4
    DEVICE = torch.device("cuda:1")
    set_seed(123)

    DATA_ROOT = "data/SAC"
    IMAGE_DIR = os.path.join(DATA_ROOT, "image")
    image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.png")))
    print(f"[DEBUG] Found {len(image_paths)} images in {IMAGE_DIR}")

    train_img_names = [os.path.basename(p) for p in image_paths[:1]]

    print(f"[INFO] Training using images: {train_img_names}")
    train_loader, val_loader = get_data(DATA_ROOT, BATCH_SIZE, train_img_names=train_img_names)

    model_name = "UnetRN18"
    model = create_model("Unet", encoder_name="resnet18", classes=2)
    model.to(DEVICE)

    RESULTS_DIR = "result_fixed_trainval"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"[INFO] Starting training for {EPOCHS} epochs on {DEVICE}...")
    train_model(model, model_name, train_loader, val_loader, DEVICE, EPOCHS, RESULTS_DIR)
    print(f"[INFO] Training complete. Results saved in {RESULTS_DIR}")
