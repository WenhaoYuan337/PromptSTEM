import os
import torch
from glob import glob
from segmentation_models_pytorch import create_model
from datasets import get_data
from util import train_model, set_seed

# train.py

if __name__ == "__main__":
    EPOCHS = 50
    BATCH_SIZE = 4
    DEVICE = torch.device("cuda:1")
    set_seed(123)

    DATA_ROOT = "data/SAC"
    IMAGE_DIR = os.path.join(DATA_ROOT, "image")
    image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.png")))
    print(f"[DEBUG] Found {len(image_paths)} images in {IMAGE_DIR}")

    # 手动指定前4张为训练集，其他为验证集
    train_img_names = [os.path.basename(p) for p in image_paths[:12]]

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


# if __name__ == "__main__":
#     EPOCHS = 50
#     BATCH_SIZE = 16
#     DEVICE = torch.device("cuda:1")
#     set_seed(123)
#
#     DATA_ROOT = "../data/pdc" #SAC
#     IMAGE_DIR = os.path.join(DATA_ROOT, "image")
#     image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.png")))
#     print(f"[DEBUG] Found {len(image_paths)} images in {IMAGE_DIR}")
#
#     for img_path in image_paths:
#         train_img_name = os.path.basename(img_path)
#         img_stem = os.path.splitext(train_img_name)[0]
#
#         RESULTS_DIR = os.path.join("result_pdc", img_stem)
#         os.makedirs(RESULTS_DIR, exist_ok=True)
#
#         print(f"\n[INFO] Training using {train_img_name} as training image...")
#         train_loader, val_loader = get_data(DATA_ROOT, BATCH_SIZE, train_img_name=train_img_name)
#
#         model_name = "UnetRN18"
#         model = create_model("Unet", encoder_name="resnet18", classes=2)
#
#         model.to(DEVICE)
#
#         print(f"[INFO] Starting training for {EPOCHS} epochs on {DEVICE}...")
#         train_model(model, model_name, train_loader, val_loader, DEVICE, EPOCHS, RESULTS_DIR, train_img_name)
#         print(f"[DEBUG] Loaded train: {len(train_loader)}, val: {len(val_loader)}")
#
#         print(f"[INFO] Completed training on {train_img_name}. Results saved in {RESULTS_DIR}")


# import os
#
# import torch
# from segmentation_models_pytorch import create_model
#
# from datasets import get_data
# from util import train_model, set_seed
#
# if __name__ == "__main__":
#     EPOCHS = 50
#     BATCH_SIZE = 16
#     DEVICE = torch.device("cuda:1")
#     set_seed(123)
#     RESULTS_DIR = "result_one_shot_0514"
#
#     train_loader, val_loader = get_data("../data/SMN/LR", BATCH_SIZE, train_img_name="1.png")
#
#     model_name = "UnetRN18"
#     model = create_model("Unet", encoder_name="resnet18", classes=2)
#
#     os.makedirs(RESULTS_DIR, exist_ok=True)
#     metrics_path = os.path.join(RESULTS_DIR, "metrics.png")
#
#     print(f"Training {model_name} on {DEVICE}")
#     model.to(DEVICE)
#     train_model(model, model_name, train_loader, val_loader, DEVICE, EPOCHS, RESULTS_DIR)
#     print(f"Completed training {model_name}. Results saved in {RESULTS_DIR}")
