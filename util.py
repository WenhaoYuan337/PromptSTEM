import os
import random
import time

import numpy as np
import torch
import tqdm

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from segmentation_models_pytorch import losses
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
import torchvision.transforms as T
import cv2

# def train_model(model, model_name, train_loader, test_loader, device, epochs, results_dir, train_img_name):
def train_model(model, model_name, train_loader, test_loader, device, epochs, results_dir):
    start_time = time.time()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-4, weight_decay=5e-4)
    loss_fl = losses.FocalLoss(mode='multiclass', alpha=0.25)
    loss_jd = losses.JaccardLoss(mode='multiclass')

    # model_results_dir = os.path.join(results_dir, model_name)
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "log.csv")
    best_model_path = os.path.join(results_dir, "best_model.pt")
    plot_path = os.path.join(results_dir, "metrics.png")

    with open(log_path, "w") as f:
        f.write("epoch,phase,loss,mPA,mDice,mIoU\n")

    metrics = {
        "train_loss": [], "test_loss": [],
        "train_mPA": [], "test_mPA": [],
        "train_mDice": [], "test_mDice": [],
        "train_mIoU": [], "test_mIoU": [],
        "train_recall": [], "test_recall": [],
        "train_precision": [], "test_precision": []
    }

    best_mIoU = 0
    best_mDice = 0
    best_mPA = 0
    best_recall = 0
    best_precision = 0

    for epoch in range(epochs):

        model.train()
        train_loss, train_cm = 0, np.zeros((2, 2))
        for x, y in tqdm.tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{epochs}"):
            x, y = x.to(device), y.to(device).long()
            pred = model(x.float())
            loss = loss_fl(pred, y) + loss_jd(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_mPA, train_mDice, train_mIoU, train_recall, train_precision = metrice(y.cpu().numpy(),
                                                                                    pred.argmax(dim=1).cpu().numpy())
        with open(log_path, "a") as f:
            f.write(
                f"{epoch + 1},train,{train_loss / len(train_loader):.4f},{train_mPA:.4f},{train_mDice:.4f},{train_mIoU:.4f},{train_recall:.4f},{train_precision:.4f}\n")

        tqdm.tqdm.write(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, mPA: {train_mPA:.4f}, mDice: {train_mDice:.4f}, mIoU: {train_mIoU:.4f}")

        metrics["train_loss"].append(train_loss / len(train_loader))
        metrics["train_mPA"].append(train_mPA)
        metrics["train_mDice"].append(train_mDice)
        metrics["train_mIoU"].append(train_mIoU)
        metrics["train_recall"].append(train_recall)
        metrics["train_precision"].append(train_precision)

        test_loss = 0
        with torch.no_grad():
            for x, y in tqdm.tqdm(test_loader, desc=f"Test Epoch {epoch + 1}/{epochs}"):
                x, y = x.to(device), y.to(device).long()
                pred = model(x.float())
                test_loss += (loss_fl(pred, y) + loss_jd(pred, y)).item()
        test_mPA, test_mDice, test_mIoU, test_recall, test_precision = metrice(y.cpu().numpy(),
                                                                               pred.argmax(dim=1).cpu().numpy())

        with open(log_path, "a") as f:
            f.write(
                f"{epoch + 1},test,{test_loss / len(test_loader):.4f},{test_mPA:.4f},{test_mDice:.4f},{test_mIoU:.4f},{test_recall:.4f},{test_precision:.4f}\n")

        tqdm.tqdm.write(
            f"Epoch {epoch + 1}/{epochs}, Test Loss: {test_loss:.4f}, mPA: {test_mPA:.4f}, mDice: {test_mDice:.4f}, mIoU: {test_mIoU:.4f}")

        metrics["test_loss"].append(test_loss / len(test_loader))
        metrics["test_mPA"].append(test_mPA)
        metrics["test_mDice"].append(test_mDice)
        metrics["test_mIoU"].append(test_mIoU)
        metrics["test_recall"].append(test_recall)
        metrics["test_precision"].append(test_precision)

        del x, y, pred
        torch.cuda.empty_cache()

        vis_dir = os.path.join(results_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)

        val_vis_img = "17.png"
        vis_targets = [(val_vis_img, "test"), (val_vis_img, "train")]

        for vis_img_name, vis_tag in vis_targets:
            vis_img_path = os.path.join(os.path.dirname(test_loader.dataset.image_paths[0]), vis_img_name)
            if not os.path.exists(vis_img_path):
                print(f"Warning: {vis_img_name} not found, skipping visualization.")
                continue

            raw_img = cv2.imread(vis_img_path)
            raw_img = cv2.resize(raw_img, (1024, 1024))
            image_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB) / 255.0
            preprocess = T.Compose([T.ToTensor()])
            input_tensor = preprocess(image_rgb).unsqueeze(0).to(device).float()

            model.eval()
            try:
                target_layer = model.decoder.blocks[-1].conv1
            except Exception:
                raise ValueError("error: model.decoder.blocks[-1].conv1")

            cam = GradCAM(model=model, target_layers=[target_layer])
            with torch.no_grad():
                output = model(input_tensor)
                pred_mask = output.argmax(1).squeeze().cpu().numpy()

            target_category = 1
            binary_mask = np.zeros_like(pred_mask, dtype=np.uint8)
            binary_mask[pred_mask == target_category] = 1
            targets = [SemanticSegmentationTarget(target_category, binary_mask)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
            cam_image = show_cam_on_image(image_rgb, grayscale_cam, use_rgb=True)

            cam_save_path = os.path.join(vis_dir, f"epoch_{epoch + 1}_{vis_tag}_cam.png")
            cv2.imwrite(cam_save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

            overlay = image_rgb.copy()
            color_mask = np.zeros_like(overlay)
            color_mask[pred_mask == target_category] = [1.0, 0.0, 0.0]
            overlay = cv2.addWeighted(overlay, 0.6, color_mask, 0.4, 0)
            overlay = (overlay * 255).astype(np.uint8)

            overlay_save_path = os.path.join(vis_dir, f"epoch_{epoch + 1}_{vis_tag}_overlay.png")
            cv2.imwrite(overlay_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            tqdm.tqdm.write(f"Saved CAM and overlay of {vis_img_name} to {cam_save_path} and {overlay_save_path}")

            del input_tensor, raw_img, output, grayscale_cam, cam_image, cam, targets
            torch.cuda.empty_cache()

        if test_mIoU > best_mIoU:
            best_mIoU = test_mIoU
            # torch.save(model, best_model_path)

        if test_mPA > best_mPA:
            best_mPA = test_mPA

        if test_mDice > best_mDice:
            best_mDice = test_mDice

        if test_recall > best_recall:
            best_recall = test_recall

        if test_precision > best_precision:
            best_precision = test_precision

        if (epoch + 1) % 5 == 0:
            epoch_model_path = os.path.join(results_dir, f"model_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), epoch_model_path)

    plot_metrics(metrics, plot_path)

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,test_loss,train_mPA,test_mPA,train_mDice,test_mDice,train_mIoU,test_mIoU,train_recall,test_recall,train_precision,test_precision\n")
        for i in range(epochs):
            f.write(f"{i + 1},{metrics['train_loss'][i]},{metrics['test_loss'][i]},"
                    f"{metrics['train_mPA'][i]},{metrics['test_mPA'][i]},"
                    f"{metrics['train_mDice'][i]},{metrics['test_mDice'][i]},"
                    f"{metrics['train_mIoU'][i]},{metrics['test_mIoU'][i]},"
                    f"{metrics['train_recall'][i]},{metrics['test_recall'][i]},"
                    f"{metrics['train_precision'][i]},{metrics['test_precision'][i]}\n")
        f.write(f"Best mPA: {best_mPA}, Best mDice: {best_mDice}, Best mIoU: {best_mIoU}, Best recall: {best_recall}, Best precision: {best_precision}\n")

    end_time = time.time()
    total_time = end_time - start_time

    with open(log_path, "a") as f:
        f.write(f"\nTotal Training Time (seconds): {total_time:.2f}\n")


def plot_metrics(metrics, plot_path):
    required_keys = ["train_loss", "test_loss", "train_mPA", "test_mPA",
                     "train_mDice", "test_mDice", "train_mIoU", "test_mIoU",
                     "train_recall", "test_recall", "train_precision", "test_precision"]
    for key in required_keys:
        if key not in metrics:
            raise ValueError(f"Missing key in metrics dictionary: {key}")

    num_epochs = len(metrics["train_loss"])
    if num_epochs == 0:
        raise ValueError("Metrics data is empty. Cannot generate plot.")

    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 15))

    plt.subplot(3, 2, 1)
    plt.plot(epochs, metrics["train_loss"], "b", label="Train Loss", linewidth=2)
    plt.plot(epochs, metrics["test_loss"], "r", label="Test Loss", linewidth=2)
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epoch")

    plt.subplot(3, 2, 2)
    plt.plot(epochs, metrics["train_mPA"], "b", label="Train mPA", linewidth=2)
    plt.plot(epochs, metrics["test_mPA"], "r", label="Test mPA", linewidth=2)
    plt.legend()
    plt.title("Mean Pixel Accuracy (mPA)")
    plt.xlabel("Epoch")

    plt.subplot(3, 2, 3)
    plt.plot(epochs, metrics["train_mDice"], "b", label="Train mDice", linewidth=2)
    plt.plot(epochs, metrics["test_mDice"], "r", label="Test mDice", linewidth=2)
    plt.legend()
    plt.title("Mean Dice Coefficient (mDice)")
    plt.xlabel("Epoch")

    plt.subplot(3, 2, 4)
    plt.plot(epochs, metrics["train_mIoU"], "b", label="Train mIoU", linewidth=2)
    plt.plot(epochs, metrics["test_mIoU"], "r", label="Test mIoU", linewidth=2)
    plt.legend()
    plt.title("Mean Intersection over Union (mIoU)")
    plt.xlabel("Epoch")

    plt.subplot(3, 2, 5)
    plt.plot(epochs, metrics["train_recall"], "b", label="Train Recall", linewidth=2)
    plt.plot(epochs, metrics["test_recall"], "r", label="Test Recall", linewidth=2)
    plt.legend()
    plt.title("Recall")
    plt.xlabel("Epoch")

    plt.subplot(3, 2, 6)
    plt.plot(epochs, metrics["train_precision"], "b", label="Train Precision", linewidth=2)
    plt.plot(epochs, metrics["test_precision"], "r", label="Test Precision", linewidth=2)
    plt.legend()
    plt.title("Precision")
    plt.xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Metrics plot saved to {plot_path}")


def metrice(y_true, y_pred):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=[0, 1])

    mpa = np.nanmean(np.diag(cm) / np.maximum(cm.sum(axis=1), 1))

    iou = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm) + 1e-7)
    miou = np.nanmean(iou)

    dice = 2 * np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) + 1e-7)
    mdice = np.nanmean(dice)

    tp = cm[1, 1]  # True Positive for class 1
    fn = cm[1, 0]  # False Negative for class 1
    fp = cm[0, 1]  # False Positive for class 1

    recall = tp / (tp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)

    return mpa, mdice, miou, recall, precision


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
