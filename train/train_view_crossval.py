import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from tqdm import tqdm

# === IMPORT PERSONALIZZATI ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)


from dataset.json_video_dataset import JSONVideoDataset
from model.video_mamba_wrapper import VideoMambaWrapper

# === CONFIG ===
DATASET_DIR = "/home/vrai/new_dataset/TOP"
SPLIT_DIR = "data/split/TOP"
NUM_CLASSES = 2
NUM_FRAMES = 16
BATCH_SIZE = 8
NUM_EPOCHS = 15
LR = 1e-4
NUM_WORKERS = 12
TRAIN_LAST_LAYERS = 4
CHECKPOINT_BASE = "model/checkpoints"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === METRICHE ===
precision_metric = MulticlassPrecision(num_classes=NUM_CLASSES, average='macro').to(device)
recall_metric = MulticlassRecall(num_classes=NUM_CLASSES, average='macro').to(device)
f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average='macro').to(device)

# === LOOP SU TUTTI I FOLD ===
for fold_file in sorted(os.listdir(SPLIT_DIR)):
    if not fold_file.endswith(".json"):
        continue

    fold_name = fold_file.replace(".json", "")
    EXPERIMENT_NAME = f"TOP_{fold_name}_Layer_{TRAIN_LAST_LAYERS}_B{BATCH_SIZE}_LR{LR}_CV"
    ckpt_dir_best = os.path.join(CHECKPOINT_BASE, "best", EXPERIMENT_NAME)
    ckpt_dir_last = os.path.join(CHECKPOINT_BASE, "last", EXPERIMENT_NAME)
    os.makedirs(ckpt_dir_best, exist_ok=True)
    os.makedirs(ckpt_dir_last, exist_ok=True)
    ckpt_path_best = os.path.join(ckpt_dir_best, "best_model.pth")
    ckpt_path_last = os.path.join(ckpt_dir_last, "last_model.pth")
    
    print(f"Starting fold: {fold_name}")

    # === DATASET ===
    json_path = os.path.join(SPLIT_DIR, fold_file)
    train_dataset = JSONVideoDataset(json_path, DATASET_DIR, split="train", num_frames=NUM_FRAMES)
    val_dataset = JSONVideoDataset(json_path, DATASET_DIR, split="val", num_frames=NUM_FRAMES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # === MODELLO ===
    model = VideoMambaWrapper(
        load_pretrained=False,
        load_checkpoint=True,
        apply_finetune=True,
        get_checkpoint_path="/home/vrai/video-mamba/model/checkpoints/best/melr_10_layer_AdamW_B8_W005_DOUT_02_DPath_05_LR_2e4/best_model.pth",
        train_last_layers=TRAIN_LAST_LAYERS,
        num_class=NUM_CLASSES
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=os.path.join("runs", EXPERIMENT_NAME))

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        preds_all, labels_all = [], []

        for videos, labels in tqdm(train_loader, desc=f"[{fold_name}] Epoch {epoch+1}/{NUM_EPOCHS}"):
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            preds_all.append(predicted)
            labels_all.append(labels)

        acc = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)
        preds_all = torch.cat(preds_all)
        labels_all = torch.cat(labels_all)

        prec_train = precision_metric(preds_all, labels_all).item()
        rec_train = recall_metric(preds_all, labels_all).item()
        f1_train = f1_metric(preds_all, labels_all).item()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", acc, epoch)
        writer.add_scalar("Precision/train", prec_train, epoch)
        writer.add_scalar("Recall/train", rec_train, epoch)
        writer.add_scalar("F1/train", f1_train, epoch)

        # === VALIDAZIONE ===
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        preds_all, labels_all = [], []

        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                preds_all.append(predicted)
                labels_all.append(labels)

        val_acc = 100.0 * val_correct / val_total
        val_loss /= len(val_loader)
        preds_all = torch.cat(preds_all)
        labels_all = torch.cat(labels_all)

        prec_val = precision_metric(preds_all, labels_all).item()
        rec_val = recall_metric(preds_all, labels_all).item()
        f1_val = f1_metric(preds_all, labels_all).item()

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Precision/val", prec_val, epoch)
        writer.add_scalar("Recall/val", rec_val, epoch)
        writer.add_scalar("F1/val", f1_val, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path_best)
            print(f"{fold_name} | Epoch {epoch+1}: new best model saved ({val_acc:.2f}%)")

    # === SALVA MODELLO FINALE ===
    torch.save(model.state_dict(), ckpt_path_last)
    writer.close()
    print(f"Completed fold {fold_name} | Best val acc: {best_val_acc:.2f}%\n")

