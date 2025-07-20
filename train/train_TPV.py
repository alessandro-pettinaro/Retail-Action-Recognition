import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# ========== PATH SETUP ==========
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from dataset.view_dataset import ViewDataset
from model.video_mamba_wrapper import VideoMambaWrapper

# ========== CONFIG ==========
VIEW_NAME = "TPV"
DATA_DIR = "/home/vrai/video-mamba/data/dataset_view/TPV"
TRAIN_DIR = os.path.join(DATA_DIR, "train_AGUMENTED_50")
VAL_DIR = os.path.join(DATA_DIR, "val")


NUM_CLASSES = 2
WEIGHT_DECAY= ""
NUM_FRAMES = 16
BATCH_SIZE = 8
NUM_EPOCHS = 15
LR = 1e-4
NUM_WORKERS = 12
TRAIN_LAST_LAYERS = 4
EXPERIMENT_NAME = f"{VIEW_NAME}_Layer_{TRAIN_LAST_LAYERS}_B{BATCH_SIZE}_LR{LR}_ADAMW_{WEIGHT_DECAY}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== METRICS ==========
precision_metric = MulticlassPrecision(num_classes=NUM_CLASSES, average='macro').to(device)
recall_metric = MulticlassRecall(num_classes=NUM_CLASSES, average='macro').to(device)
f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average='macro').to(device)

# ========== DATASET E DATALOADER ==========

train_dataset = ViewDataset(TRAIN_DIR, size=(224, 224), num_frames=NUM_FRAMES)
val_dataset = ViewDataset(VAL_DIR, size=(224, 224), num_frames=NUM_FRAMES)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ========== MODELLO ==========
model = VideoMambaWrapper(
    load_pretrained=False,
    load_checkpoint=True,
    apply_finetune=True,
    head_remove=True,
    get_checkpoint_path="/home/vrai/video-mamba/model/checkpoints/best/melr_10_layer_AdamW_B8_W005_DOUT_02_DPath_05_LR_2e4/best_model.pth",
    train_last_layers=TRAIN_LAST_LAYERS,
    num_class=NUM_CLASSES
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)


writer = SummaryWriter(log_dir=os.path.join("runs", EXPERIMENT_NAME))

best_val_acc = 0.0
ckpt_path = os.path.join("checkpoints", "best", EXPERIMENT_NAME , "best_model.pth")
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

# ========== TRAINING LOOP ==========
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    preds_all, labels_all = [], []

    for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
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

    precision_train = precision_metric(preds_all, labels_all).item()
    recall_train = recall_metric(preds_all, labels_all).item()
    f1_train = f1_metric(preds_all, labels_all).item()

    print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] 🔁 Loss: {train_loss:.4f} | Acc: {acc:.2f}% | Precision: {precision_train:.2f} | Recall: {recall_train:.2f} | F1: {f1_train:.2f}")

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/train", acc, epoch)
    writer.add_scalar("Precision/train", precision_train, epoch)
    writer.add_scalar("Recall/train", recall_train, epoch)
    writer.add_scalar("F1/train", f1_train, epoch)

    # ========== VALIDAZIONE ==========
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

    print(f"Validation | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | Precision: {prec_val:.2f} | Recall: {rec_val:.2f} | F1: {f1_val:.2f}\n")

    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.add_scalar("Precision/val", prec_val, epoch)
    writer.add_scalar("Recall/val", rec_val, epoch)
    writer.add_scalar("F1/val", f1_val, epoch)


    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), ckpt_path)
        print(f"Epoch {epoch+1}: New best model saved (Acc: {val_acc:.2f}%)")

writer.close()

# ========== SALVATAGGIO MODELLO FINALE ==========
last_model_path = os.path.join("checkpoints", "last", EXPERIMENT_NAME, "last_model.pth")
os.makedirs(os.path.dirname(last_model_path), exist_ok=True)
torch.save(model.state_dict(), last_model_path)

print("🎯 Training completed.")
