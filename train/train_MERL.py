import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from dataset.merl_dataset import MerlDataset
from model.video_mamba_wrapper import VideoMambaWrapper

# ===================== CONFIG =====================
TRAIN_DIR = "/home/vrai/video-mamba/data/dataset_merl/train"
VAL_DIR =  "/home/vrai/video-mamba/data/dataset_merl/val"
NUM_FRAMES = 16
BATCH_SIZE =  16
NUM_EPOCHS = 35 
NUM_WORKERS = 10
NUM_CLASSES = 6
LR = 2e-4
WARMUP_EPOCHS = 5

# ===================== DEVICE =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== DATA =====================
train_dataset = MerlDataset(TRAIN_DIR, size=(224, 224), num_frames=16)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)

val_dataset = MerlDataset(VAL_DIR, size=(224, 224), num_frames=16)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

# ===================== MODEL =====================
model = VideoMambaWrapper(load_pretrained=True, apply_finetune=True, train_last_layers=10, num_class=6).to(device)

# ===================== LOSS, OPTIMIZER & SCHEDULER =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ===================== METRICS =====================
precision = MulticlassPrecision(num_classes=NUM_CLASSES, average='macro').to(device)
recall = MulticlassRecall(num_classes=NUM_CLASSES, average='macro').to(device)
f1 = MulticlassF1Score(num_classes=NUM_CLASSES, average='macro').to(device)

# ===================== LOGGING =====================
writer = SummaryWriter(log_dir=os.path.join("runs/Prova_merl_10"),)
best_val_acc = 0.0
checkpoint_path = os.path.join("checkpoints", "best", "Prova_merl_10", "best_model.pth")
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

# ===================== EARLY STOPPING INIT =====================
early_stop_patience = 7
early_stop_counter = 0

# ===================== TRAINING LOOP =====================
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    preds_all = []
    labels_all = []

    train_bar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{NUM_EPOCHS}] Training", leave=False)
    for videos, labels in train_bar:
        videos = videos.to(device)
        labels = labels.to(device)

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

        train_bar.set_postfix(loss=loss.item())

    acc = 100.0 * correct / total
    train_loss = running_loss / len(train_loader)
    preds_all = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)
    precision_train = precision(preds_all, labels_all).item()
    recall_train = recall(preds_all, labels_all).item()
    f1_train = f1(preds_all, labels_all).item()

    print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] 🔁 Loss: {train_loss:.4f} | Acc: {acc:.2f}% | Precision: {precision_train:.2f} | Recall: {recall_train:.2f} | F1: {f1_train:.2f}")

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/train", acc, epoch)
    writer.add_scalar("Precision/train", precision_train, epoch)
    writer.add_scalar("Recall/train", recall_train, epoch)
    writer.add_scalar("F1/train", f1_train, epoch)

    # ===================== VALIDATION =====================
    if val_loader:
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        preds_all = []
        labels_all = []

        val_bar = tqdm(val_loader, desc=" Validation", leave=False)
        with torch.no_grad():
            for videos, labels in val_bar:
                videos = videos.to(device)
                labels = labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                preds_all.append(predicted)
                labels_all.append(labels)

                val_bar.set_postfix(loss=loss.item())

        val_acc = 100.0 * val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)
        preds_all = torch.cat(preds_all)
        labels_all = torch.cat(labels_all)
        precision_val = precision(preds_all, labels_all).item()
        recall_val = recall(preds_all, labels_all).item()
        f1_val = f1(preds_all, labels_all).item()

        print(f"✅ Validation | Loss: {val_loss_avg:.4f} | Acc: {val_acc:.2f}% | Precision: {precision_val:.2f} | Recall: {recall_val:.2f} | F1: {f1_val:.2f}\n")

        writer.add_scalar("Loss/val", val_loss_avg, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Precision/val", precision_val, epoch)
        writer.add_scalar("Recall/val", recall_val, epoch)
        writer.add_scalar("F1/val", f1_val, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Best model saved at epoch {epoch+1} with accuracy {val_acc:.2f}%")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{early_stop_patience}")

        # Logging del learning rate corrente
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            break
        writer.add_scalar("LR", current_lr, epoch)

        scheduler.step(val_loss)

    # ===================== EARLY STOPPING CHECK =====================
    if early_stop_counter >= early_stop_patience:
        print(f" Early stopping triggered at epoch {epoch+1}")
        break

writer.close()

# ===================== SAVE LAST MODEL =====================
last_model_path = os.path.join("checkpoints", "last", "Prova_merl_10", "last_model.pth")
os.makedirs(os.path.dirname(last_model_path), exist_ok=True)
torch.save(model.state_dict(), last_model_path)


print("Training completed.")
