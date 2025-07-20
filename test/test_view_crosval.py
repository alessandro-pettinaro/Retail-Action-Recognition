import os
import sys
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)


from dataset.json_video_dataset import JSONVideoDataset
from model.video_mamba_wrapper import VideoMambaWrapper

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# === CONFIG ===
SPLIT_DIR = "data/split/TOP"  # contiene fold_01.json ... fold_17.json
DATASET_ROOT = "/home/vrai/new_dataset/TOP"  # contiene take/, release/
CHECKPOINT_DIR = "/home/vrai/video-mamba/model/checkpoints/best"
VIEW_NAME = "TOP"

BATCH_SIZE = 6
NUM_WORKERS = 12
NUM_CLASSES = 2
CLASS_NAMES = ["release","take"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === LOOP SU TUTTI I FOLD JSON ===
for json_file in sorted(os.listdir(SPLIT_DIR)):
    if not json_file.endswith(".json"):
        continue

    fold_name = json_file.replace(".json", "")
    fold_index = fold_name.split("_")[-1]
    directory = f"{VIEW_NAME}_{fold_name}_Layer_4_B8_LR0.0001_CV"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, directory, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint non trovato per fold {fold_name}: {checkpoint_path}")
        continue

    print(f"\n======================  Testing {fold_name} ======================")

    # === MODELLO ===
    model = VideoMambaWrapper(
        load_checkpoint=True,
        get_checkpoint_path=checkpoint_path,
        num_class=NUM_CLASSES
    ).to(device)
    model.eval()

    # === DATASET TEST DAL JSON ===
    json_path = os.path.join(SPLIT_DIR, json_file)
    test_dataset = JSONVideoDataset(json_path, root_dir=DATASET_ROOT, split="test", num_frames=16)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # === INFERENZA ===
    all_preds, all_labels = [], []

    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc=f" Inference {fold_name}"):
            videos = videos.to(device)
            labels = labels.to(device)
            outputs = model(videos)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # === REPORT ===
    print("\n Classification Report:")
    report_text = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
    print(report_text)

    # === CONFUSION MATRIX ===
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # === SALVA CONFUSION MATRIX COME IMMAGINE ===
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {fold_name}")
    plt.tight_layout()

    os.makedirs("report/TOP/Cross_Validation/Confusion_matrices/TOP_CrossValidation_best", exist_ok=True)
    plt.savefig(f"report/TOP/Cross_Validation/Confusion_matrices/TOP_CrossValidation_best/conf_matrix_{fold_name}.png")
    plt.close()

    # === SALVA CLASSIFICATION REPORT COME IMMAGINE ===
    report_dict = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    plt.figure(figsize=(10, len(report_df) * 0.6))  # auto-adatta altezza
    sns.heatmap(report_df.iloc[:, :-1], annot=True, fmt=".2f", cmap="Blues", cbar=False)
    plt.title(f"Classification Report - {fold_name}")
    plt.ylabel("Class")
    plt.xlabel("Metric")
    plt.tight_layout()

    os.makedirs("report/TOP/Cross_Validation/Classification_matrices/TOP_CrossValidation_best", exist_ok=True)
    plt.savefig(f"report/TOP/Cross_Validation/Classification_matrices/TOP_CrossValidation_best/class_report_{fold_name}.png")
    plt.close()
