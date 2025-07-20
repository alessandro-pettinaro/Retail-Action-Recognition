import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from dataset.merl_dataset import MerlDataset
from model.video_mamba_wrapper import VideoMambaWrapper

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
# ============ CONFIG ============

TEST_DIR = "/home/vrai/video-mamba/data/dataset_merl/test"
BATCH_SIZE = 8
NUM_WORKERS = 12
NUM_CLASSES = 6
CLASS_NAMES = ["Background","Hand In Shelf","Inspect Product","Inspect Shelf","Reach To Shelf","Retract From Shelf"]
CHECKPOINT_PATH = "/home/vrai/video-mamba/model/checkpoints/best/melr_10_layer_AdamW_B8_W005_DOUT_02_DPath_05_LR_2e4/best_model.pth"

# ============ DEVICE ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ DATA ============
test_dataset = MerlDataset(TEST_DIR, size=(224, 224), num_frames=16)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

# ============ MODEL ============
model = VideoMambaWrapper(load_checkpoint=True,get_checkpoint_path=CHECKPOINT_PATH,num_class=NUM_CLASSES).to(device) 

model.eval()

# ============ INFERENCE ============
all_preds = []
all_labels = []

with torch.no_grad():
    for videos, labels in tqdm(test_loader, desc="Inference"):
        videos = videos.to(device)
        labels = labels.to(device)
        outputs = model(videos)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

# ============ REPORT ============
    report_dict = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    print(report_dict)
    report_df = pd.DataFrame(report_dict).transpose()

    plt.figure(figsize=(10, len(report_df) * 0.6))  # auto-adatta altezza
    sns.heatmap(report_df.iloc[:, :-1], annot=True, fmt=".2f", cmap="Blues", cbar=False)
    plt.title(f"Classification Report")
    plt.ylabel("Class")
    plt.xlabel("Metric")
    plt.tight_layout()

    os.makedirs("report/Merl/2Migliore/Classification_matrices/", exist_ok=True)
    plt.savefig(f"report/Merl/2Migliore/Classification_matrices/class_report.png")
    plt.close()


# ============ CONFUSION MATRIX ============
cm = confusion_matrix(all_labels, all_preds)

print("Confusion Matrix:")
print(cm)

# ============ VISUALIZE CONFUSION MATRIX ============
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

os.makedirs("report/Merl/2Migliore/Confusion_matrices/", exist_ok=True)
plt.savefig(f"report/Merl/2Migliore/Confusion_matrices/conf_matrix.png")
plt.close()