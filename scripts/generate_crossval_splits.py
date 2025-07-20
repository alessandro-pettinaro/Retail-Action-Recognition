import os
import re
import json
from collections import defaultdict

# Percorso del dataset (contenente take/ e release/)
DATASET_DIR = "/home/vrai/new_dataset/TOP"
CLASSES = ["take", "release"]
OUTPUT_DIR = "data/split/TOP"

# Regex per trovare l'ID nel formato: ID_0, ID_1, ...
ID_PATTERN = re.compile(r'ID_(\d+)')


# Mappa ID -> lista dei path relativi ai video
id_to_videos = defaultdict(list)

# Step 1: raccogli i video per ID
for cls in CLASSES:
    class_dir = os.path.join(DATASET_DIR, cls)
    for file in os.listdir(class_dir):
        match = ID_PATTERN.search(file)
        if match:
            id_ = int(match.group(1))
            relative_path = os.path.join(cls, file)
            id_to_videos[id_].append(relative_path)

# Ordina tutti gli ID trovati
all_ids = sorted(id_to_videos.keys())
num_ids = len(all_ids)

# Step 2: crea gli split e salvali come JSON
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in range(num_ids):
    test_id = all_ids[i]
    val_id = all_ids[(i + 1) % num_ids]
    train_ids = [id_ for id_ in all_ids if id_ != test_id and id_ != val_id]

    split = {
        "train": [],
        "val": [],
        "test": []
    }

    for id_ in all_ids:
        if id_ == test_id:
            split["test"].extend(id_to_videos[id_])
        elif id_ == val_id:
            split["val"].extend(id_to_videos[id_])
        else:
            split["train"].extend(id_to_videos[id_])

    # Salva il file JSON dello split
    out_path = os.path.join(OUTPUT_DIR, f"fold_{i+1:02d}.json")
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)

print(f"✅ Generati {num_ids} split nella cartella '{OUTPUT_DIR}'")
