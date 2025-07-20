import os
import random
import shutil
import torch
import torchvision.transforms.v2 as T
from torchvision.io import read_video, write_video



DATASET_DIR = '/home/vrai/video-mamba/data/dataset_view/TPV'
INPUT_DIR = os.path.join(DATASET_DIR, 'train')
OUTPUT_DIR = os.path.join(DATASET_DIR, 'train_AGUMENTED_50')
AUG_PROBABILITY = 0.5
NUM_AUG_PER_VIDEO = 1

transform = T.Compose([
    T.RandomApply([
        T.ColorJitter(
            brightness=0.1,  # variazione leggera di luminosità
            contrast=0.5,   # variazione minima di contrasto
            saturation=0.5, # variazione minima di saturazione
            hue=0.2         # variazione minima di tonalità
        )
    ], p=0.5),
])

def augment_and_save_video(input_path, output_path):
    # Legge video in tensor: (T, H, W, C)
    video, _, info = read_video(input_path, pts_unit='sec')
    video = video.float() / 255.0  # Normalizza tra 0 e 1
    video = video.permute(0, 3, 1, 2)  # (T, C, H, W)

    # Applica trasformazione frame per frame
    video_aug = torch.stack([transform(frame) for frame in video])

    # Inverte permutazione e riscalo [0,255]
    video_aug = (video_aug * 255.0).byte().permute(0, 2, 3, 1)  # (T, H, W, C)

    # Salva video
    write_video(output_path, video_aug, fps=int(info['video_fps']))

# Generazione dataset aumentato
def create_augmented_dataset():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    for class_name in os.listdir(INPUT_DIR):
        class_in = os.path.join(INPUT_DIR, class_name)
        if not os.path.isdir(class_in):
            continue

        class_out = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_out, exist_ok=True)

        for filename in os.listdir(class_in):
            in_path = os.path.join(class_in, filename)
            name, ext = os.path.splitext(filename)
            out_orig = os.path.join(class_out, f"{name}{ext}")
            shutil.copy2(in_path, out_orig)

            for i in range(NUM_AUG_PER_VIDEO):
                if random.random() < AUG_PROBABILITY:
                    out_aug = os.path.join(class_out, f"{name}_aug{i+1}{ext}")
                    try:
                        augment_and_save_video(in_path, out_aug)
                    except Exception as e:
                        print(f"[ERRORE su {in_path}]: {e}")

if __name__ == '__main__':
    create_augmented_dataset()
    print("Dataset aumentato salvato in:", OUTPUT_DIR)
