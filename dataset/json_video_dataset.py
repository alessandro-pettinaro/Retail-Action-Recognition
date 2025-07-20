import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision import transforms
import torchvision.transforms.functional as TF

class JSONVideoDataset(Dataset):
    """
    Dataset per classificazione video, caricando i path da uno split JSON (train/val/test).
    Ritorna video come tensori (C, T, H, W).
    """

    def __init__(self, json_path, root_dir, split='train', size=(224, 224), num_frames=16, extensions=(".mp4", ".avi"), augment_prob=0.0):
        self.root_dir = root_dir  # es. "dataset/"
        self.size = size
        self.num_frames = num_frames
        self.extensions = extensions
        self.augment_prob = augment_prob
        self.samples = []  # Lista (video_path, label)

        # Mapping label (cartella) → indice
        self.label_map = {"take": 0, "release": 1}

        # Caricamento video path dal file json
        with open(json_path, 'r') as f:
            data = json.load(f)

        for path in data[split]:
            label_name = os.path.normpath(path).split(os.sep)[0].lower()
            if label_name in self.label_map:
                label = self.label_map[label_name]
                self.samples.append((os.path.join(root_dir, path), label))

        # Trasformazione base
        self.base_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        video = self.load_video(path)
        return video, label

    def load_video(self, path):
        video, _, _ = read_video(path, pts_unit="sec")
        if video.size(0) == 0:
            raise RuntimeError(f"Nessun frame trovato in {path}")

        total_frames = video.size(0)

        # Campionamento uniforme
        if total_frames >= self.num_frames:
            indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
            video = video[indices]
        else:
            pad = self.num_frames - total_frames
            last = video[-1].unsqueeze(0).repeat(pad, 1, 1, 1)
            video = torch.cat([video, last], dim=0)

        # (T, H, W, C) → (T, C, H, W)
        video = video.permute(0, 3, 1, 2).float() / 255.0

        transformed_frames = []
        for frame in video:
            pil_img = TF.to_pil_image(frame)
            transformed_frames.append(self.base_transform(pil_img))

        # (T, C, H, W) → (C, T, H, W)
        video_tensor = torch.stack(transformed_frames, dim=0)
        return video_tensor.permute(1, 0, 2, 3)
