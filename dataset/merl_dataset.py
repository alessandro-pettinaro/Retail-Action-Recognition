import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import torch.nn.functional as F
from torchvision import transforms

class MerlDataset(Dataset):
    """
    Dataset per classificazione video.
    Restituisce i video nel formato (C, T, H, W) con resize e normalizzazione.
    """

    def __init__(self, root_dir, size=(224, 224), extensions=(".avi", ".mp4"), num_frames=16):
        self.root_dir = root_dir
        self.size = size
        self.extensions = extensions
        self.num_frames = num_frames
        self.samples = []
        self.label_map = {}

        # Raccolta file video
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            self.label_map[class_name] = idx
            for file_name in sorted(os.listdir(class_dir)):
                if file_name.lower().endswith(self.extensions):
                    file_path = os.path.join(class_dir, file_name)
                    self.samples.append((file_path, idx))

        # Normalizzazione standard ImageNet
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

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
        if total_frames >= self.num_frames:
            indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
            video = video[indices]
        else:
            pad_frames = self.num_frames - total_frames
            last_frame = video[-1].unsqueeze(0).repeat(pad_frames, 1, 1, 1)
            video = torch.cat([video, last_frame], dim=0)

        # (T, H, W, C) → (T, C, H, W), normalizza pixel [0,1]
        video = video.permute(0, 3, 1, 2).float() / 255.0

        # Resize tutti i frame (batch)
        video = F.interpolate(video, size=self.size, mode='bilinear', align_corners=False)

        # Normalizza per canale
        video = self.normalize(video)  # Normalizza ogni frame: (T, C, H, W)

        # Trasforma (T, C, H, W) → (C, T, H, W)
        video = video.permute(1, 0, 2, 3)
        return video
