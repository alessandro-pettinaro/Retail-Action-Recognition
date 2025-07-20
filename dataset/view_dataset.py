import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


class ViewDataset(Dataset):
    """
    Dataset per classificazione video da directory strutturata in sottocartelle per classe.
    Estrae video come tensori (C, T, H, W), con data augmentation opzionale su una percentuale dei video.
    """

    def __init__(self, root_dir, size=(224, 224), num_frames=16, extensions=(".mp4", ".avi"), augment_prob=0.3):
        self.root_dir = root_dir
        self.size = size
        self.num_frames = num_frames
        self.extensions = extensions
        self.augment_prob = augment_prob
        self.samples = []      # Lista (video_path, label)
        self.label_map = {}    # Mappa: classe → indice

        # Costruzione della lista dei video
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            self.label_map[class_name.lower()] = idx
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(self.extensions):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, idx))

        # Trasformazione base (senza augmentation)
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
        """
        Carica un video da `path` e restituisce un tensore (C, T, H, W) con trasformazioni coerenti.
        """
        video, _, _ = read_video(path, pts_unit="sec")
        if video.size(0) == 0:
            raise RuntimeError(f"Nessun frame trovato in {path}")

        total_frames = video.size(0)

        # Campionamento uniforme dei frame
        if total_frames >= self.num_frames:
            indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
            video = video[indices]
        else:
            pad = self.num_frames - total_frames
            last = video[-1].unsqueeze(0).repeat(pad, 1, 1, 1)
            video = torch.cat([video, last], dim=0)

        # (T, H, W, C) → (T, C, H, W)
        video = video.permute(0, 3, 1, 2).float() / 255.0

        transform = self.base_transform

        # Applica la stessa trasformazione a tutti i frame
        transformed_frames = []
        for frame in video:
            pil_img = TF.to_pil_image(frame)
            transformed_frames.append(transform(pil_img))

        # (T, C, H, W) → (C, T, H, W)
        video_tensor = torch.stack(transformed_frames, dim=0)
        return video_tensor.permute(1, 0, 2, 3)