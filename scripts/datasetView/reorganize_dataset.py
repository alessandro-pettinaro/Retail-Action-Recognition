import os
import shutil
import random
from sklearn.model_selection import train_test_split

def reorganize_dataset(source_dir, dest_dir):
    """
    Riorganizza il dataset raggruppando i video per vista (FPV, TOP, TPV) e classe (take, release),
    senza fare la suddivisione in train, val, test.
    I file vengono copiati con nomi univoci per evitare conflitti.
    """
    vistas = ['FPV', 'TPV', 'TOP']
    actions = ['take', 'release']

    new_dataset_dir = os.path.join(dest_dir, 'new_dataset')
    os.makedirs(new_dataset_dir, exist_ok=True)

    for vista in vistas:
        for action in actions:
            os.makedirs(os.path.join(new_dataset_dir, vista, action), exist_ok=True)

    for persona_id in os.listdir(source_dir):
        persona_dir = os.path.join(source_dir, persona_id)

        if os.path.isdir(persona_dir):
            for vista in vistas:
                vista_dir = os.path.join(persona_dir, f'{vista}_synchronized_output_video')

                if os.path.isdir(vista_dir):
                    for action in actions:
                        action_dir = os.path.join(vista_dir, action)

                        if os.path.isdir(action_dir):
                            for obj in os.listdir(action_dir):
                                obj_dir = os.path.join(action_dir, obj)

                                if os.path.isdir(obj_dir):
                                    for sub_obj in os.listdir(obj_dir):
                                        sub_obj_dir = os.path.join(obj_dir, sub_obj)

                                        if os.path.isdir(sub_obj_dir):
                                            for video in os.listdir(sub_obj_dir):
                                                video_path = os.path.join(sub_obj_dir, video)

                                                if os.path.isfile(video_path) and video.lower().endswith(('.mp4', '.avi', '.mkv')):
                                                    # Genera nome univoco
                                                    unique_name = f"{vista}_{action}_{persona_id}_{obj}_{sub_obj}_{video}"
                                                    dst_path = os.path.join(new_dataset_dir, vista, action, unique_name)

                                                    # Se il file esiste già, aggiunge numero random
                                                    if os.path.exists(dst_path):
                                                        base, ext = os.path.splitext(unique_name)
                                                        unique_name = f"{base}_{random.randint(1000, 9999)}{ext}"
                                                        dst_path = os.path.join(new_dataset_dir, vista, action, unique_name)

                                                    shutil.copy(video_path, dst_path)

    print("Riorganizzazione completata!")

def split_dataset(source_dataset_dir, dest_dir, train_pct=0.7, val_pct=0.15, test_pct=0.15):
    """
    Organizza i video da 'new_dataset' in train/val/test per ciascuna vista e azione.
    I file vengono copiati con nomi univoci se necessario.
    """
    assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-5, "Le percentuali devono sommare a 1.0"

    vistas = ['FPV', 'TPV', 'TOP']
    actions = ['take', 'release']
    splits = ['train', 'val', 'test']

    dataset_view_dir = os.path.join(dest_dir, 'dataset_view')
    os.makedirs(dataset_view_dir, exist_ok=True)

    for vista in vistas:
        for action in actions:
            action_dir = os.path.join(source_dataset_dir, vista, action)

            if not os.path.isdir(action_dir):
                continue

            video_files = [f for f in os.listdir(action_dir)
                           if os.path.isfile(os.path.join(action_dir, f)) and f.lower().endswith(('.mp4', '.avi', '.mkv'))]

            random.shuffle(video_files)
            total = len(video_files)
            train_end = int(total * train_pct)
            val_end = train_end + int(total * val_pct)

            split_files = {
                'train': video_files[:train_end],
                'val': video_files[train_end:val_end],
                'test': video_files[val_end:]
            }

            for split in splits:
                split_dir = os.path.join(dataset_view_dir, vista, split, action)
                os.makedirs(split_dir, exist_ok=True)

                for file_name in split_files[split]:
                    src_path = os.path.join(action_dir, file_name)
                    dst_path = os.path.join(split_dir, file_name)

                    # Se esiste già, rinomina aggiungendo un identificatore
                    if os.path.exists(dst_path):
                        base, ext = os.path.splitext(file_name)
                        file_name = f"{base}_{random.randint(1000, 9999)}{ext}"
                        dst_path = os.path.join(split_dir, file_name)

                    shutil.copy(src_path, dst_path)

    print("Suddivisione completata in 'dataset_view'!")

# Directory di origine e destinazione
source_directory = '/home/vrai/Video ed Excel'         # Inserisci il percorso del tuo dataset di origine
destination_directory = '/home/vrai/'                  # Cartella di destinazione per la riorganizzazione
destination_dataset = '/home/vrai/video-mamba/data'        # Cartella per la suddivisione in train/val/test

# Passaggio 1: Riorganizzare il dataset
reorganize_dataset(source_directory, destination_directory)

# Passaggio 2: Suddividere il dataset in train, val, test
dateset = os.path.join(destination_directory, "new_dataset")
split_dataset(dateset, destination_dataset)
