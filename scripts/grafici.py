import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# === CONFIGURA IL PERCORSO AL FILE TENSORBOARD ===

logdir = "./runs/TPV_Layer_4_B8_LR0.0001_ADAMW_"  # es: "./runs/Jul10_12-00-00_gpu1"
output_dir = "/home/vrai/video-mamba/plots/TPV/TPV_Layer_4_B8_LR0.0001_ADAMW_/"  # 📂 Cartella in cui salvare le immagini

# Crea la cartella se non esiste
os.makedirs(output_dir, exist_ok=True)

event_file = None
for root, dirs, files in os.walk(logdir):
    for file in files:
        if "tfevents" in file:
            event_file = os.path.join(root, file)
            break

if not event_file:
    raise FileNotFoundError("Nessun file .tfevents trovato nella cartella!")

# Carica eventi
ea = EventAccumulator(event_file)
ea.Reload()

# Tipi di metriche da cercare
base_metrics = ["Loss", "Accuracy", "Precision", "Recall", "F1"]

for base in base_metrics:
    metric_data = {}

    for phase in ["train", "val"]:
        tag = f"{base}/{phase}"
        if tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            metric_data[phase] = (steps, values)

    if not metric_data:
        print(f"⚠️  Nessun dato trovato per {base}.")
        continue

    # Plot
    plt.figure()
    for phase, (steps, values) in metric_data.items():
        plt.plot(steps, values, label=phase)

    plt.xlabel("Epoch")
    plt.ylabel(base)
    plt.title(f"{base} - train vs val")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Salva nella cartella indicata
    output_path = os.path.join(output_dir, f"{base}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Salvato: {output_path}")
