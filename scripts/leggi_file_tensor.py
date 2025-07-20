from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

# Path completo al file specifico
event_file_path = "./runs/FPV_fold_18_Layer_4_B8_LR0.0001_CV/events.out.tfevents.1752147188.e8b21c66de39.41.17"

# Verifica che il file esista
if not os.path.exists(event_file_path):
    print(f"File non trovato: {event_file_path}")
    exit(1)

# Carica i dati dal file
event_acc = EventAccumulator(event_file_path)
event_acc.Reload()

# Mostra i tag e i valori scalari
scalar_tags = event_acc.Tags().get('scalars', [])
if not scalar_tags:
    print("Nessun valore scalare trovato nel file.")
else:
    for tag in scalar_tags:
        print(f"\n🔹 Tag: {tag}")
        for e in event_acc.Scalars(tag):
            print(f"Step: {e.step}, Value: {e.value}")
