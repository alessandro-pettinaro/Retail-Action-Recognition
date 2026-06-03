# RetailMamba

**Riconoscimento di azioni in ambito retail tramite reti neurali, basato sul modello VideoMamba.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Action%20Recognition-blueviolet?style=flat-square)

Progetto di deep learning per il **riconoscimento di azioni davanti a uno scaffale** (ispezione dei prodotti, presa e rilascio di articoli), realizzato addestrando e valutando il modello **VideoMamba** su due dataset distinti.

---

## Indice

- [Panoramica](#panoramica)
- [Obiettivo](#obiettivo)
- [Il modello: VideoMamba](#il-modello-videomamba)
- [Dataset](#dataset)
- [Pipeline di sviluppo](#pipeline-di-sviluppo)
- [Viste di classificazione](#viste-di-classificazione)
- [Metriche di valutazione](#metriche-di-valutazione)
- [Pesi dei modelli](#pesi-dei-modelli)
- [Requisiti](#requisiti)
- [Installazione e avvio](#installazione-e-avvio)
- [Risultati](#risultati)
- [Documentazione](#documentazione)
- [Autore](#autore)

---

## Panoramica

L'elaborazione automatica dei video è una delle sfide più attuali del deep learning: la grande disponibilità di contenuti multimediali e la necessità di estrarre informazioni in modo efficiente spingono verso modelli capaci di gestire sequenze video lunghe e di durata variabile.

In questo contesto si colloca **VideoMamba**, un modello basato su **State Space Models (SSM)** pensato per superare i limiti delle architetture convenzionali basate su *attention*, grazie a una pipeline leggera e scalabile, particolarmente efficace sui video di lunga durata.

In questo progetto è stato utilizzato **VisionMamba** nella configurazione **Middle**, la componente di VideoMamba adattata al riconoscimento visivo in sequenze video.

---

## Obiettivo

Addestrare e valutare VideoMamba per il riconoscimento di azioni in ambito **retail** — come l'ispezione dei prodotti o l'interazione con gli scaffali (presa e rilascio di articoli) — utilizzando due dataset: **MERL Shopping Dataset** ed **ExtraEye**.

---

## Il modello: VideoMamba

| Caratteristica | Descrizione |
| --- | --- |
| Architettura | State Space Models (SSM), alternativa all'attention |
| Configurazione | VisionMamba **Middle** |
| Punti di forza | Pipeline leggera e scalabile, efficace su video lunghi |
| Pesi di partenza | Pre-addestrati su **Kinetics-400** |

---

## Dataset

- **MERL Shopping Dataset** — utilizzato nella prima fase per l'adattamento al dominio retail.
- **ExtraEye** — utilizzato nella seconda fase per la classificazione binaria delle azioni *take* e *release*.

---

## Pipeline di sviluppo

Lo sviluppo si è articolato in due fasi di *fine-tuning*.

### Fase 1 — Adattamento al dominio (MERL)

Fine-tuning di VideoMamba a partire dai pesi pre-addestrati su **Kinetics-400**, per adattare il modello al dominio del dataset **MERL**. Al termine è stato selezionato il modello migliore, **MERL-2**, usato come base per la fase successiva.

### Fase 2 — Generalizzazione (ExtraEye)

Ulteriore fine-tuning sul dataset **ExtraEye**, riutilizzando i pesi di MERL-2, per valutare la capacità di generalizzazione del modello su un nuovo compito: la **classificazione binaria** delle azioni *take* e *release*.

In entrambe le fasi sono state adottate diverse strategie di fine-tuning e tecniche di **cross-validation**.

---

## Viste di classificazione

L'analisi è stata condotta sulle tre principali viste di classificazione video:

| Vista | Significato |
| --- | --- |
| **FPV** | First-Person View |
| **TPV** | Third-Person View |
| **TOP** | Top-View |

È stato inoltre condotto un confronto tra alcune classi dei due dataset, per valutare coerenza, robustezza ed eventuali criticità del modello in contesti diversi.

---

## Metriche di valutazione

Le performance del modello sono state analizzate tramite:

- **Loss**
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

---

## Pesi dei modelli

I pesi dei modelli addestrati (MERL-2 e i modelli su ExtraEye) sono disponibili al seguente link:

**[Scarica i pesi](https://univpm-my.sharepoint.com/:f:/g/personal/s1126383_studenti_univpm_it/Ekj0_gLF7zhNhNSdOz1v22gBLLpkfa7lDpoQfvjZgYpU_w?e=8FMgkj)**

Dopo il download, posiziona i file nella cartella `checkpoints/` (o nel percorso indicato nei file di configurazione).

> I file dei pesi non sono inclusi nella repository per via delle dimensioni; sono ospitati esternamente.

---

## Requisiti

- [Python 3.x](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- GPU con supporto CUDA (consigliata per l'addestramento)

> Elenca le dipendenze esatte in un file `requirements.txt`.

---

## Installazione e avvio

```bash
# Clona la repository
git clone https://github.com/alessandro-pettinaro/RetailMamba.git
cd RetailMamba

# (consigliato) crea un ambiente virtuale
python -m venv venv
source venv/bin/activate   # su Windows: venv\Scripts\activate

# Installa le dipendenze
pip install -r requirements.txt
```

> Aggiungi qui i comandi specifici per avviare l'addestramento e la valutazione
> (es. `python train.py --config configs/merl.yaml`), adattandoli alla struttura del tuo progetto.

---

## Risultati

> Compila la tabella con le metriche finali ottenute per ciascuna vista.
> Aggiungi eventuali grafici di Loss/Accuracy e le matrici di confusione: sono la parte
> che fa più colpo in un progetto di questo tipo (caricali nella cartella `images/`).

| Vista | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- |
| FPV | – | – | – | – |
| TPV | – | – | – | – |
| TOP | – | – | – | – |

---

## Documentazione

Per l'analisi completa del progetto — scelte progettuali, esperimenti e risultati dettagliati — è disponibile la relazione:

📄 **[Leggi la relazione completa](docs/relazione.pdf)**

---

## Autore

**Alessandro Pettinaro**
[GitHub](https://github.com/alessandro-pettinaro)
