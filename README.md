# BloodCell - Robustezza Avversariale e Spiegabilità di CNN per Cellule del Sangue

## Obiettivo del progetto

Addestrare due classificatori CNN (**ResNet-18** e **MobileNetV2** fine-tuned) per la classificazione di cellule ematiche in 8 classi (BloodMNIST), valutarne la vulnerabilità ad attacchi avversariali (**FGSM** e **PGD**), applicare **Adversarial Training** per ottenere versioni "corazzate" dei modelli (ResNet-18 armata e **MobileNetV2 "KevlarNet"**) e analizzare la spiegabilità delle decisioni tramite **Grad-CAM** e **Integrated Gradients**, con un focus applicativo sulla rilevanza clinica per la diagnosi di **beta-talassemia**.

## Domande di ricerca

- Quanto degrada l'accuratezza dei modelli (ResNet-18, MobileNetV2) sotto attacchi avversariali FGSM e PGD al crescere dell'intensità della perturbazione (ε), e quale architettura si dimostra più robusta "di base"?
- L'Adversarial Training (fine-tuning su esempi avversariali) riesce a "corazzare" i modelli — ResNet armata e MobileNetV2 KevlarNet — mantenendo un buon compromesso tra accuratezza su dati puliti e robustezza alle perturbazioni?
- Il modello basa le proprie decisioni su strutture morfologiche clinicamente rilevanti (nucleo, cromatina, granulazione citoplasmatica) oppure su artefatti statistici del dataset? Come cambia questo focus, misurato con Grad-CAM e Integrated Gradients, prima e dopo l'Adversarial Training?
- Quali classi cellulari — in particolare Eritroblasto e Granulocita Immaturo, entrambe rilevanti per il monitoraggio della beta-talassemia — risultano più vulnerabili agli attacchi avversariali?

## Struttura della cartella

```
BloodCell-Adversarial-Robustness/
├── notebook/
│   ├── BloodMNIST - ResNet18.ipynb          # CNN base (ResNet-18): training, attacchi FGSM/PGD, Grad-CAM
│   ├── BloodMNIST - ResNet_armata.ipynb     # Adversarial Training su ResNet-18 + Grad-CAM/Integrated Gradients
│   ├── BloodMNIST - MobileNetV2.ipynb       # CNN base (MobileNetV2): training, attacchi FGSM/PGD, Grad-CAM
│   ├── BloodMNIST - MobileNetV2_armata.ipynb # Adversarial Training su MobileNetV2 -> "KevlarNet"
│   └── BloodMNIST - Confronto.ipynb         # Confronto finale tra le architetture lungo 5 dimensioni
├── immagini/                 # figure generate dai notebook (curve, matrici di confusione, Grad-CAM, ecc.)
├── modelli/                  # checkpoint modelli e log di grid search
│   └── gs_history_mobilenetv2.pkl   # incluso (piccolo); i checkpoint .pth non sono tracciati in git (vedi sotto)
├── presentazione_progetto_IAS_Murru_Melis.pptx
├── README.md
└── requirements.txt
```

> **Nota:** i checkpoint dei modelli in `modelli/*.pth` (ResNet-18, MobileNetV2, versioni base e armate/KevlarNet, ~10-45MB ciascuno) non sono versionati in questo repository per dimensione: si rigenerano rieseguendo i notebook di training. Anche alcuni file `.pkl` con risultati intermedi di grid search/attacchi molto pesanti sono esclusi (vedi `.gitignore`).

## Dataset

- Fonte: [MedMNIST - BloodMNIST](https://medmnist.com/)
- Immagini ridimensionate a 224×224 (upscalate dalle 28×28 originali) per l'uso con architetture pre-addestrate su ImageNet (ResNet-18, MobileNetV2)
- Normalizzazione con mean/std ImageNet (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`)
- Classi (8): Basophil, Eosinophil, Erythroblast, Immature Granulocytes, Lymphocyte, Monocyte, Neutrophil, Platelet
- Download automatico gestito dalla libreria `medmnist` all'interno dei notebook (nessun dato grezzo versionato nel repository)

## Librerie richieste

Vedi `requirements.txt`. Principali: `torch`, `torchvision`, `medmnist`, `torchcam`, `grad-cam`, `opencv-python`, `Pillow`, `scikit-learn`, `seaborn`, `matplotlib`, `numpy`, `tqdm`.

**NB:** a differenza del progetto DL (basato su TensorFlow/Keras), questo progetto è interamente sviluppato in **PyTorch/torchvision**, dato l'uso di modelli pre-addestrati su ImageNet (ResNet-18, MobileNetV2) e delle librerie di explainability (`torchcam`, `grad-cam`) native dell'ecosistema PyTorch.

## Motivazione della scelta del framework

Il progetto utilizza **PyTorch** come framework principale per:

- l'accesso diretto ai gradienti rispetto all'input, necessario per implementare "a mano" gli attacchi avversariali FGSM e PGD e per l'Adversarial Training;
- l'ampia disponibilità di architetture pre-addestrate (`torchvision.models`) e di librerie di spiegabilità (Grad-CAM, Integrated Gradients) mature nell'ecosistema PyTorch;
- il fine controllo sul training loop, utile per alternare epoche su dati puliti ed esempi avversariali durante l'Adversarial Fine-Tuning.

Le istruzioni di installazione sono riportate in `requirements.txt` (`pip install -r requirements.txt`).

## Ordine di esecuzione dei notebook

1. `notebook/BloodMNIST - ResNet18.ipynb` — download BloodMNIST, training ResNet-18 (grid search iperparametri), attacchi FGSM/PGD, curve ε, Grad-CAM.
2. `notebook/BloodMNIST - ResNet_armata.ipynb` — riparte dalla ResNet-18 base, applica Adversarial Fine-Tuning, valuta la robustezza del modello "armato" e la spiegabilità (Grad-CAM, Integrated Gradients).
3. `notebook/BloodMNIST - MobileNetV2.ipynb` — training MobileNetV2 (grid search), attacchi FGSM/PGD, curve ε, Grad-CAM, analisi PEAS e contesto clinico beta-talassemia.
4. `notebook/BloodMNIST - MobileNetV2_armata.ipynb` — Adversarial Fine-Tuning di MobileNetV2 per ottenere **KevlarNet** ("leggera come una MobileNet, resistente come un giubbotto antiproiettile"), valutazione robustezza e Grad-CAM.
5. `notebook/BloodMNIST - Confronto.ipynb` — carica tutti i modelli salvati in `modelli/` e confronta le architetture su: complessità, accuratezza clean, robustezza avversariale, classi critiche per beta-talassemia, spiegabilità Grad-CAM, tabella riepilogativa finale.

## Note tecniche

- **Attacchi avversariali:** FGSM (perturbazione one-step nella direzione del gradiente) e PGD (versione iterativa proiettata nell'ε-ball), con grid search su step size (`alpha`) e numero di step.
- **Adversarial Training:** fine-tuning dei modelli base includendo esempi avversariali generati "on the fly" nel training loop, per aumentare la robustezza senza modificare l'architettura.
- **Spiegabilità:** Grad-CAM (Selvaraju et al., 2017) e Integrated Gradients per visualizzare/quantificare su quali regioni dell'immagine (nucleo, citoplasma vs sfondo) si basano le predizioni, prima e dopo l'Adversarial Training.
- **Contesto clinico:** attenzione particolare alle classi Eritroblasto e Granulocita Immaturo (indicatori di eritropoiesi inefficace/risposta midollare nella beta-talassemia) e Piastrina (trombocitopenia da ipersplenismo).
- **Percorsi relativi:** i notebook referenziano le cartelle `modelli/` e `immagini/` con percorsi relativi (`../Modelli`, `../Immagini` in alcune celle): verificare/allineare la working directory o i nomi delle cartelle (case-sensitive su Linux/GitHub) prima di rieseguire da zero.

## Crediti

Questo progetto è stato realizzato dal gruppo composto da **Luigi Murru** e **Antonio Melis**, per il corso di Intelligenza Artificiale e Sicurezza (2025/26), tenuto dai prof. Giulia Orrù e Marco Micheletto presso il Corso di Laurea in Informatica Applicata e Data Analytics (IADA) dell'Università degli Studi di Cagliari.
