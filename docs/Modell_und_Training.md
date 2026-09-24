# Modell und Training

## Warum U-Net und nicht YOLO

Die erste Wahl war YOLO — naheliegend, weil es für Objekterkennung der
Standard ist. Drei Gründe sprachen dagegen, und alle drei sind für dieses
Produkt entscheidend:

**Lizenz.** Ultralytics YOLO steht unter AGPL-3.0. Bei einem Netzwerkdienst
greift die Netzklausel: Wer den Dienst benutzt, hat Anspruch auf den
Quelltext — auch der On-Prem-Kunde. Für ein Produkt, das bei Kunden auf deren
Blech läuft, ist das entweder eine Offenlegung oder eine Enterprise-Lizenz.
`segmentation_models_pytorch` steht unter MIT, die Encoder-Gewichte kommen
aus torchvision (BSD). Nichts davon zwingt zu irgendetwas.

**Form des Gegenstands.** Ein Riss ist dünn, lang, verzweigt und läuft quer
durchs ganze Bild. Ein Rechteck um ihn herum sagt fast nichts — es enthält zu
90 Prozent Beton. Die Zerlegung in Instanzen, die YOLO leistet, liefert bei
Rissen ohnehin die Zusammenhangskomponente, und die bekommt man aus einer
Maske geschenkt.

**Auflösung der Maske.** Die Segmentierungsköpfe der YOLO-Familie arbeiten
intern auf grobem Raster (typisch 160 × 160, danach hochskaliert). Für eine
Breite, die zwischen 0,2 und 0,3 mm entscheidet, ist das zu grob. Ein U-Net
segmentiert in voller Auflösung.

**Für Schrauben wird das anders sein.** Schrauben sind kompakte, zählbare
Objekte in großer Zahl — genau der Fall, für den ein Detektor gebaut ist.
Wenn `bolt` drankommt, gehört dort ein Detektor hin (RT-DETR oder YOLOX,
beide Apache-2.0), und die gemeinsame Nachbearbeitung greift nur für
`corrosion` wieder. Die Architektur ist darauf vorbereitet: `models/` kennt
ein Protokoll, keine feste Klasse.

## Der Datensatz

**CrackSeg9k** — 9.255 Bilder zu 400 × 400 mit binären Masken, konsolidiert
aus zehn Einzeldatensätzen: Crack500, DeepCrack, CrackTree, GAPs, Volker
(Rissbilder), SDNET, Masonry, Ceramic — und *Noncrack*, die rissfreien
Gegenbeispiele, ohne die ein Netz jede Fuge, jedes Kabel und jede
Schalungskante für einen Riss hält.

> Kulkarni et al., *CrackSeg9k: A Collection and Benchmark for Crack
> Segmentation Datasets and Frameworks*, arXiv:2208.13054
> Harvard Dataverse, doi:10.7910/DVN/EGIEBY
> **Lizenz: CC0 1.0 Universal — Public Domain**

Die Lizenz war das Auswahlkriterium. Viele Rissdatensätze stehen unter „nur
für Forschung" — in einem Produkt, das verkauft wird, ist das eine Zeitbombe.
CC0 heißt: kein Copyleft, keine Namensnennungspflicht, kein Risiko.

Holen und entpacken:

```
docker run --rm -v "$(pwd -W):/work" -w /work sada-vision:train \
  python training/datasets/crackseg9k.py --out data/crackseg9k
```

Das lädt rund 4 GB Parquet (mit Fortsetzung — eine abgerissene Leitung ist
kein Grund, von vorn anzufangen) und schreibt PNG-Paare nach
`data/crackseg9k/{train,test}/{images,masks}`. Die Masken werden dabei hart
auf 0/255 geschnitten: JPEG-Artefakte hinterlassen Zwischenwerte an den
Rändern, und ein Riss ist binär.

## Trainieren

**CPU** (Rauchprobe und brauchbarer Anfang, kein Auslieferungsmodell):

```
docker run --rm -v "$(pwd -W):/work" -w /work sada-vision:train \
  python training/train.py --config training/configs/crack_unet_r18.yaml
```

**GPU** (das Modell, das in Betrieb gehört) — auf Colab, RunPod oder einer
eigenen Karte ab 8 GB. In `deploy/Dockerfile.train` die torch-Zeile auf die
CUDA-Fassung umstellen:

```
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
      "torch==2.8.0" "torchvision==0.23.0"
```

dann `--config training/configs/crack_unet_r34_gpu.yaml`. Rechnet auf einer
T4 rund zwei Stunden.

Ein unterbrochener Lauf wird fortgesetzt: `--resume runs/<name>/last.pt`.
`SIGTERM` beendet die laufende Epoche und sichert — ein Lauf über Stunden
wird irgendwann abgebrochen, und ohne das wäre die Arbeit weg.

### Was eingestellt ist und warum

| Größe | Wert | Grund |
|---|---|---|
| Ausschnitt statt Verkleinerung | 256 bzw. 384 | Ein auf 256 **verkleinertes** Foto verliert den Riss. Ein zufälliger **Ausschnitt** behält die Auflösung des Originals. |
| `pos_weight` | 4,0 | Risse sind wenige Prozent der Pixel. Ohne Gegengewicht lernt das Netz „überall Hintergrund" und hat 97 Prozent Trefferquote, ohne einen einzigen Riss gefunden zu haben. |
| BCE + Dice | je 1,0 | BCE allein kippt bei dieser Schieflage, Dice allein ist am Anfang instabil. |
| Verstärkung | spiegeln, drehen, Helligkeit, Rauschen, JPEG | Was die **Form** verbiegt oder die **Kanten** verschmiert, lehrt das Netz etwas Falsches — genau dort, wo später gemessen wird. Kein elastisches Verzerren, kein starkes Weichzeichnen. |
| Aufteilung | über Dateien, `seed=42` | Zwei Ausschnitte desselben Fotos dürfen nicht auf beiden Seiten landen, sonst misst die Prüfung das Auswendiglernen mit. |

## Bewerten — und warum IoU allein irreführt

Für dünne Strukturen führt die übliche IoU in die Irre. Ein Riss von drei
Pixeln Breite, um **ein** Pixel danebengelegt, hat eine IoU nahe null —
obwohl er gefunden wurde. Umgekehrt belohnt IoU ein Netz, das den Riss
einfach dicker malt: Fläche ist billig. Beides ist genau verkehrt herum zu
dem, was dieser Dienst braucht — **die Lage** muss stimmen, und **die Breite**
darf nicht geschönt sein.

Deshalb drei Zahlen nebeneinander (`training/metrics.py`):

| Maß | Beantwortet | Zielrichtung |
|---|---|---|
| `iou`, `f1` | Vergleichbarkeit mit der Literatur | höher besser |
| `tolerant_f1` | Wurde der Riss **gefunden**? Treffer zählt im Toleranzband von 2 px | höher besser |
| `width_bias` | Malt das Netz **zu breit**? Flächenverhältnis je Längeneinheit | nahe **1,0** |

`width_bias` ist das Maß, das zur Messung passt, und es hat keine Entsprechung
in der Literatur. Steht es bei 1,3, meldet der Dienst jede Rissbreite um
30 Prozent zu groß — und zwar plausibel, also unauffällig. Es ist die Zahl,
die vor der Inbetriebnahme stimmen muss.

Ausgewählt wird der beste Prüfpunkt nach `tolerant_f1`, nicht nach IoU.

```
docker run --rm -v "$(pwd -W):/work" -w /work sada-vision:train \
  python training/evaluate.py runs/crack_unet_r18/best.pt \
    --data data/crackseg9k/test
```

Die Bewertung läuft auf dem **ganzen Bild** über die Kachelung des Dienstes,
nicht auf 256er-Ausschnitten — Kachelränder sind eine eigene Fehlerquelle,
und sie soll sie sehen. Zusätzlich sucht sie die Schwelle ab: welcher Wert
liefert das beste `tolerant_f1`, und wo steht `width_bias` dabei. Diese
Schwelle landet in der Begleitdatei und wird vom Dienst übernommen.

## Ausliefern

```
docker run --rm -v "$(pwd -W):/work" -w /work sada-vision:train \
  python training/export_onnx.py runs/crack_unet_r18/best.pt \
    --out models/crack_unet_r18.onnx
```

Es entstehen **zwei** Dateien, und beide werden gebraucht:

```
models/crack_unet_r18.onnx    das Netz
models/crack_unet_r18.json    Normierung, Kachelgröße, Schwelle, Herkunft
```

Die Begleitdatei ist kein Beiwerk. Stehen dort andere Mittelwerte als beim
Training, rechnet der Dienst mit falsch normierten Bildern — und das fällt
nicht als Fehler auf, sondern als schlechte Erkennung. Deshalb wird sie
erzeugt und nicht von Hand geschrieben.

Der Export prüft sich selbst: dasselbe Bild durch torch und durch
onnxruntime, größte Abweichung muss unter `1e-3` liegen. Sonst bricht er ab,
statt ein stilles Problem auszuliefern.

Danach in Betrieb nehmen:

```
SADAVISION_CRACK_MODEL=crack_unet_r18.onnx
SADAVISION_MODEL_DIR=/models
```

und prüfen, dass `/health` nicht mehr `degraded` meldet und
`model.trained = true` in der Antwort steht.

## Feintuning auf eigene Fotos

Der wichtigste nächste Schritt nach dem ersten Modell. **Öffentliche
Datensätze kennen euren Beton nicht** — nicht eure Schalung, nicht eure
Oberflächen, nicht eure Beleuchtung, nicht die Kamera des iPads.

1. **Sammeln.** 300 bis 500 Fotos reichen für einen deutlichen Sprung, wenn
   sie aus dem echten Einsatz stammen. Dabei bewusst auch das Schwierige:
   nasse Flächen, Schattenkanten, Schalungsfugen, Kabelkanäle, Beschriftungen
   — also alles, was der Ridge-Filter fälschlich für einen Riss hält.
2. **Annotieren.** CVAT (MIT) selbst gehostet oder Roboflow. Ausgabe ist eine
   Binärmaske je Bild, Dateinamen gleich, Ablage wie CrackSeg9k
   (`images/`, `masks/`).
3. **Weitertrainieren**, nicht von vorn: das CrackSeg9k-Modell als
   Ausgangspunkt, kleine Lernrate (etwa ein Fünftel), wenige Epochen. Dabei
   einen Teil CrackSeg9k beimischen, sonst vergisst das Netz, was es kann.
4. **Getrennt bewerten** — auf eigenen Fotos *und* auf CrackSeg9k. Wird eines
   von beiden schlechter, stimmt die Mischung nicht.

## Rissbreite messen: was das Modell leistet und was nicht

Das Netz liefert eine **Wahrscheinlichkeit je Pixel**. Die Breite entsteht
danach, in `pipeline/width.py`, quer zum Verlauf und mit Interpolation
zwischen den Abtastpunkten — nicht aus der harten Maske. Das ist der Grund,
warum der Dienst das Wahrscheinlichkeitsbild braucht und nicht nur eine
Maske, und warum `width_bias` mitgemessen wird: ein Netz mit systematischem
Breitenfehler verschiebt jede Messung, egal wie gut die Nachbearbeitung ist.

Die Grenze der Genauigkeit ist am Ende die Kamera, nicht das Netz. Bei
0,08 mm je Pixel — ein typischer Wert aus einem halben Meter Abstand — ist
ein Riss von 0,2 mm zweieinhalb Pixel breit. Darunter wird jede Angabe zur
Schätzung. Wer 0,1 mm auflösen will, muss näher heran oder ein Makroobjektiv
nehmen; das ist Physik, kein Modellproblem.
