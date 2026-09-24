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

### Was in diesem Datensatz wirklich drin ist — und warum das zählt

Nachgemessen an 800 zufälligen Masken des Testteils (Fläche geteilt durch
Skelettlänge, also die mittlere Breite des annotierten Risses):

| Mittlere Breite der Annotation | Anteil |
|---|---|
| bis 6 px | 40 % |
| 6–12 px | 30 % |
| 12–25 px | 19 % |
| über 25 px | 10 % |
| *ohne Riss (Gegenbeispiele)* | *16 % aller Bilder* |

Median: **6,7 px**. Und darin steckt die wichtigste Einschränkung dieses
Datensatzes für unseren Zweck.

CrackSeg9k kommt überwiegend aus der **Straßenzustandserfassung** — Crack500,
GAPs und CrackTree sind Fahrbahnaufnahmen. Die dortigen Risse sind
zentimeterbreite Fahrbahnrisse, Ausbrüche und Schlaglöcher; ein Teil der
Masken ist gar keine Linie, sondern ein Fleck. Ein Beispiel aus dem Testteil
zeigt eine Abplatzung im Asphalt, annotiert als „Riss".

Für ein Bauwerksprüfungs-Produkt heißt das zweierlei:

**Erstens: die Maßstäbe passen nicht zusammen.** Bei einem typischen Maßstab
von 0,08 mm/px ist ein Riss von 6,7 px rund **0,54 mm** breit. Die Grenzwerte,
um die es im Stahlbeton geht, liegen bei 0,2 bis 0,4 mm — also bei zwei bis
fünf Pixeln. Das Netz lernt hier also überwiegend Risse, die **breiter sind
als die, auf die es bei euch ankommt**. Das ist ein guter Teil der Erklärung
für `width_bias > 1`.

**Zweitens: das ist kein Mangel, sondern ein Auftrag.** Als Vortraining ist
der Datensatz hervorragend — 9.000 Bilder, CC0, mit Gegenbeispielen, und das
Netz lernt daran, was eine rissartige Struktur überhaupt ist. Was es *nicht*
lernt, ist euer Beton bei eurem Aufnahmeabstand. Deshalb steht das Feintuning
auf eigenen Fotos nicht als Kür am Ende, sondern als der eigentliche zweite
Schritt — und die eigenen Fotos sollten **nah genug** aufgenommen sein, dass
ein 0,2-mm-Riss über mehr als zwei, drei Pixel geht.

### Der Hebel: nur auf linienhaften Annotationen trainieren

```
--max-annotation-width 12
```

Lässt alle Masken weg, deren mittlere Breite (Fläche / Skelettlänge) über
dem Wert liegt — also die Abplatzungen, Schlaglöcher und zentimeterbreiten
Fahrbahnrisse. **Die Gegenbeispiele bleiben drin**; ohne sie hält das Netz
jede Schalungsfuge für einen Riss.

Die Breiten werden einmal gemessen und neben dem Datensatz abgelegt
(`annotation_widths.json`), danach kostet der Filter nichts.

Was das bringt und was es kostet:

| Grenze | Es bleiben | Wirkung |
|---|---|---|
| aus | 100 % | wie gehabt — mehr Daten, aber der Breitenmaßstab stimmt nicht |
| 20 px | ~75 % | nur die gröbsten Flächen raus, kaum Datenverlust |
| 12 px | ~45 % | klar auf Linien ausgerichtet, die Hälfte der Daten weg |
| 8 px | ~27 % | nahe an Haarrissen, aber zu wenig zum Trainieren von Grund auf |

**Der Filter ist kein Selbstzweck.** Wer von Grund auf trainiert, will die
Datenmenge und lässt ihn aus. Wer ein vortrainiertes Netz auf feine Risse
nachzieht, setzt ihn auf 12 und mischt eigene Fotos dazu. Ob er die
Breitenverzerrung tatsächlich senkt, zeigt `measure_width_error.py` — nicht
die Überlegung.

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
docker run --rm --shm-size=2g --cpus 8 \
  -v "$(pwd -W):/work" -w /work sada-vision:train \
  python training/train.py --config training/configs/crack_unet_r18.yaml
```

> **`--shm-size=2g` ist Pflicht.** Docker gibt einem Container 64 MB unter
> `/dev/shm`. Die DataLoader-Arbeiter reichen ihre Stapel darüber weiter und
> laufen voll — der Lauf stirbt dann nach Minuten mit
> `DataLoader worker is killed by signal: Bus error`, und zwar mitten in der
> Epoche. Wer die Meldung nicht kennt, sucht sie im eigenen Code.

**Die Epochenzahl muss zum Zeitfenster passen.** Der Kosinus-Zeitplan senkt
die Lernrate über genau die eingestellte Spanne ab. Ein auf 14 Epochen
eingestellter Lauf, der bei Epoche 3 abgebrochen wird, endet mit hoher
Lernrate — und liefert ein schlechteres Modell als ein sauber zu Ende
gefahrener Lauf über 3 Epochen. Erst die Schrittzeit messen
(`--limit-train 672 --epochs 1`), dann die Epochen festlegen.

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

### `pos_weight` und `width_bias` ziehen gegeneinander

Das ist kein Randproblem, sondern die zentrale Abwägung dieses Modells, und
sie wurde beim ersten Lauf sofort sichtbar: nach einer Epoche stand
`tolerant_f1` bei 0,68 — und `width_bias` bei **1,35**.

Der Grund liegt in der Verlustfunktion. `pos_weight = 4` sagt dem Netz, dass
ein übersehener Risspixel viermal so teuer ist wie ein fälschlich als Riss
markierter. Das ist richtig, damit überhaupt etwas gefunden wird — führt aber
dazu, dass das Netz im Zweifel **großzügig** malt. Bei einem Gegenstand, der
drei Pixel breit ist, wird aus Großzügigkeit sofort ein Drittel mehr Breite.

Beides gleichzeitig lässt sich nicht über den Verlust lösen. Der Weg ist die
Arbeitsteilung:

- **Das Training** optimiert auf *Finden* (`pos_weight` hoch, Auswahl nach
  `tolerant_f1`).
- **Die Schwelle** korrigiert das *Messen*. `evaluate.py` sucht sie ab, und
  eine höhere Schwelle schneidet den großzügigen Rand wieder weg.

Deshalb wird die gefundene Schwelle in die Begleitdatei geschrieben, und
deshalb hat sie im Dienst Vorrang vor der globalen Vorgabe. Rangfolge:
**Aufrufer → Modell → Vorgabe.** Fällt die mittlere Stufe weg, misst der
Dienst mit einem fremden Wert und meldet systematisch zu breit — ohne dass
irgendetwas darauf hinweist. `tests/test_onnx.py` hält das fest.

Bleibt `width_bias` auch bei der besten Schwelle deutlich über 1,1, hilft
keine Schwelle mehr: dann ist entweder zu kurz trainiert oder `pos_weight`
zu hoch.

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

## Die Abnahmezahl: Wie genau wird gemessen?

IoU, F1 und tolerantes F1 beantworten, ob der Riss **gefunden** wurde. Keine
davon beantwortet die Frage, auf die es bei diesem Produkt ankommt: *wie weit
liegt die gemeldete Rissbreite neben der wahren?*

Das lässt sich messen, ohne ein einziges Foto nachzumessen. Die Masken des
Datensatzes sind von Menschen gezeichnet — sie **sind** die Wahrheit. Also
läuft dieselbe Geometrie- und Breitenmessung einmal auf der wahren Maske und
einmal auf der Vorhersage, und die beiden Zahlen werden verglichen:

```
docker run --rm -v "$(pwd -W):/work" -w /work sada-vision:train \
  python training/measure_width_error.py runs/crack_unet_r18/best.pt \
    --data data/crackseg9k/test --limit 300
```

Heraus kommt der Fehler der **ganzen Kette** — Netz, Schwelle, Skelett, Lot,
Interpolation — in Pixeln, dazu die Zahl der nicht gefundenen und der
erfundenen Risse. Mit einem Maßstab von 0,08 mm/px sind 0,5 px Fehler
0,04 mm, und **das** ist die Zahl, die in eine Produktbeschreibung gehört:
nicht „IoU 0,74", sondern „misst auf 0,04 mm genau".

Zwei Zahlen daraus sind getrennt zu lesen:

- `median_error_px` ist der **systematische Versatz**. Steht er bei +0,8,
  meldet der Dienst jeden Riss um 0,8 px zu breit — das lässt sich über die
  Schwelle korrigieren und gehört korrigiert.
- `median_abs_error_px` ist die **Streuung**. Die bleibt, und sie ist die
  ehrliche Angabe der Genauigkeit.

Die Grenze des Verfahrens gehört in jede Aussage, die daraus abgeleitet wird:
verglichen wird gegen die *gezeichnete* Maske, nicht gegen den Riss. Wo der
Annotierende die Kante anders gesetzt hat als die Physik, steckt der Fehler
schon in der Wahrheit. Eine belastbare absolute Genauigkeit gibt nur ein
Vergleich gegen ein Rissbreitenlineal oder ein Mikroskop an einem echten
Bauteil — das steht aus.

## Der erste Lauf — die Zahlen

> 25.09.2026 · `crack_unet_r18` · 5 Epochen auf 8 CPU-Kernen, 3 h 24 min ·
> CrackSeg9k, ungefiltert · U-Net mit ResNet-18, 256er-Ausschnitte

**Training** (Prüfteil, je Epoche):

| Epoche | tolerantes F1 | `width_bias` |
|---|---|---|
| 1 | 0,676 | 1,345 |
| 2 | 0,753 | 1,229 |
| 3 | 0,735 | 1,366 |
| 4 | 0,755 | 1,350 |
| **5** | **0,778** | 1,312 |

Die Erkennung steigt stetig, die Breitenverzerrung bleibt bei rund 1,3 hängen.
Genau so war es zu erwarten — siehe den Abschnitt zu `pos_weight` oben.

**Schwellensuche** (Testteil, 500 Bilder). Das tolerante F1 ist über den
ganzen Bereich fast flach, die Verzerrung fällt stetig:

| Schwelle | tolerantes F1 | `width_bias` |
|---|---|---|
| 0,50 | 0,803 | 1,407 |
| 0,75 | **0,808** (bestes Finden) | 1,285 |
| 0,85 | 0,807 | 1,193 |
| **0,93** | 0,797 (**gewählt**) | **1,059** |
| 0,95 | 0,788 | 0,981 |

Die Auswahlregel kostet 1,3 Prozentpunkte beim Finden und drückt die
Verzerrung von **+28,5 auf +5,9 Prozent**. Für ein Messprodukt ist das der
richtige Handel.

**Messgenauigkeit** (Testteil, 374 Bilder, nur linienhafte Annotationen bis
12 px, Schwelle 0,93):

| Größe | typische Wahrheit | Versatz | Streuung (Median \|e\|) |
|---|---|---|---|
| mittlere Breite | 6,2 px | +1,13 px (+19 %) | **1,90 px** |
| größte Breite | 16,9 px | −1,43 px (−11 %) | 5,01 px |
| Länge | 432 px | −28,7 px (−7 %) | 65 px |

26 von 400 annotierten Rissen nicht gefunden, **0 erfunden**.

### Was diese Zahlen bedeuten — und was nicht

**Die Kette steht.** Vom Foto bis zum Millimeterwert läuft alles, der Export
stimmt auf 7,6 · 10⁻⁶ mit PyTorch überein, die Schwelle kommt im Dienst an,
und auf einem echten Rissfoto kommt in 157 ms ein sauber verfolgter Befund
heraus.

**Die Messgenauigkeit reicht noch nicht.** 1,9 px Streuung auf einem
typischen 6,2-px-Riss sind rund 30 Prozent. Bei 0,08 mm/px ist das ein
Fehler von etwa 0,15 mm — für eine Entscheidung zwischen 0,2 und 0,3 mm
**zu grob**. Das ist keine Schwäche der Nachbearbeitung: auf synthetischen
Rissen bekannter Breite misst dieselbe Kette auf unter 0,4 px genau
(`tests/test_width.py`). Es ist die Güte dieses Modells.

Drei Hebel, in der Reihenfolge ihrer Wirkung:

1. **Richtig trainieren.** 5 Epochen auf CPU gegen 60 auf einer GPU mit
   384er-Ausschnitten und ResNet-34. Das ist der größte Sprung und kostet
   zwei Stunden Rechenzeit.
2. **Eigene Fotos.** Siehe oben: der Datensatz kennt euren Beton nicht, und
   seine Risse sind im Mittel zwei- bis dreimal breiter als eure.
3. **Näher heran.** Ein 0,2-mm-Riss muss über mehr als zwei Pixel gehen,
   sonst ist jede Angabe eine Schätzung. Das ist Physik, kein Modellproblem.

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
