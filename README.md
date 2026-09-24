# SADA Vision

Erkennung und Vermessung von Bauteilschäden auf Fotos. Ein Bild geht hinein,
ein vermessener Befund kommt heraus: **wo** der Riss liegt, **wie breit** er
an jeder Stelle ist, **wie** er verläuft.

Risse sind gebaut. Schrauben und Korrosion folgen unter denselben Pfaden und
demselben Vertrag.

```
POST /api/v1/detect/crack     Bild rein, Befund raus
POST /api/v1/preview/crack    dasselbe, als gezeichnetes PNG
GET  /health                  Zustand, mit Modellangabe
GET  /docs                    OpenAPI
```

## In fünf Minuten

Es braucht nur Docker — kein Python auf dem Rechner.

```bash
# Bauen
docker build -f deploy/Dockerfile -t sada-vision:dev .

# Starten
docker run --rm -p 127.0.0.1:8080:8080 \
  -v "$(pwd -W)/models:/models:ro" \
  sada-vision:dev

# Fragen
curl -s -F "image=@foto.jpg" -F "marker_size_mm=60" \
     http://127.0.0.1:8080/api/v1/detect/crack | jq .summary

# Ansehen
curl -s -F "image=@foto.jpg" -F "mm_per_px=0.08" \
     http://127.0.0.1:8080/api/v1/preview/crack -o befund.png
```

Oder mit compose: `docker compose -f deploy/compose.yml up --build`

## Ohne Modell läuft er trotzdem — und sagt es

Liegt keine ONNX-Datei in `models/`, springt ein Ridge-Filter ein. Die
Geometrie und die Breitenmessung arbeiten dann genauso sauber, aber die
**Erkennung** ist nicht belastbar: jede Schalungsfuge, jedes Kabel, jeder
Schattenriss kommt mit.

Erkennbar ist dieser Zustand an drei Stellen, alle absichtlich schwer zu
übersehen:

- `model.trained = false` in **jeder** Antwort
- `/health` meldet `degraded`
- im Vorschaubild steht `UNTRAINED FALLBACK` quer über dem oberen Rand

Ein Dienst, der ohne Modell gar nicht erst startet, lässt sich nicht in
Betrieb nehmen, bevor das Modell fertig ist. Einer, der stillschweigend
Unsinn misst, ist schlimmer. Also: er läuft, und er sagt Bescheid.

## Was herauskommt

```json
{
  "scale":   { "source": "aruco", "known": true, "mm_per_px": 0.0812 },
  "summary": { "instance_count": 2, "severity": "moderate",
               "max_width_mm": 0.277, "total_length_mm": 147.8 },
  "instances": { "items": [ {
      "instance_id": 0,
      "pattern": "branched", "orientation_class": "vertical",
      "width_max_mm": 0.277, "width_mean_mm": 0.183, "length_mm": 98.4,
      "tortuosity": 1.21, "branch_count": 2, "severity": "moderate",
      "paths": { "items": [ {
          "point_count": 146,
          "path_yx":  [312.5, 88.0, 314.1, 89.2, ...],
          "width_mm": [0.21, 0.22, 0.24, ...]
      } ] }
  } ] }
}
```

`path_yx` ist **flach** und **`(y, x)`**: `[y0, x0, y1, x1, …]`, Länge
`2 × point_count`. `width_mm[i]` gehört zum Punkt `i`. Der vollständige
Vertrag steht in [`docs/API_Vertrag.md`](docs/API_Vertrag.md).

## Millimeter statt Pixel

Ohne Maßstab meldet der Dienst `-1` — er rät nicht. Drei Wege führen zu
Millimetern, in dieser Rangfolge:

| Weg | Feld | Genauigkeit |
|---|---|---|
| **ArUco-Marker** im Bild | `marker_size_mm` | am besten — rechnet die Perspektive über eine Homographie mit |
| **Vorgabe** | `mm_per_px` | so gut wie die Annahme dahinter |
| **LiDAR** aus der AR-App | `depth_mm`, `focal_px`, `tilt_deg` | gut bei annähernd frontaler Fläche |

Ein gedruckter Marker neben dem Riss ist der billigste Weg zu einer
belastbaren Zahl: ein Blatt Papier, ein Klebestreifen, fertig.

Widersprechen sich zwei Quellen um mehr als 15 Prozent, steht das als Warnung
in der Antwort — dann stimmt eine der beiden Annahmen nicht.

## Wie die Breite gemessen wird

Nicht aus dem Rechteck, nicht aus der Fläche: **quer zum Verlauf, auf dem
Wahrscheinlichkeitsbild.**

```
Maske ─► Zusammenhangskomponenten ─► Skelett ─► Graph ─► Äste
                                                          │
                              je Stützstelle: Lot auf die Tangente,
                              Abtastung des Wahrscheinlichkeitsbildes,
                              Rand durch Interpolation ─► Subpixelbreite
```

Die naheliegende Distanztransformation rundet auf ganze Pixel — bei einem
Riss von drei Pixeln sind das 30 Prozent Sprungweite, und bei 0,2 gegen
0,3 mm entscheidet genau das. Sie bleibt als schnelle Alternative erreichbar
(`width_method=distance_transform`).

Dass das lotrechte Verfahren wirklich genauer ist, steht nicht als Behauptung
im Text, sondern als Test in `tests/test_width.py`.

## Entwickeln

```bash
# Tests (86 Stück, laufen im Container)
docker build -f deploy/Dockerfile --target dev -t sada-vision:test .
docker run --rm -v "$(pwd -W):/app" sada-vision:test pytest -q

# Aufräumen prüfen
docker run --rm -v "$(pwd -W):/app" sada-vision:test ruff check src training tests
```

Die Tests prüfen die Messung gegen **synthetische Risse bekannter Breite** —
auf echten Fotos weiß niemand, wie breit der Riss wirklich war.

## Modell trainieren

```bash
docker build -f deploy/Dockerfile.train -t sada-vision:train .

# Datensatz holen (CrackSeg9k, ~9.000 Bilder, CC0)
docker run --rm -v "$(pwd -W):/work" -w /work sada-vision:train \
  python training/datasets/crackseg9k.py --out data/crackseg9k

# Trainieren, bewerten, ausliefern
#   --shm-size=2g ist Pflicht: Dockers Vorgabe von 64 MB reicht den
#   DataLoader-Arbeitern nicht, und der Lauf stirbt mitten drin mit
#   "Bus error" statt mit einer brauchbaren Meldung.
docker run --rm --shm-size=2g -v "$(pwd -W):/work" -w /work sada-vision:train \
  python training/train.py --config training/configs/crack_unet_r18.yaml
docker run --rm -v "$(pwd -W):/work" -w /work sada-vision:train \
  python training/evaluate.py runs/crack_unet_r18/best.pt
docker run --rm -v "$(pwd -W):/work" -w /work sada-vision:train \
  python training/export_onnx.py runs/crack_unet_r18/best.pt \
    --out models/crack_unet_r18.onnx
```

Auf CPU ist das eine Rauchprobe. Der Lauf, der in Betrieb geht, gehört auf
eine GPU — Einzelheiten und Begründungen in
[`docs/Modell_und_Training.md`](docs/Modell_und_Training.md).

## Lizenzen der Bestandteile

Bewusst nichts Virales im Auslieferungsweg. **Ultralytics YOLO (AGPL-3.0)
wird nicht verwendet** — bei einem Netzwerkdienst greift die Netzklausel, und
der On-Prem-Kunde hätte Anspruch auf den Quelltext.

| Bestandteil | Lizenz |
|---|---|
| FastAPI, pydantic, `segmentation_models_pytorch` | MIT |
| OpenCV | Apache 2.0 |
| PyTorch, torchvision, scikit-image, scipy, numpy | BSD |
| ONNX Runtime | MIT |
| CrackSeg9k (Datensatz) | CC0 1.0 — Public Domain |

## Weiter

| Datei | Inhalt |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Das Handbuch: Regeln, Struktur, Stand |
| [`docs/API_Vertrag.md`](docs/API_Vertrag.md) | Der Vertrag auf dem Draht |
| [`docs/Modell_und_Training.md`](docs/Modell_und_Training.md) | Datensatz, Training, Maße, Export |
| [`docs/TraceForm_Anbindung.md`](docs/TraceForm_Anbindung.md) | Wie der Befund in TraceForm ankommt |
| [`docs/AR_Rueckprojektion.md`](docs/AR_Rueckprojektion.md) | Vom Bildpfad zum 3D-Verlauf auf dem Bauteil |
