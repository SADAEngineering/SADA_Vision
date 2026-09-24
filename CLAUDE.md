# SADA Vision

Erkennung und Vermessung von Bauteilschäden auf Fotos — Risse zuerst,
Schrauben und Korrosion danach. Ein eigenständiger Dienst in einem Container,
der später in TraceForm und in die AR-App einrückt.

Python 3.12 · FastAPI · PyTorch (nur Training) · ONNX Runtime (Betrieb) ·
OpenCV · scikit-image · U-Net aus `segmentation_models_pytorch`

> Fassung 1.0, 24.09.2026 · übergeordnet: SADA-Gesamthandbuch
>
> Erster Stand. Der Dienst läuft, der Vertrag steht, das Modell ist in Arbeit.
> Was noch fehlt, steht unter „Wo wir stehen".

## Was das ist

Ein Bild geht hinein, ein **vermessener Befund** kommt heraus: wo der Riss
liegt (als geordnete Koordinatenfolge `y, x`), wie breit er an jeder Stelle
ist (in Pixeln, und in Millimetern sobald ein Maßstab feststeht), wie lang er
ist, wie er verläuft und in welches Band er fällt.

Kein Bildbetrachter, kein Gutachten, keine Bauteilverwaltung — das ist
TraceForm. Dieser Dienst kennt **kein** Bauteil, keinen Mandanten, keinen
Auftrag. Er bekommt ein Foto und gibt Geometrie zurück. Genau deshalb kann er
als ein Container neben TraceForm stehen, gegen die AR-App sprechen und
irgendwann auch für ein anderes Produkt arbeiten.

| Anwendungsfall | Stand | Aufgabe im Vertrag |
|---|---|---|
| **Risse im Beton** | gebaut, Modell in Arbeit | `crack` |
| Schrauben | geplant | `bolt` |
| Korrosion | geplant | `corrosion` |

Alle drei laufen über denselben Pfad, denselben Vertrag und dieselbe
Nachbearbeitung. Was sich unterscheidet, ist das Modell und das, was aus der
Maske abgeleitet wird.

## Struktur

```
src/sada_vision/
  api/            FastAPI-Host, Routen, DTOs, Abbildung intern -> Draht
  domain/         Interne Typen. Kennen numpy, kennen keinen Vertrag.
  pipeline/       Der Lauf: Vorverarbeitung, Geometrie, Breite, Einordnung
  models/         Segmentierer: ONNX im Betrieb, Ridge-Filter als Notbehelf
  scale/          Pixel -> Millimeter: Marker, Vorgabe, LiDAR
training/         Datensatz, Training, Bewertung, ONNX-Export - fährt nie mit
deploy/           Dockerfile (Dienst), Dockerfile.train, compose
tools/            vision.ps1 - die langen Docker-Aufrufe in kurz
tests/            Auch die synthetischen Risse mit bekannter Breite
docs/             Vertrag, TraceForm-Anbindung, AR-Rückprojektion, Modell
models/           Die Gewichte. Nicht in Git.
samples/          Probebilder. Nicht in Git.
```

**In `deploy/Dockerfile` steht `runtime` zuletzt, und das muss so bleiben.**
Ein Build ohne `--target` nimmt die letzte Stufe. Stünde dort die Test-Stufe,
lieferte `docker build -t sada-vision .` ein Image aus, das beim Start pytest
aufruft — ohne jede Meldung, die darauf hinweist.

**Die Richtung ist eine Einbahnstraße:** `api` kennt `pipeline`, `pipeline`
kennt `models` und `domain`, `domain` kennt nichts. Der Vertrag lebt
ausschließlich in `api/schemas.py`, die Übersetzung ausschließlich in
`api/mapping.py`. Wer in `pipeline/` ein DTO importiert, hat die Grenze
gerissen.

## Die acht Regeln

**1 · Der API-Vertrag ist bindend und nur additiv.** Felder hinzufügen ja,
umbenennen oder entfernen nein — dieselbe Regel und derselbe Grund wie in
TraceForm: Unity deserialisiert mit `JsonUtility`, und ein umbenanntes Feld
liefert dort stumm `null` statt eines Fehlers. Daraus folgt die Form: keine
Wörterbücher, keine Aufzählungen, keine Polymorphie, kein `null`
(unbekannte Zahl = `-1`, unbekannter Text = `""`), Objektlisten im Umschlag
`{ "items": [...] }`. Der Verlauf ist ein **flaches** Zahlenfeld
`path_yx = [y0, x0, y1, x1, ...]` mit `point_count` daneben. Alles steht in
`docs/API_Vertrag.md`.

**2 · Millimeter werden nie geraten.** Ohne Maßstab meldet der Dienst `-1`,
nicht eine Schätzung. Eine Rissbreite ohne Einheit ist eine Zahl; eine
falsche Rissbreite mit Einheit ist ein Befund, auf den jemand eine
Entscheidung baut. `scale.source` sagt immer, woher der Maßstab kam, und
`scale.confidence`, wie gut. Widersprechen sich zwei Quellen um mehr als
15 Prozent, steht das als Warnung in der Antwort.

**3 · Ein Modell ohne Gewichte lügt nicht, es sagt es.** Fehlt die
ONNX-Datei, läuft der Ridge-Filter als Notbehelf weiter — aber jede Antwort
trägt dann `model.trained = false`, `/health` meldet `degraded`, und im
Vorschaubild steht `UNTRAINED FALLBACK` quer über dem Rand. Ein Dienst, der
ohne Modell gar nicht startet, lässt sich nicht in Betrieb nehmen, bevor das
Modell fertig ist; einer, der stillschweigend Unsinn misst, ist schlimmer.

**4 · Im Dienst-Image liegt kein PyTorch.** Trainiert wird außerhalb,
ausgeliefert wird ONNX. Das sind rund 2 GB je laufendem Container, und
on-prem zählt das. `training/` ist kein Teil des Dienstes und wird nie von
`src/` importiert — nur umgekehrt (`evaluate.py` benutzt die Kachelung des
Dienstes, damit Bewertung und Betrieb dasselbe rechnen).

**5 · Nichts Virales im Auslieferungsweg.** Ultralytics YOLO steht unter
AGPL-3.0; bei einem Netzwerkdienst greift die Netzklausel, und der Kunde
bekäme Anspruch auf den Quelltext. Deshalb: `segmentation_models_pytorch`
(MIT), torchvision-Encoder (BSD), OpenCV (Apache 2.0), scikit-image (BSD),
FastAPI (MIT), Datensatz CrackSeg9k (CC0). **Vor jeder neuen Abhängigkeit
die Lizenz prüfen** — und bei Copyleft nicht „später klären".

**6 · Koordinaten sind Pixel des Originalbildes, in `(y, x)`.** Auch wenn
intern verkleinert gerechnet wurde, auch wenn im Zuschnitt gearbeitet wurde.
Der Aufrufer legt die Antwort auf sein Foto und trifft. Ursprung links oben.
Winkel: 0 Grad waagerecht, positiv gegen den Uhrzeigersinn, Bereich
`[-90, 90)`.

**7 · Gemessen wird quer zum Verlauf, auf dem Wahrscheinlichkeitsbild.**
Nicht auf der harten Maske, nicht entlang der Bildachsen. Die
Distanztransformation rundet auf ganze Pixel — bei einem Riss von drei Pixeln
sind das 30 Prozent Sprungweite, und bei 0,2 gegen 0,3 mm entscheidet genau
das. Das lotrechte Verfahren interpoliert den Rand zwischen zwei
Abtastpunkten und ist deshalb die Vorgabe. `distance_transform` bleibt als
schnelle Alternative erreichbar.

**8 · Was der Dienst nicht weiß, sagt er nicht.** Die Einordnung (`linear`,
`branched`, `map`, `horizontal`/`vertical`/`diagonal`, das Breitenband) ist
**abgeleitete Geometrie, kein Gutachten**. Die Ursache eines Risses —
Schwinden, Setzung, Bewehrungskorrosion, Zwang — folgt daraus nicht, und die
Zulässigkeit einer Breite hängt an der Expositionsklasse, die dieser Dienst
nicht kennt. Er liefert den Befund, TraceForm und ein Mensch die Bewertung.

## Tests

**Synthetische Risse mit bekannter Breite sind der Kern.** Der Zweck des
Dienstes ist eine Zahl in Millimetern; ob die stimmt, zeigt sich nur an einem
Riss, dessen Breite man vorher kennt. Auf echten Fotos weiß das niemand.
`tests/synthetic.py` zeichnet sie — vierfach vergrößert und dann verkleinert,
damit eine Breite von 2,5 Pixeln überhaupt darstellbar ist.

Sechs Gruppen müssen existieren:

1. **Messung gegen bekannte Breite** (`test_width.py`) — und der Nachweis,
   dass das lotrechte Verfahren genauer ist als die Distanztransformation.
   Sonst wäre die Vorgabe eine Behauptung.
2. **Geometrie** (`test_geometry.py`) — Skelett, Astzerlegung, Ringe,
   Vereinfachung, Hauptrichtung. Der T-Test ist der wichtigste: eine Kreuzung
   ist im Skelett fast nie *ein* Pixel, und wer das übersieht, bekommt aus
   einem T acht Äste statt drei.
3. **Durchstich** (`test_pipeline.py`) — mit einem Orakel-Segmentierer, der
   die wahre Maske kennt. Damit wird die Pipeline geprüft und nicht die Güte
   eines Netzes: Koordinatenumrechnung, Zuschnitt, Verkleinerung, Maßstab.
4. **Vertrag** (`test_api.py`) — kein `null` in der ganzen Antwort, Listen im
   Umschlag, `path_yx` doppelt so lang wie `point_count`, Koordinaten im
   Bild. Wer einen dieser Tests rot macht, bricht einen Aufrufer.
5. **ONNX-Pfad** (`test_onnx.py`) — der, der im Betrieb läuft. Gegen ein
   365 Byte großes Testmodell, das kein Netz ist, sondern ein fester
   Rechenweg: dunkle Pixel gelten als Riss. Die teuerste Fehlerklasse ist
   hier die **falsche Normierung** — sie fällt nicht als Fehler auf,
   sondern als schlechte Erkennung, Monate später.
6. **Kachelung und Last** (`test_tiling.py`, `test_workload.py`) — jeder
   Pixel genau einmal abgedeckt, keine Naht, und der Dienst bleibt
   während einer Rechnung ansprechbar. Beides sind Fehler, die nur unter
   Last auftreten und die eine einzelne Anfrage nie bemerkt.

Die Güte des *Modells* prüft keiner davon — das machen
`training/evaluate.py` (Erkennung) und `training/measure_width_error.py`
(Messung) auf dem zurückgehaltenen Testteil, über die Kachelung des
Dienstes und nicht auf 256er-Ausschnitten.

Ausgeführt wird im Container, weil es auf den Arbeitsrechnern kein Python
gibt:

```
docker build -f deploy/Dockerfile --target dev -t sada-vision:test .
docker run --rm -v "$(pwd -W):/app" sada-vision:test pytest -q
```

## Maße — und warum nicht IoU allein

Für dünne Strukturen führt die übliche IoU in die Irre. Ein Riss von drei
Pixeln, um ein Pixel danebengelegt, hat eine IoU nahe null, obwohl er
gefunden wurde. Umgekehrt belohnt IoU ein Netz, das den Riss einfach dicker
malt — Fläche ist billig. Beides ist verkehrt herum zu dem, was hier zählt.
Deshalb drei Zahlen nebeneinander (`training/metrics.py`):

| Maß | Beantwortet |
|---|---|
| `iou`, `f1` | Vergleichbarkeit mit der Literatur |
| `tolerant_f1` | Wurde der Riss **gefunden**? Treffer im Toleranzband |
| `width_bias` | Malt das Netz **zu breit**? Über 1,0 = zu dick, und damit jede gemeldete Breite zu groß |

Ausgewählt wird der beste Prüfpunkt nach `tolerant_f1`, nicht nach IoU.

**Die Abnahmezahl ist keine davon.** Sie kommt aus
`training/measure_width_error.py`: dieselbe Breitenmessung einmal auf der
annotierten Wahrheit und einmal auf der Vorhersage, Differenz in Pixeln.
Das ist der Fehler der ganzen Kette — Netz, Schwelle, Skelett, Lot,
Interpolation. Bei 0,08 mm/px sind 0,5 px gleich 0,04 mm, und **das** ist
die Zahl, die in eine Produktbeschreibung gehört: nicht „IoU 0,74", sondern
„misst auf 0,04 mm genau". Systematischer Versatz und Streuung werden
getrennt ausgewiesen — der Versatz lässt sich über die Schwelle korrigieren,
die Streuung bleibt.

## Konventionen

- Wurzel `sada_vision`. Bezeichner und Dateinamen englisch, **Kommentare und
  Dokumentation deutsch, Oberfläche und Fehlermeldungen des Servers
  englisch** — wie in TraceForm, aus demselben Grund: die Fehler stehen in
  der Maske.
- DTOs enden auf `Dto` und leben nur in `api/schemas.py`.
- Aufgabenbezeichner sind kleingeschrieben und einsilbig: `crack`, `bolt`,
  `corrosion`. Sie stehen im Pfad, im Modellnamen und im `label`.
- Einstellungen kommen aus Umgebungsvariablen mit dem Präfix
  `SADAVISION_`; Standardwerte in `config.py`, nirgends sonst.
- Serilog-Äquivalent ist `structlog`: strukturiert nach stdout, mit
  `request_id`. Ereignisnamen deutsch und gepunktet (`modell.geladen`,
  `bild.abgelehnt`).
- Durchgehend Typannotationen, `from __future__ import annotations` oben.
- `ruff` mit Zeilenlänge 100 ist verbindlich.
- Secrets nur aus Umgebungsvariablen, **nie** im Repo.
- Vor einer Änderung committen.

## Zwei Rechner

Gilt wie in TraceForm: **Laptop** (`D:\SADA\04_Tool\`) und **Cloud-Rechner**
(`%USERPROFILE%\source\repos\SADAEngineering\`). Dieses Repo heißt auf beiden
`SADA_Vision`, liegt auf dem Cloud-Rechner neben `SADA_TraceForm`. Die
vollständige Tabelle der Nachbarrepos steht in `SADA_TraceForm/CLAUDE.md` —
hier wird sie nicht verdoppelt.

Zwei Besonderheiten dieses Repos:

- **Kein Python auf den Arbeitsrechnern.** Alles läuft über Docker. Wer
  `python` tippt, bekommt den Microsoft-Store.
- **Keine GPU.** Ein CPU-Lauf ist eine Rauchprobe und ein brauchbarer
  Anfang, kein Auslieferungsmodell. Der Lauf, der in Betrieb geht, gehört
  auf Colab, RunPod oder eine Karte — `training/configs/crack_unet_r34_gpu.yaml`.
- **`data/` und `runs/` stehen nicht in Git** und wandern nicht mit. Der
  Datensatz wird neu geholt (`training/datasets/crackseg9k.py`), die
  Gewichte kommen aus der Registry.

## Vertiefung — bei Bedarf lesen

| Datei | Wann |
|---|---|
| `docs/API_Vertrag.md` | bevor du ein Feld anfasst oder einen Aufrufer schreibst |
| `docs/Modell_und_Training.md` | Datensatz, Training, Bewertung, Export, Feintuning auf eigene Fotos |
| `docs/TraceForm_Anbindung.md` | wie der Befund in TraceForm ankommt und was dort daraus wird |
| `docs/AR_Rueckprojektion.md` | **bevor du an der Unity-Seite anfängst** — der Weg vom Bildpfad zum 3D-Verlauf auf der Bauteiloberfläche |
| `SADA_TraceForm/CLAUDE.md` | Konventionen, Rechner, Betrieb über beide Produkte |

## Wo wir stehen

**Gebaut und geprüft:** Dienst, Vertrag, Pipeline, Breitenmessung, alle drei
Maßstabswege, Vorschaubild, Container, Trainingsweg samt Datensatz, Bewertung
und ONNX-Export.

**In Arbeit:** das Rissmodell. Bis es liegt, läuft der Notbehelf und sagt es.

**Als Nächstes, in dieser Reihenfolge:**

1. Rissmodell auf GPU fertig trainieren, bewerten, exportieren, in Betrieb
2. Eigene SADA-Fotos annotieren und feintunen — **das ist kein Feinschliff,
   sondern der eigentliche zweite Schritt.** CrackSeg9k kommt überwiegend aus
   der Straßenzustandserfassung: der Median der annotierten Risse liegt bei
   6,7 px mittlerer Breite, bei 0,08 mm/px also rund 0,54 mm. Die
   Grenzwerte im Stahlbeton liegen bei 0,2 bis 0,4 mm — zwei bis fünf Pixel.
   Das Netz lernt dort also überwiegend Risse, die breiter sind als die, auf
   die es ankommt. Einzelheiten in `docs/Modell_und_Training.md`
3. Anbindung an TraceForm (`docs/TraceForm_Anbindung.md`)
4. AR-Rückprojektion in der Unity-App (`docs/AR_Rueckprojektion.md`)
5. Schrauben (`bolt`), dann Korrosion (`corrosion`)

**Noch nicht entschieden:** ob der Dienst einen eigenen Modul-Schlüssel in
der Lizenzierung bekommt (`traceform.vision`?) oder unter
`traceform.reliability` mitläuft. Das ist eine Produktfrage, keine technische.
