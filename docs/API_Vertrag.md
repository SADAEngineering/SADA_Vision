# Der API-Vertrag

> Fassung 1.1 · Stichtag 24.09.2026 · ab hier **nur additiv**
>
> **1.1 (25.09.2026)** — drei Felder dazu, kein bestehendes angefasst:
> `touches_border` und `width_samples_excluded` je Befund,
> `width_at_junction` je Ast. Aufrufer der Fassung 1.0 laufen unverändert
> weiter; wer misst, sollte sie lesen.

Was hier steht, ist der Vertrag auf dem Draht. Er gilt für alle Aufrufer:
TraceForm, die Unity-AR-App, der Connector, Partnersysteme.

## Die Grundregel

**Felder kommen hinzu. Kein Feld wird umbenannt, keines entfernt, keines
ändert seine Bedeutung.**

Der Grund ist nicht Ordnungsliebe. Die Unity-App deserialisiert mit
`JsonUtility`, und das meldet ein unbekanntes Feld nicht als Fehler — es
liefert stumm `null` beziehungsweise `0`. Ein umbenanntes Feld äußert sich
dort als Rissbreite `0,00 mm` auf dem Tablet eines Prüfers, nicht als
Absturz. Das war im Altsystem die teuerste Fehlerklasse, und genau sie soll
hier nicht wiederkommen.

Bindend ist der Vertrag **auf dem Draht**. Quelltext, Klassennamen,
Modulgrenzen und interne Typen dürfen sich frei ändern.

## Vier Formregeln

Sie folgen alle aus `JsonUtility`:

1. **Keine Wörterbücher, keine Aufzählungen, keine Polymorphie.** Was eine
   Aufzählung sein könnte, ist eine Zeichenkette. Die gültigen Werte stehen
   unten — ein Aufrufer, der einen unbekannten Wert bekommt, darf davon nicht
   umfallen, sondern behandelt ihn wie `unknown`.
2. **Objektlisten im Umschlag** `{ "items": [...] }`. `JsonUtility` kann
   keine Liste auf oberster Ebene. Zahlenreihen (`path_yx`, `width_px`)
   bleiben nackte Felder — `float[]` kann es, und der Umschlag wäre dort nur
   Ballast.
3. **Kein `null`.** Unbekannte Zahl = `-1`, unbekannter Text = `""`.
   `JsonUtility` kennt kein `float?`.
4. **Der Verlauf ist flach:** `path_yx = [y0, x0, y1, x1, …]`, Länge
   `2 * point_count`. Ein Pfad mit 2.000 Stützstellen als Objektliste wäre
   rund dreimal so groß — und genau diese Pfade gehen über Mobilfunk ins
   Tablet.

## Koordinaten

- Pixel des **Originalbildes**, auch wenn der Dienst intern verkleinert
  gerechnet hat.
- Reihenfolge **`(y, x)`** — Zeile zuerst. Wer `(x, y)` erwartet, zeichnet
  gespiegelt.
- Ursprung links oben, y nach unten.
- EXIF-Drehung ist bereits angewandt. Die gemeldete `image.width` /
  `image.height` ist die **aufrechte** Größe, nicht die im Datenstrom.
- Winkel in Grad: 0 = waagerecht, positiv gegen den Uhrzeigersinn, Bereich
  `[-90, 90)`.

## Endpunkte

| Methode | Pfad | Zweck |
|---|---|---|
| `POST` | `/api/v1/detect/crack` | Risse erkennen und vermessen |
| `POST` | `/api/v1/preview/crack` | dasselbe, als gezeichnetes PNG |
| `POST` | `/api/v1/detect/bolt` | geplant — antwortet `501` |
| `POST` | `/api/v1/detect/corrosion` | geplant — antwortet `501` |
| `GET` | `/health/live` | Prozess lebt |
| `GET` | `/health/ready` | bereit für Verkehr |
| `GET` | `/health` | Zustand mit Modellangabe |
| `GET` | `/docs`, `/openapi.json` | Beschreibung |

### Anfrage

`multipart/form-data`. Ein Pflichtfeld, der Rest optional.

| Feld | Typ | Vorgabe | Bedeutung |
|---|---|---|---|
| `image` | Datei | — | JPEG oder PNG |
| `mm_per_px` | float | `-1` | Bekannter Maßstab |
| `marker_size_mm` | float | `-1` | Kantenlänge des ArUco-Markers im Bild |
| `marker_id` | int | `-1` | Kennung; `-1` = größter Marker |
| `marker_dictionary` | string | `""` | Vorgabe `DICT_4X4_50` |
| `depth_mm` | float | `-1` | LiDAR: Abstand zur Fläche |
| `focal_px` | float | `-1` | LiDAR: Brennweite **in Pixeln des gesendeten Bildes** |
| `tilt_deg` | float | `-1` | LiDAR: Schräglage der Fläche |
| `threshold` | float | `-1` | Maskenschwelle |
| `min_area_px` | int | `-1` | Kleinster Befund |
| `max_instances` | int | `-1` | Obergrenze der Befunde |
| `width_method` | string | `""` | `perpendicular` (Vorgabe) oder `distance_transform` |
| `include_paths` | bool | `true` | `false` = nur Kennzahlen, kein Verlauf |

Kopfzeile `X-Correlation-Id` wird übernommen und zurückgegeben; fehlt sie,
vergibt der Dienst eine.

### Antwort

```json
{
  "contract_version": "1.1",
  "task": "crack",
  "request_id": "5f3c…",
  "image":  { "width": 4032, "height": 3024, "exif_rotated": false },
  "model":  { "name": "crack_unet_r18", "kind": "onnx", "trained": true },
  "scale":  { "source": "aruco", "known": true, "mm_per_px": 0.0812,
              "confidence": 0.97, "note": "Marker 7, Kante 739,2 px = 60 mm." },
  "summary": { "instance_count": 2, "severity": "moderate",
               "max_width_px": 3.41, "max_width_mm": 0.2769,
               "total_length_px": 1820.5, "total_length_mm": 147.826 },
  "instances": { "items": [ { … } ] },
  "warnings":  { "items": [ "…" ] },
  "duration_ms": 412.7
}
```

#### `instances.items[]`

| Feld | Typ | Bedeutung |
|---|---|---|
| `instance_id` | int | 0-basiert, **nach Breite absteigend** |
| `label` | string | `crack` (später `bolt`, `corrosion`) |
| `score` | float | Mittlere Wahrscheinlichkeit über der Maske |
| `bbox_yxyx` | float[4] | `[y0, x0, y1, x1]` |
| `pattern` | string | `linear` \| `branched` \| `map` \| `unknown` |
| `orientation_deg` | float | Hauptrichtung des längsten Astes |
| `orientation_class` | string | `horizontal` \| `vertical` \| `diagonal` \| `unknown` |
| `tortuosity` | float | Weglänge / Luftlinie. `1,0` = gerade |
| `touches_border` | bool | **1.1** — der Riss läuft aus dem Bild heraus. `length` und `bbox` sind dann Untergrenzen, keine Messwerte |
| `width_samples_excluded` | int | **1.1** — so viele Stützstellen lagen an einer Verzweigung und zählen nicht in die Breitenstatistik |
| `branch_count` | int | Verzweigungsknoten |
| `severity` | string | `hairline` \| `fine` \| `moderate` \| `wide` \| `severe` \| `unknown` |
| `width_max_px`, `width_mean_px`, `width_p95_px` | float | Breite in Pixeln |
| `width_max_mm`, `width_mean_mm`, `width_p95_mm` | float | in Millimetern, `-1` ohne Maßstab |
| `length_px`, `length_mm` | float | Summe über alle Äste |
| `area_px` | float | Maskenfläche |
| `paths` | Umschlag | die Äste |

#### `paths.items[]`

| Feld | Typ | Bedeutung |
|---|---|---|
| `point_count` | int | Zahl der Stützstellen `N` |
| `path_yx` | float[2N] | `[y0, x0, y1, x1, …]` |
| `width_px` | float[N] | Breite **an jeder Stützstelle**, quer zum Verlauf |
| `width_mm` | float[N] | dieselbe in Millimetern, `-1` wo unbekannt |
| `width_at_junction` | bool[N] | **1.1** — `true`, wo die Stelle an einer Verzweigung liegt: der Wert steht da, gilt aber nicht |
| `length_px`, `length_mm` | float | Länge dieses Astes |
| `is_loop` | bool | Anfang = Ende |

**Ein Riss hat mehrere Äste, wenn er sich verzweigt.** Ein unverzweigter Riss
hat genau einen. Wer nur eine Linie braucht, nimmt den längsten.

## Die Zeichenketten-Werte

Aufrufer behandeln jeden **unbekannten** Wert wie `unknown` und fallen nicht
um. Neue Werte sind ausdrücklich erlaubt — das ist der Preis dafür, dass es
keine Aufzählungen gibt.

- `scale.source`: `none` · `explicit` · `aruco` · `lidar`
- `model.kind`: `onnx` · `classic`
- `pattern`: `linear` · `branched` · `map` · `unknown`
- `orientation_class`: `horizontal` · `vertical` · `diagonal` · `unknown`
- `severity`: `hairline` (< 0,10 mm) · `fine` (< 0,20) · `moderate` (< 0,30) ·
  `wide` (< 0,50) · `severe` (≥ 0,50) · `unknown` (kein Maßstab) ·
  `none` (kein Befund)

**`severity` ist eine Beschreibung, keine Bewertung.** Die zulässige
Rissbreite hängt von der Expositionsklasse ab — DIN EN 1992-1-1/NA nennt
0,2 / 0,3 / 0,4 mm. Welche gilt, weiß dieser Dienst nicht. Die Bänder stehen
in `config.py` (`SEVERITY_BANDS_MM`) und sind änderbar, ohne den Vertrag zu
brechen.

## Welcher Breitenwert gilt

Drei Zahlen stehen je Befund, und sie sind unterschiedlich belastbar.

**`width_p95_mm` ist die brauchbarste.** Sie nimmt die breiteste Stelle
mit, ohne von einem einzelnen Ausreißer bestimmt zu werden.

**`width_max_mm` ist die empfindlichste.** Sie ist ein einzelner Messwert
aus hunderten, und der breiteste Punkt ist genau der, an dem die
Segmentierung am ehesten danebenliegt. Seit 1.1 sind die Verzweigungen
heraus, aber ein Rest bleibt. **Und sie bestimmt `severity`** — wer eine
Einstufung speichert, speichert damit die empfindlichste Zahl.

**`width_mean_mm`** ist der Mittelwert über den ganzen Riss. Er ist stabil,
beantwortet aber nicht die Frage nach der kritischen Stelle.

### Verzweigungen

An einer Kreuzung misst **jedes** Verfahren zu breit, aus einem
geometrischen Grund: in eine Gabelung passt ein größerer Kreis als in den
Riss, und ein Lot quer zum einen Ast schneidet den anderen. Der Effekt
reicht etwa eine Rissbreite weit.

Seit 1.1 zählen diese Stellen nicht mehr in `width_max`, `width_mean` und
`width_p95`. Sie stehen weiter in `width_px` / `width_mm` — wer sie
zeichnen will, kann das —, sind aber in `width_at_junction` markiert, und
`width_samples_excluded` sagt, wie viele es waren.

An einem gemessenen Beispiel: bei einem verzweigten Riss mit 167
Stützstellen lagen 10 an einer Kreuzung; der Höchstwert über alle war
1,40 mm, ohne sie 1,25 mm. **Elf Prozent**, immer in dieselbe Richtung.

### Abtastung

Die Breite wird an jeder Stützstelle gemessen, und die Stützstellen liegen
in **festem Abstand** entlang des Verlaufs (Vorgabe 3 px, einstellbar über
`SADAVISION_PATH_STEP_PX`). Das ist Absicht: eine Vereinfachung nach Form
lässt auf einem geraden Stück zwei Punkte über vierzig Pixel stehen, und
die breiteste Stelle dazwischen sieht dann niemand.

Praktisch heißt das: `point_count` wächst mit der Länge, nicht mit der
Kurvigkeit. Ein Riss von 500 px hat rund 167 Stützstellen. Wer nur die
Linie zeichnen will und die Datenmenge drücken muss, nimmt
`include_paths=false` und die Kennzahlen.

## Zwei Felder, die man nicht überlesen darf

**`model.trained`.** Ist es `false`, läuft der Notbehelf ohne gelernte
Gewichte — ein Ridge-Filter, der jede Fuge, jedes Kabel und jeden Schattenriss
mitnimmt. Die Geometrie ist trotzdem sauber gerechnet, die Erkennung nicht
belastbar. **Ein Aufrufer, der Befunde speichert, muss diesen Zustand
sichtbar machen oder die Annahme verweigern.**

**`scale.known`.** Ist es `false`, sind alle `*_mm` gleich `-1`. Dann gibt es
keine Rissbreite in Millimetern, und `severity` ist `unknown`.

**`touches_border`** (seit 1.1). Ist es `true`, läuft der Riss aus dem Bild
heraus. `length_mm` und `bbox_yxyx` messen dann die Bildkante mit, nicht den
Riss — es sind Untergrenzen. Wer sie als Länge protokolliert, protokolliert
den Bildausschnitt.

## Fehler

Einheitliche Form, englisch formuliert (sie stehen in der Maske):

```json
{ "error": "unprocessable_entity", "detail": "Format nicht erkannt.", "request_id": "…" }
```

| Code | `error` | Wann |
|---|---|---|
| 404 | `not_found` | unbekannte Aufgabe |
| 422 | `unprocessable_entity` | Bild unlesbar, zu groß, zu klein |
| 422 | `validation_failed` | Feld fehlt oder hat den falschen Typ |
| 500 | `internal_error` | Fehler bei uns — Stapelspur nur im Protokoll |
| 501 | `not_implemented` | `bolt`, `corrosion` — geplant, nicht trainiert |

## Beispiel: Unity

```csharp
[Serializable] public class PathDto {
    public int point_count;
    public float[] path_yx;      // [y0,x0,y1,x1,…]
    public float[] width_px;
    public float[] width_mm;
    public bool[]  width_at_junction;   // seit 1.1 - dort gilt die Breite nicht
    public float length_px;
    public float length_mm;
    public bool  is_loop;
}
[Serializable] public class PathListDto { public PathDto[] items; }

[Serializable] public class InstanceDto {
    public int    instance_id;
    public string label;
    public float  score;
    public float[] bbox_yxyx;
    public string pattern;
    public float  orientation_deg;
    public string orientation_class;
    public float  tortuosity;
    public int    branch_count;
    public bool   touches_border;          // seit 1.1
    public int    width_samples_excluded;  // seit 1.1
    public string severity;
    public float  width_max_mm;
    public float  length_mm;
    public PathListDto paths;
}
[Serializable] public class InstanceListDto { public InstanceDto[] items; }

[Serializable] public class DetectionResponseDto {
    public string contract_version;
    public string task;
    public ImageInfoDto image;
    public ModelInfoDto model;
    public ScaleDto scale;
    public SummaryDto summary;
    public InstanceListDto instances;
    public WarningListDto warnings;
}

// Verlauf auslesen - Reihenfolge ist (y, x)!
for (int i = 0; i < path.point_count; i++) {
    float y = path.path_yx[2 * i];
    float x = path.path_yx[2 * i + 1];
    float w = path.width_mm[i];   // -1 = unbekannt

    // An einer Verzweigung steht ein Wert, aber er gilt nicht:
    // dort misst jedes Verfahren zu breit. Zeichnen ja, messen nein.
    bool gilt = !path.width_at_junction[i];
}
```

## Versionierung

`contract_version` steht in jeder Antwort und in der Kopfzeile
`X-Contract-Version`. Sie ändert sich in der **Nebenstelle**, wenn Felder
hinzukommen (`1.0` → `1.1`), und in der **Hauptstelle** nur, wenn der Vertrag
bräche — was nicht vorgesehen ist. Ein Aufrufer, der eine höhere Nebenstelle
sieht, arbeitet unverändert weiter.
