# Vom Bildpfad zum Rissverlauf auf dem Bauteil

> Entwurf, 24.09.2026 · noch nichts davon gebaut · vor Arbeit an der
> Unity-Seite lesen

Ziel: Der Riss, den der Dienst im Foto gefunden hat, liegt hinterher als
3D-Linie **auf der Oberfläche des Bauteils** im AR-Modell — sichtbar aus jeder
Richtung, wiederfindbar beim nächsten Besuch, gespeichert im
Bauteilkoordinatensystem.

**Nur der Verlauf über die Oberfläche, nicht in die Tiefe.** Wie tief ein Riss
ins Bauteil reicht, sieht kein LiDAR und keine Kamera; das wäre Ultraschall
oder Bohrkern. Was hier entsteht, ist eine Kurve auf einer Fläche, mit der
Rissbreite als Eigenschaft je Punkt.

## Die Arbeitsteilung

**Die Rückprojektion läuft auf dem Gerät, nicht im Dienst.** Das Netz aus dem
LiDAR-Scan liegt im iPad, ändert sich laufend und ist groß. Es zum Server zu
schicken, wäre mehrere Megabyte je Aufnahme für ein Ergebnis, das die App in
Millisekunden lokal hat.

```
  iPad                                      SADA Vision
  ────                                      ───────────
  Foto + Intrinsik + Pose + Tiefe
        │
        ├── Foto, focal_px, depth_mm ─────────►  Segmentierung
        │                                        Skelett, Breite
        ◄── path_yx (Pixel), width_mm ─────────  Maßstab
        │
        ├─ Strahl je Punkt aus der Intrinsik
        ├─ Treffer gegen das LiDAR-Netz
        ├─ Glättung auf der Fläche
        └─ Ablage im Bauteilkoordinatensystem ──►  TraceForm
```

Der Dienst liefert also weiterhin nur, was er kann: **wo im Bild** der Riss
liegt und **wie breit** er ist. Alles Räumliche bleibt in der App.

## Schritt für Schritt

### 1 · Aufnehmen

Aus einem ARKit-Bild mitnehmen:

| Größe | Quelle | Wofür |
|---|---|---|
| Foto | `ARFrame.capturedImage` | geht an den Dienst |
| Intrinsik `fx, fy, cx, cy` | `ARCamera.intrinsics` | Strahl je Pixel |
| Pose (Kamera → Welt) | `ARCamera.transform` | Strahlursprung und -richtung |
| Tiefe | `ARFrame.sceneDepth` | Maßstab, Rückfallebene für Treffer |
| Netz | `ARMeshAnchor` / `ARMeshManager` | die Fläche, auf die getroffen wird |

**Der häufigste Fehler steckt schon hier:** Das Foto, das an den Dienst geht,
ist fast nie so groß wie `capturedImage`. Wer es auf 1600 Pixel verkleinert,
muss die Intrinsik mitskalieren:

```
faktor  = breite_gesendet / breite_original
fx' = fx * faktor    cx' = cx * faktor
fy' = fy * faktor    cy' = cy * faktor
```

Genau `fx'` gehört als `focal_px` in die Anfrage. Wird die ungeskalierte
Brennweite geschickt, ist jede Millimeterangabe um denselben Faktor falsch —
und zwar plausibel falsch, also unauffällig.

### 2 · Fragen

```
POST /api/v1/detect/crack
  image       = das verkleinerte Foto
  focal_px    = fx'
  depth_mm    = Tiefe in der Bildmitte, oder besser: am Riss
  tilt_deg    = Winkel zwischen Blickrichtung und Flächennormale
```

`depth_mm` und `tilt_deg` dienen nur dem Maßstab. Liegt ein gedruckter
ArUco-Marker mit im Bild, ist der genauer — dann `marker_size_mm` schicken;
der Dienst nimmt ihn vorrangig und rechnet die Perspektive über eine
Homographie mit.

### 3 · Strahl je Punkt

Für jeden Punkt `(y, x)` aus `path_yx`:

```
Richtung im Kameraraum:
    d_cam = normalize( ( (x - cx') / fx',
                        -(y - cy') / fy',      // y im Bild zeigt nach unten
                        -1 ) )                 // ARKit blickt nach -z

Richtung in der Welt:
    d_welt = rotation( ARCamera.transform ) * d_cam
    o_welt = position( ARCamera.transform )
```

Zwei Fallen:

- **Vorzeichen von y.** Bildkoordinaten laufen nach unten, Kamerakoordinaten
  nach oben. Wer das Minus vergisst, bekommt einen an der Waagerechten
  gespiegelten Riss — und der sieht auf einer glatten Wand *plausibel* aus.
- **Händigkeit.** ARKit ist rechtshändig, Unity linkshändig. Das
  AR-Foundation-Gerüst rechnet das für die Kamerapose um, aber eine von Hand
  gebaute Richtung muss dieselbe Umrechnung durchlaufen. Immer gegen einen
  bekannten Punkt prüfen: der Bildmittelpunkt muss auf das treffen, was in
  der Mitte des Sucherbilds steht.

### 4 · Treffen

Strahl gegen das LiDAR-Netz schneiden — `ARRaycastManager` mit
`TrackableType.Depth | TrackableType.PlaneWithinPolygon`, oder direkt gegen
den `MeshCollider` der `ARMeshManager`-Netze. Ergebnis je Punkt: Trefferpunkt
und Flächennormale.

Was dabei schiefgeht und was dagegen hilft:

| Fall | Was passiert | Umgang |
|---|---|---|
| Kein Treffer (Loch im Netz) | Punkt fehlt | Zwischen den Nachbarn auf der Fläche interpolieren; bei mehr als ~5 fehlenden Punkten hintereinander die Linie trennen statt zu raten |
| Treffer auf dem falschen Objekt | Riss klebt am Geländer davor | Nur Treffer annehmen, deren Entfernung nahe an `depth_mm` liegt (±20 %) |
| Streifende Treffer | Normale steht fast senkrecht zum Strahl | Treffer verwerfen, wenn `dot(d_welt, normale) > -0,25` — dort ist die Lage ohnehin nicht bestimmbar |
| Z-Fighting beim Zeichnen | Linie flimmert in der Fläche | Punkt um 2–3 mm entlang der Normalen abheben |

### 5 · Glätten — auf der Fläche, nicht im Raum

Die rohe Trefferfolge ist verrauscht: das ARKit-Netz hat wenige Zentimeter
Auflösung, und jeder Treffer trägt dessen Dreiecksfehler. Eine gewöhnliche
Glättung im Raum zieht die Linie **von der Fläche weg** — sie schneidet
Ecken ab und schwebt dann über konvexen Kanten.

Deshalb: glätten, dann **jeden Punkt wieder auf das Netz projizieren**
(nächster Punkt auf dem nächstgelegenen Dreieck). Zwei bis drei Durchgänge
mit einem kleinen Fenster reichen.

Danach **in 3D neu abtasten**. Die Punkte kommen gleichmäßig verteilt *im
Bild* an; auf einer schräg stehenden Fläche sind sie es im Raum nicht mehr —
am fernen Ende liegen sie weit auseinander, am nahen dicht. Ein gleichmäßiger
Abstand von etwa 5 mm auf der Fläche ist eine brauchbare Vorgabe. Die
Rissbreite wird dabei mitinterpoliert.

### 6 · Breite mitführen, nicht aus dem Netz holen

`width_mm[i]` gehört zum Punkt `i` und wandert unverändert mit. Aus dem
LiDAR-Netz lässt sich eine Rissbreite **nicht** gewinnen: das Netz löst
Zentimeter auf, der Riss ist Zehntelmillimeter breit. Wer es versucht, misst
die Dreiecksgröße des Scanners.

Zum Zeichnen: Band (`LineRenderer` mit variabler Breite) oder Einfärbung nach
Band. Eine Darstellung in **wahrer** Breite ist auf dem Bildschirm unsichtbar
— 0,3 mm aus zwei Metern. Sinnvoll ist eine überhöhte, aber monotone
Darstellung plus Zahlenwert an der breitesten Stelle.

### 7 · Verankern

Ein Weltkoordinatensystem von ARKit überlebt die Sitzung nicht. Der Verlauf
muss deshalb **im Bauteilkoordinatensystem** abgelegt werden — in demselben,
in dem TraceForm das CAD-Modell führt, über die Einpassung, die die AR-App
für die Anzeige ohnehin schon hat:

```
p_bauteil = inverse( T_welt_bauteil ) * p_welt
```

Erst diese Fassung geht an TraceForm. Dann liegt der Riss beim nächsten
Besuch wieder da, wo er ist, auch nach Neustart, auch von einem anderen
Gerät, auch im Web-Viewer ohne AR.

## Was in TraceForm ankommt

Vorschlag für die Erweiterung des Vertrags — **additiv**, ein neues Feld neben
`path_yx`, kein Ersatz:

```json
"path_xyz": [x0, y0, z0, x1, y1, z1, …],
"path_normal_xyz": [nx0, ny0, nz0, …],
"path_space": "component",
"path_frame_id": "design-4711/part-88",
"projection_quality": 0.87,
"projection_gaps": 2
```

Dieselben vier Formregeln wie im übrigen Vertrag: flaches Zahlenfeld, keine
Wörterbücher, kein `null`. `projection_quality` ist der Anteil der Punkte mit
echtem Treffer — darunter steht eine interpolierte Vermutung, und das gehört
in den Befund.

Erzeugt wird das in der App, nicht im Dienst.

## Offene Punkte

- **Mehrere Aufnahmen desselben Risses.** Zwei Fotos aus zwei Richtungen
  ergeben zwei Linien, die sich um Zentimeter unterscheiden können. Ob sie
  zusammengeführt oder als zwei Befunde geführt werden, ist eine fachliche
  Entscheidung für TraceForm, keine technische.
- **Verlaufsende an Bauteilkanten.** Läuft ein Riss über eine Kante, trifft
  der Strahl dahinter ins Leere oder auf die Nachbarfläche. Ob die Linie dort
  endet oder weiterläuft, muss die App entscheiden — der Dienst sieht im Foto
  nur eine durchgehende Linie.
- **Genauigkeit.** Wie gut die Lage tatsächlich wird, hängt am ARKit-Netz und
  an der Einpassung, nicht am Dienst. Vor dem ersten Feldeinsatz gegen ein
  Bauteil mit eingemessenen Punkten prüfen — sonst ist die Zahl im Protokoll
  eine Hoffnung.
