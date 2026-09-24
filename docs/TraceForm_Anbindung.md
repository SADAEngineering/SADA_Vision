# Anbindung an TraceForm

> Entwurf, 24.09.2026 · noch nichts davon gebaut

## Die Grenze

SADA Vision kennt **kein** TraceForm. Kein Mandant, kein Bauteil, kein
Auftrag, keine Anmeldung. Es bekommt ein Bild und gibt Geometrie zurück.

Das ist dieselbe Einbahnstraße wie zwischen TraceForm und der Plattform, nur
eine Ebene tiefer: **TraceForm kennt Vision, Vision kennt TraceForm nie.**
Kein Produktname, kein Fachbegriff, kein Schlüssel aus TraceForm wandert in
dieses Repo. Was der Dienst über den Aufruf wissen muss, bekommt er als
Formularfeld — Maßstab und Feineinstellung, sonst nichts.

Der Preis dafür ist eine Regel, die nicht verhandelbar ist:

> **Der Vision-Container wird nie nach außen veröffentlicht.**
> Er hat keine Anmeldung, keine Mandantentrennung und keine Begrenzung je
> Aufrufer. Er ist im Compose-Netz erreichbar und sonst nirgends. Wer den
> Port aufmacht, hat einen offenen Bildverarbeiter im Internet.

Die Anmeldung macht TraceForm, bevor es fragt.

## Wo der Aufruf hingehört

**In den Worker, nicht in die Anfrage des Anwenders.** Eine Rissanalyse
braucht auf einem 12-Megapixel-Foto Sekunden. Wer das synchron in eine
Blazor-Seite hängt, hat eine Maske, die blockiert, und einen Zeitablauf am
Reverse Proxy.

Der Weg über die Job-Queue der Plattform:

```
Anwender lädt Foto hoch
   └─ Datei in den Objektspeicher (MinIO), Anhang am Item
   └─ Job einstellen:  befund.riss   { attachmentId, mmPerPx?, markerSizeMm? }
                                    │
Worker nimmt den Job                ▼
   ├─ Datei aus dem Objektspeicher holen
   ├─ POST http://vision:8080/api/v1/detect/crack
   ├─ Antwort prüfen:  model.trained?  scale.known?
   └─ Befund am Item speichern, Job abschließen
                                    │
Maske zeigt den Befund              ▼   (SignalR oder Abruf)
```

Job-Art nach der TraceForm-Regel `bereich.aktion`: **`befund.riss`**, später
`befund.schraube`, `befund.korrosion`.

## Was in TraceForm gespeichert wird

Nicht die ganze Antwort. Der Verlauf eines verzweigten Risses kann tausende
Stützstellen haben; die gehören nicht in eine Spalte, über die sortiert wird.

**In die Datenbank:** die Kennzahlen je Befund — `width_max_mm`,
`width_mean_mm`, `length_mm`, `pattern`, `orientation_class`, `severity`,
`score`, dazu `model_name`, `model_trained`, `scale_source`,
`scale_mm_per_px` und der Zeitpunkt. Danach wird gefiltert, sortiert und
ausgewertet.

**In den Objektspeicher:** die vollständige Antwort als JSON, neben dem Foto,
unter demselben Anhang. Von dort holt sie der Viewer, wenn jemand den Verlauf
sehen will.

**Zwei Felder müssen mitgespeichert werden, auch wenn sie unbequem sind:**

- `model_trained` — ein Befund aus dem Notbehelf ist nicht dasselbe wie einer
  aus dem trainierten Netz, und in einem halben Jahr weiß das niemand mehr.
- `model_name` — wenn das Modell ausgetauscht wird, ändern sich Zahlen. Ohne
  diese Spalte lässt sich später nicht erklären, warum derselbe Riss im März
  0,28 mm hatte und im Juni 0,33 mm.

Das ist derselbe Gedanke wie TraceForm-Regel 5: der Zustand gehört der
Instanz. Ein Befund ist eine **Messung zu einem Zeitpunkt mit einem
bestimmten Werkzeug**, nicht eine Eigenschaft des Bauteils.

## DTOs

Die Vision-DTOs werden nach `TraceForm.Contracts` gespiegelt — wie die
Unity-Verträge auch. Sie sind dort **nicht** die Entitäten. Vorschlag für den
Ort: `src/TraceForm.Contracts/Findings/`.

Weil beide Verträge dieselben vier Formregeln einhalten (keine Wörterbücher,
keine Aufzählungen, kein `null`, Listen im Umschlag), lässt sich die Antwort
von Vision fast unverändert an Unity weiterreichen. Das ist Absicht und der
Grund, warum dieser Dienst sich die `JsonUtility`-Beschränkungen auferlegt,
obwohl er selbst nie mit Unity spricht.

## Der Dienstaufruf

Nach TraceForm-Konvention eine Service-Klasse mit `C_`-Präfix in
`TraceForm.Infrastructure`:

```
C_VisionClient : IVisionClient
    Task<CrackFindingDto> DetectCrackAsync(
        Stream image, VisionScaleOptions scale, CancellationToken ct)
```

```csharp
public sealed class C_VisionClient : IVisionClient
{
    private readonly HttpClient _http;

    public async Task<CrackFindingDto> DetectCrackAsync(
        Stream image, string fileName, VisionScaleOptions scale, CancellationToken ct)
    {
        using var form = new MultipartFormDataContent();

        var file = new StreamContent(image);
        file.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
        form.Add(file, "image", fileName);

        // Millimeter nur mitschicken, wenn sie bekannt sind. Der Dienst
        // raet nicht - und wir raten ihm auch nichts vor.
        if (scale.MarkerSizeMm is { } marker)
            form.Add(new StringContent(
                marker.ToString(CultureInfo.InvariantCulture)), "marker_size_mm");
        else if (scale.MmPerPx is { } mmPerPx)
            form.Add(new StringContent(
                mmPerPx.ToString(CultureInfo.InvariantCulture)), "mm_per_px");

        using var response = await _http.PostAsync("/api/v1/detect/crack", form, ct);

        // 501 heisst "geplant, noch nicht trainiert" - das gehoert in die
        // Maske als "noch nicht verfuegbar", nicht ins Fehlerprotokoll.
        if (response.StatusCode == HttpStatusCode.NotImplemented)
            throw new VisionTaskNotReadyException("crack");

        response.EnsureSuccessStatusCode();
        var dto = await response.Content.ReadFromJsonAsync<CrackFindingDto>(
            cancellationToken: ct)
            ?? throw new InvalidOperationException("Empty response from vision.");

        // Ein Befund aus dem Notbehelf ist kein Befund. Er wird gespeichert,
        // aber markiert - in einem halben Jahr weiss das sonst niemand mehr.
        if (!dto.Model.Trained)
            _log.LogWarning(
                "Vision lief ohne trainiertes Modell ({Model}) - Befund unsicher.",
                dto.Model.Name);

        return dto;
    }
}
```

`CultureInfo.InvariantCulture` ist kein Zierrat: unter deutschem Gebietsschema
wird aus `0.08` sonst `0,08`, und der Dienst liest das als Formularfeld, das
er nicht deuten kann — die Anfrage scheitert mit `422`, und zwar nur auf
Rechnern mit deutschem Gebietsschema.

Zu beachten:

- Kopfzeile `X-SADA-Client: traceform-worker/<version>` mitschicken, und
  `X-Correlation-Id` aus dem Job durchreichen — dann lässt sich ein Befund
  über beide Protokolle hinweg verfolgen.
- Adresse aus der Konfiguration (`Vision:BaseUrl`), nie fest verdrahtet.
  Fehlt sie, ist die Funktion abgeschaltet und die Schaltfläche grau — nicht
  ein Fehler beim Klicken.
- Zeitablauf großzügig (60 s) und **ein** Wiederholungsversuch. Ein zweiter
  bringt bei einem überlasteten Bildverarbeiter nichts außer mehr Last.
- `501` ist kein Fehler, sondern die Antwort für `bolt` und `corrosion`,
  solange sie nicht trainiert sind. Das gehört in die Maske als „noch nicht
  verfügbar", nicht ins Fehlerprotokoll.

## Betrieb

Der Dienst kommt als weiterer Eintrag in `compose.onprem.yml` und
`compose.dev.yml` — **ohne** `ports:`, nur im internen Netz:

```yaml
vision:
  image: ghcr.io/sadaengineering/sada-vision:${VISION_TAG:-latest}
  restart: unless-stopped
  environment:
    SADAVISION_MODEL_DIR: /models
    SADAVISION_LOG_JSON: "true"
    OMP_NUM_THREADS: "4"
  volumes:
    - vision-models:/models:ro
  healthcheck:
    test: ["CMD", "curl", "-fsS", "http://127.0.0.1:8080/health/live"]
    interval: 30s
  # kein ports: - bewusst.
```

Das passt zu TraceForm-Regel 7: **ein Artefakt für SaaS und On-Prem**, kein
ausgehender Internetzugang für den Kernbetrieb. Der Vision-Container
telefoniert nirgendwohin; das Modell liegt im Volume und kommt beim Update
mit dem Image oder über die Registry.

**Speicher und Kerne.** Das Modell belegt rund 100 MB, ein Lauf auf einem
12-Megapixel-Foto einige hundert MB Spitze. Eine Begrenzung auf 2 GB und
zwei bis vier Kerne ist ein guter Anfang; `OMP_NUM_THREADS` muss dazu passen,
sonst reißt sich jede Bibliothek alle Kerne und sie behindern sich
gegenseitig.

## Lizenzierung

Noch offen: eigener Modul-Schlüssel `traceform.vision` oder mitlaufend unter
`traceform.reliability`. Dafür spricht ein eigener Schlüssel — die Erkennung
ist ein abgrenzbarer Wert und lässt sich getrennt verkaufen. Dagegen
spricht, dass ein weiterer Schlüssel die Matrix aus vier auf fünf verbreitert.
**Produktfrage, keine technische.** Bis sie entschieden ist, läuft der Dienst
ohne Schlüsselprüfung — er kennt das Lizenzwesen ohnehin nicht, die Prüfung
säße in TraceForm vor dem Job.
