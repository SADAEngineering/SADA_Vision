# samples/

Arbeitsordner für Probebilder. **Der Inhalt steht nicht in Git** (außer dieser
Datei) — Fotos von Bauteilen gehören in den Objektspeicher, nicht ins Repo.

Eine Probe erzeugen und durchschicken:

```powershell
.\tools\vision.ps1 serve
.\tools\vision.ps1 probe samples\foto.jpg marker_size_mm=60
```

Das schreibt `befund.json` und `vorschau.png`.

## Woran man in der Vorschau erkennt, dass etwas nicht stimmt

| Was zu sehen ist | Was los ist |
|---|---|
| Roter Balken **UNTRAINED FALLBACK** oben | Kein Modell geladen — es läuft der Ridge-Filter. Zahlen nicht verwenden. |
| Dutzende kleiner Kreuze über die ganze Fläche | Derselbe Fall: der Filter findet im Betonrauschen überall Kanten. |
| Der Maßstabsmarker ist selbst als Befund markiert | Sollte nicht vorkommen — er wird ausgeblendet. Wenn doch: `marker_size_mm` wurde nicht mitgeschickt, der Marker also gar nicht erkannt. |
| Breiten in Pixeln statt Millimetern | Kein Maßstab. `scale.source` in der JSON sagt, woran es lag. |
| Die Linie sitzt neben dem Riss | Bild zu stark verkleinert (`SADAVISION_MAX_EDGE_PX`) oder außerhalb der Schärfe. |
