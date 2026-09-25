"""Die DTOs auf dem Draht.

**Der Vertrag ist bindend und nur additiv** - Felder hinzufuegen ja, umbenennen
oder entfernen nein. Grund ist derselbe wie in TraceForm: Unity deserialisiert
mit ``JsonUtility``, und ein umbenanntes Feld liefert dort stumm ``null``
statt eines Fehlers. Das war im Altsystem die teuerste Fehlerklasse.

Daraus folgen vier Formregeln:

1. **Keine Woerterbuecher, keine Aufzaehlungen, keine Polymorphie.**
   Alles, was eine Aufzaehlung sein koennte, ist eine Zeichenkette. Die
   gueltigen Werte stehen in ``docs/API_Vertrag.md``; ein unbekannter Wert
   darf den Aufrufer nicht umwerfen.
2. **Listen von Objekten im Umschlag** ``{ "items": [...] }``. Zahlenreihen
   (``path_yx``, ``width_px``) bleiben nackte Felder - dort ist der Umschlag
   nur Ballast, und ``float[]`` kann JsonUtility.
3. **Kein ``null``.** Unbekannte Zahlen sind ``-1``, unbekannte Texte ``""``.
   ``JsonUtility`` kennt kein ``float?``.
4. **Der Verlauf ist flach:** ``path_yx = [y0, x0, y1, x1, ...]`` mit
   ``point_count`` daneben. Ein Pfad mit 2.000 Stuetzstellen als Objektliste
   waere rund dreimal so gross - und genau diese Pfade gehen ueber Mobilfunk
   ins Tablet.

Koordinaten sind Pixel des Originalbildes, Ursprung links oben, ``(y, x)``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Dto(BaseModel):
    """Gemeinsame Basis. ``protected_namespaces`` aus, weil ``model_name``
    und ``model_kind`` zum Vertrag gehoeren und pydantic sonst warnt."""

    model_config = ConfigDict(protected_namespaces=())


class ImageInfoDto(_Dto):
    width: int
    height: int
    exif_rotated: bool = False


class ModelInfoDto(_Dto):
    name: str
    kind: str = Field(description='"onnx" oder "classic"')
    trained: bool = Field(
        description="false = Notbehelf ohne gelernte Gewichte, Befund nicht belastbar"
    )


class ScaleDto(_Dto):
    source: str = Field(description='"none" | "explicit" | "aruco" | "lidar"')
    known: bool
    mm_per_px: float = Field(description="-1, wenn unbekannt")
    confidence: float
    note: str = ""


class PathDto(_Dto):
    """Ein Ast des Risses."""

    point_count: int
    path_yx: list[float] = Field(
        description="Flach: [y0, x0, y1, x1, ...], Laenge 2 * point_count"
    )
    width_px: list[float] = Field(description="Breite quer zum Verlauf, je Stuetzstelle")
    width_mm: list[float] = Field(description="-1 je Stelle, wenn kein Massstab")
    width_at_junction: list[bool] = Field(
        default_factory=list,
        description=(
            "Seit 1.1. True, wo die Stuetzstelle an einer Verzweigung liegt: "
            "dort misst jedes Verfahren zu breit. Der Wert steht trotzdem da, "
            "zaehlt aber nicht in width_max/mean/p95 des Befundes."
        ),
    )
    length_px: float
    length_mm: float = -1.0
    is_loop: bool = False


class PathListDto(_Dto):
    items: list[PathDto] = Field(default_factory=list)


class InstanceDto(_Dto):
    """Ein zusammenhaengender Befund."""

    instance_id: int
    label: str = "crack"
    score: float = 0.0
    bbox_yxyx: list[float] = Field(description="[y0, x0, y1, x1] im Originalbild")

    pattern: str = Field(description='"linear" | "branched" | "map" | "unknown"')
    orientation_deg: float = Field(
        description="0 = waagerecht, positiv gegen den Uhrzeigersinn, [-90, 90)"
    )
    orientation_class: str = Field(
        description='"horizontal" | "vertical" | "diagonal" | "unknown"'
    )
    tortuosity: float = Field(description="Weglaenge / Luftlinie, 1,0 = gerade")
    branch_count: int = 0
    touches_border: bool = Field(
        default=False,
        description=(
            "Seit 1.1. Der Riss laeuft aus dem Bild heraus. length und bbox "
            "sind dann Untergrenzen, keine Messwerte."
        ),
    )
    width_samples_excluded: int = Field(
        default=0,
        description=(
            "Seit 1.1. So viele Stuetzstellen lagen an einer Verzweigung und "
            "sind nicht in die Breitenstatistik eingegangen."
        ),
    )
    severity: str = Field(
        description='"hairline" | "fine" | "moderate" | "wide" | "severe" | "unknown"'
    )

    width_max_px: float = -1.0
    width_mean_px: float = -1.0
    width_p95_px: float = -1.0
    width_max_mm: float = -1.0
    width_mean_mm: float = -1.0
    width_p95_mm: float = -1.0

    length_px: float = 0.0
    length_mm: float = -1.0
    area_px: float = 0.0

    paths: PathListDto = Field(default_factory=PathListDto)


class InstanceListDto(_Dto):
    items: list[InstanceDto] = Field(default_factory=list)


class SummaryDto(_Dto):
    """Das Wichtigste in einem Blick - fuer Listen und Ampeln."""

    instance_count: int = 0
    severity: str = "unknown"
    max_width_px: float = -1.0
    max_width_mm: float = -1.0
    total_length_px: float = 0.0
    total_length_mm: float = -1.0


class WarningListDto(_Dto):
    items: list[str] = Field(default_factory=list)


class DetectionResponseDto(_Dto):
    contract_version: str
    task: str
    request_id: str
    image: ImageInfoDto
    model: ModelInfoDto
    scale: ScaleDto
    summary: SummaryDto
    instances: InstanceListDto
    warnings: WarningListDto
    duration_ms: float


class ErrorDto(_Dto):
    """Fehler sind englisch formuliert - sie stehen in der Maske."""

    error: str
    detail: str = ""
    request_id: str = ""


class TaskStatusDto(_Dto):
    task: str
    model_name: str
    model_kind: str
    trained: bool
    weights_present: bool
    ready: bool


class TaskStatusListDto(_Dto):
    items: list[TaskStatusDto] = Field(default_factory=list)


class HealthDto(_Dto):
    status: str
    service: str
    version: str
    contract_version: str
    tasks: TaskStatusListDto
