"""Einstellungen des Dienstes.

Alles kommt aus Umgebungsvariablen mit dem Praefix ``SADAVISION_``.
Keine Geheimnisse im Repo, keine Pfade fest verdrahtet.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SADAVISION_",
        env_file=".env",
        extra="ignore",
    )

    # --- Dienst -----------------------------------------------------------
    service_name: str = "sada-vision"
    log_level: str = "INFO"
    log_json: bool = True

    # --- Modelle ----------------------------------------------------------
    model_dir: Path = Path("models")
    # Leer = kein ONNX vorhanden, der Dienst faellt auf das klassische
    # Ridge-Verfahren zurueck und meldet das in jeder Antwort.
    crack_model: str = "crack_unet_r18.onnx"
    onnx_threads: int = 0  # 0 = onnxruntime entscheidet

    # --- Bildverarbeitung -------------------------------------------------
    # Bombenschutz beim Dekodieren. 100 Megapixel: eine Zip-Bombe hat
    # Gigapixel, eine echte Kamera hoechstens ein paar Dutzend Megapixel -
    # und 48 sind bei heutigen Telefonen normal. Bei 40 waere ein ganz
    # gewoehnliches Handyfoto abgelehnt worden.
    max_image_pixels: int = 100_000_000
    # Groesste Kante, auf die ein Bild vor der Segmentierung verkleinert wird.
    # 0 = nie verkleinern. Die Geometrie wird anschliessend zurueckgerechnet.
    max_edge_px: int = 4096
    tile_size: int = 512
    tile_overlap: int = 64

    # --- Nachbearbeitung --------------------------------------------------
    mask_threshold: float = 0.5
    min_component_area_px: int = 40
    min_branch_length_px: float = 12.0
    rdp_epsilon_px: float = 1.0
    # Abstand der Stuetzstellen entlang des Verlaufs. Gleichmaessig statt
    # nach Form vereinfacht - sonst wird die Breite nur dort gemessen, wo
    # die Linie zufaellig knickt.
    path_step_px: float = 3.0
    # Wie weit die Verzerrung um eine Verzweigung reicht, als Vielfaches der
    # dort gemessenen Breite.
    junction_radius_factor: float = 1.5
    width_method: str = "perpendicular"  # "perpendicular" | "distance_transform"

    # --- Grenzen ----------------------------------------------------------
    max_upload_bytes: int = 30 * 1024 * 1024
    # Mehr gleichzeitige Rechnungen als Kerne machen alles langsamer, nicht
    # schneller - jede Bibliothek nimmt sich ohnehin schon mehrere Kerne.
    max_concurrent_analyses: int = 2
    max_instances: int = 200
    max_points_per_path: int = 2000

    def model_path(self, filename: str) -> Path:
        return self.model_dir / filename


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Nur fuer Tests."""
    global _settings
    _settings = None


# Schwellen der Rissbreite in Millimetern.
#
# Bewusst als Tabelle und nicht im Code verstreut: die zulaessige Rissbreite
# haengt von der Expositionsklasse ab (DIN EN 1992-1-1/NA nennt 0,2 / 0,3 /
# 0,4 mm). Was hier steht, ist eine Beschreibung des Befundes, keine Bewertung
# der Zulaessigkeit - die trifft TraceForm mit dem Wissen um das Bauteil.
SEVERITY_BANDS_MM: tuple[tuple[str, float], ...] = (
    ("hairline", 0.10),
    ("fine", 0.20),
    ("moderate", 0.30),
    ("wide", 0.50),
    ("severe", float("inf")),
)
