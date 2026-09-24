"""Massstab aus einem gedruckten Marker im Bild.

Ein ArUco-Marker bekannter Kantenlaenge neben dem Riss ist der genaueste und
billigste Weg zu Millimetern: ein Blatt Papier, ein Klebestreifen, fertig. Aus
seinen vier Ecken folgt nicht nur eine Zahl, sondern eine **Homographie** -
und damit ein *ortsabhaengiger* Massstab, der die Schraeglage der Aufnahme
mitnimmt. Ein Riss am oberen Bildrand wird dann anders skaliert als einer am
unteren, was bei schraeg gehaltener Kamera genau richtig ist.

Der Marker muss in derselben Ebene liegen wie der Riss. Ein Marker am Boden
vor einer Wand misst den Boden.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..domain import ScaleInfo

# 4x4 mit 50 Kennungen: gross gedruckt auch aus zwei Metern noch erkannt,
# klein genug, dass Verwechslungen praktisch nicht vorkommen.
DEFAULT_DICTIONARY = "DICT_4X4_50"


class ArucoScaleResolver:
    def __init__(self, dictionary: str = DEFAULT_DICTIONARY) -> None:
        self._dictionary_name = dictionary
        dict_id = getattr(cv2.aruco, dictionary, None)
        if dict_id is None:
            raise ValueError(f"Unbekanntes ArUco-Woerterbuch: {dictionary}")
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters()
        # Subpixelverfeinerung der Ecken - ohne sie ist der Massstab auf ganze
        # Pixel gerundet, und daran haengt jede Millimeterangabe im Ergebnis.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    def resolve(
        self,
        rgb: np.ndarray,
        marker_mm: float,
        marker_id: int = -1,
    ) -> ScaleInfo:
        if marker_mm is None or marker_mm <= 0:
            return ScaleInfo(
                source="none", note="marker_size_mm muss groesser als 0 sein."
            )

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return ScaleInfo(
                source="none",
                note=(
                    f"Kein Marker aus {self._dictionary_name} gefunden. "
                    "Marker vollstaendig, scharf und unverdeckt ablichten."
                ),
            )

        ids_flat = ids.flatten().tolist()
        chosen = _pick(corners, ids_flat, marker_id)
        if chosen is None:
            return ScaleInfo(
                source="none",
                note=f"Marker {marker_id} nicht im Bild (gefunden: {ids_flat}).",
            )
        quad, found_id = chosen

        sides = [
            float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)
        ]
        side_px = float(np.mean(sides))
        if side_px < 12.0:
            return ScaleInfo(
                source="none",
                note=f"Marker mit {side_px:.0f} px zu klein abgebildet - naeher heran.",
            )

        # Wie quadratisch ist das Viereck noch? Starke Abweichung heisst
        # starke Schraeglage - die Homographie faengt sie auf, aber die
        # Verlaesslichkeit sinkt.
        spread = (max(sides) - min(sides)) / max(side_px, 1e-6)
        confidence = float(np.clip(1.0 - spread, 0.2, 1.0))

        # Marker-Ecken auf ein Quadrat der wahren Kantenlaenge abbilden:
        # die Zielebene ist dann direkt in Millimetern.
        plane = np.array(
            [[0.0, 0.0], [marker_mm, 0.0], [marker_mm, marker_mm], [0.0, marker_mm]],
            dtype=np.float64,
        )
        homography, _ = cv2.findHomography(quad.astype(np.float64), plane)

        return ScaleInfo(
            source="aruco",
            mm_per_px=marker_mm / side_px,
            confidence=confidence,
            note=(
                f"Marker {found_id}, Kante {side_px:.1f} px = {marker_mm:g} mm. "
                + (
                    "Perspektive ueber Homographie angerechnet."
                    if spread > 0.02
                    else "Aufnahme nahezu frontal."
                )
            ),
            homography=homography,
            _plane_mm_per_unit=1.0,
        )


def _pick(
    corners: list[np.ndarray], ids: list[int], wanted: int
) -> tuple[np.ndarray, int] | None:
    """Gewuenschte Kennung, sonst der groesste Marker im Bild."""
    quads = [
        (np.asarray(c).reshape(4, 2), int(i))
        for c, i in zip(corners, ids, strict=True)
    ]
    if wanted is not None and wanted >= 0:
        for quad, found in quads:
            if found == wanted:
                return quad, found
        return None
    return max(quads, key=lambda q: cv2.contourArea(q[0].astype(np.float32)))
