"""Rissbreite je Punkt des Verlaufs.

Zwei Verfahren, beide auf demselben Skelett:

**distance_transform** - die Distanztransformation der Maske gibt je Pixel den
Abstand zum naechsten Hintergrund. Auf dem Skelett ist das der Radius des
groessten eingeschriebenen Kreises, die Breite also das Doppelte. Schnell,
aber auf ganze Pixel gerundet: bei einem Riss von drei Pixeln Breite sind das
gut 30 Prozent Sprungweite.

**perpendicular** (Vorgabe) - quer zum Verlauf wird das *Wahrscheinlichkeits-
bild* abgetastet, nicht die harte Maske, und der Rand dort gesetzt, wo die
Wahrscheinlichkeit die Schwelle schneidet. Zwischen zwei Abtastpunkten wird
linear interpoliert; das ergibt Subpixelgenauigkeit. Das ist der Unterschied
zwischen "etwa 2 bis 3 Pixel" und "2,4 Pixel" - und bei 0,3 mm Grenzwert
entscheidet genau das ueber den Befund.

Beide messen die Breite *in der Bildebene*. Steht das Foto schraeg auf dem
Bauteil, ist das nicht die wahre Breite - dafuer sorgt der Massstab
(``scale/``), nicht dieses Modul.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

_STEP = 0.25          # Abtastschritt quer zum Verlauf, in Pixeln
_MIN_RADIUS = 3.0
_MAX_RADIUS = 64.0
_SMOOTH = 2           # Halbfenster der Tangentenglaettung, in Stuetzstellen


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """Abstand zum Hintergrund, in Pixeln.

    Enthaelt die Maske keinen einzigen Hintergrundpixel, gibt OpenCV einen
    Ersatzwert nahe der float32-Grenze zurueck - das Doppelte davon laeuft
    ueber. Deshalb die Kappung auf die Bilddiagonale: weiter weg als bis zur
    gegenueberliegenden Ecke kann kein Rand liegen.
    """
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    h, w = mask.shape[:2]
    return np.minimum(dist, float(np.hypot(h, w)))


def tangents(points: np.ndarray) -> np.ndarray:
    """Geglaettete Tangenten je Stuetzstelle, normiert, als ``(N, 2)``.

    Zentrale Differenz ueber ein Fenster: die Richtung von Pixel zu Pixel ist
    achtfach gequantelt (nur 45-Grad-Schritte), und ein daraus gebildetes Lot
    wuerde die Breite um bis zu einem Faktor 1,41 verfehlen.
    """
    n = points.shape[0]
    if n == 1:
        return np.array([[0.0, 1.0]], dtype=np.float64)

    k = min(_SMOOTH, max(1, n // 2))
    idx = np.arange(n)
    lo = np.clip(idx - k, 0, n - 1)
    hi = np.clip(idx + k, 0, n - 1)
    vec = points[hi] - points[lo]

    norm = np.hypot(vec[:, 0], vec[:, 1])
    bad = norm < 1e-9
    if np.any(bad):
        # Entartete Stellen (Ring, Doppelpunkt) bekommen die Nachbarrichtung.
        fallback = points[np.clip(idx + 1, 0, n - 1)] - points[np.clip(idx - 1, 0, n - 1)]
        vec[bad] = fallback[bad]
        norm = np.hypot(vec[:, 0], vec[:, 1])
        norm[norm < 1e-9] = 1.0
    return vec / norm[:, None]


def widths_from_distance(points: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Breite = zweifacher Abstand zum Rand."""
    ys = np.clip(points[:, 0], 0, dist.shape[0] - 1)
    xs = np.clip(points[:, 1], 0, dist.shape[1] - 1)
    d = map_coordinates(dist, [ys, xs], order=1, mode="nearest")
    return (2.0 * d).astype(np.float32)


def widths_perpendicular(
    points: np.ndarray,
    prob: np.ndarray,
    dist: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Subpixelbreite aus dem Wahrscheinlichkeitsbild, quer zum Verlauf."""
    n = points.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    # Suchradius aus der groben Breite ableiten - weit genug, um den Rand zu
    # finden, eng genug, um an Verzweigungen nicht in den Nachbarast zu laufen.
    coarse = widths_from_distance(points, dist).astype(np.float64)
    radius = np.clip(coarse * 1.5 + 2.0, _MIN_RADIUS, _MAX_RADIUS)
    steps = int(np.ceil(float(radius.max()) / _STEP))

    t = tangents(points)
    # Lot auf die Tangente, in (y, x): (ty, tx) -> (tx, -ty)
    normal = np.stack([t[:, 1], -t[:, 0]], axis=1)

    offsets = np.arange(1, steps + 1, dtype=np.float64) * _STEP  # (K,)

    def side(sign: float) -> np.ndarray:
        ys = points[:, 0:1] + sign * normal[:, 0:1] * offsets[None, :]
        xs = points[:, 1:2] + sign * normal[:, 1:2] * offsets[None, :]
        sampled = map_coordinates(
            prob, [ys.ravel(), xs.ravel()], order=1, mode="constant", cval=0.0
        ).reshape(n, steps)
        return _first_crossing(sampled, offsets, threshold, radius)

    return (side(+1.0) + side(-1.0)).astype(np.float32)


def _first_crossing(
    sampled: np.ndarray,
    offsets: np.ndarray,
    threshold: float,
    radius: np.ndarray,
) -> np.ndarray:
    """Abstand bis zum ersten Unterschreiten der Schwelle, linear verfeinert."""
    n, k = sampled.shape
    below = sampled < threshold

    # Ausserhalb des erlaubten Radius gilt alles als Hintergrund, damit die
    # Suche dort in jedem Fall endet.
    beyond = offsets[None, :] > radius[:, None]
    below = below | beyond

    has = below.any(axis=1)
    first = np.where(has, below.argmax(axis=1), k - 1)

    d_out = offsets[first]
    prev_idx = first - 1

    # Wert an der letzten Stelle *vor* dem Sprung; bei first == 0 ist das der
    # Punkt auf dem Skelett selbst, dessen Wahrscheinlichkeit wir als 1 setzen
    # (er liegt per Konstruktion in der Maske).
    p_prev = np.where(
        prev_idx >= 0,
        sampled[np.arange(n), np.clip(prev_idx, 0, k - 1)],
        1.0,
    )
    p_cur = sampled[np.arange(n), first]
    d_prev = np.where(prev_idx >= 0, offsets[np.clip(prev_idx, 0, k - 1)], 0.0)

    # np.where rechnet beide Zweige aus - die Division muss deshalb selbst
    # maskiert werden, sonst gibt es Warnungen und inf an Stellen, die
    # anschliessend ohnehin verworfen werden.
    denom = p_prev - p_cur
    usable = np.abs(denom) > 1e-9
    frac = np.zeros_like(denom, dtype=np.float64)
    np.divide(p_prev - threshold, denom, out=frac, where=usable)
    frac = np.clip(frac, 0.0, 1.0)
    crossing = d_prev + frac * (d_out - d_prev)

    # Kein Rand gefunden -> Radius als Obergrenze melden, nicht null.
    crossing = np.where(has, crossing, radius)
    return np.maximum(crossing, 0.0)


def junction_mask(
    points: np.ndarray,
    junctions: np.ndarray,
    widths_px: np.ndarray,
    radius_factor: float = 1.5,
    radius_floor: float = 2.0,
) -> np.ndarray:
    """Markiert die Stuetzstellen, an denen die Breite nicht gilt.

    An einer Verzweigung misst jedes Verfahren zu breit, und zwar aus einem
    geometrischen Grund: in eine Gabelung passt ein groesserer Kreis als in
    den Riss, und ein Lot quer zum einen Ast schneidet den anderen. Der
    Effekt reicht etwa eine Rissbreite weit - deshalb haengt der Radius an
    der dort gemessenen Breite und ist keine feste Zahl.

    Die Werte werden **nicht geloescht**. Sie stehen weiter in der Antwort,
    nur zaehlen sie nicht in Hoechstwert, Mittel und Perzentil. Wer sie
    zeichnen will, kann das; wer eine Rissbreite braucht, nimmt sie nicht.
    """
    n = points.shape[0]
    if n == 0 or junctions is None or len(junctions) == 0:
        return np.zeros(n, dtype=bool)

    junctions = np.asarray(junctions, dtype=np.float64).reshape(-1, 2)
    abstand = np.linalg.norm(
        points[:, None, :].astype(np.float64) - junctions[None, :, :], axis=2
    ).min(axis=1)
    radius = np.maximum(widths_px.astype(np.float64) * radius_factor, radius_floor)
    return abstand <= radius


def to_millimetres(width_px: np.ndarray, mm_per_px: np.ndarray) -> np.ndarray:
    """-1 bleibt -1: unbekannter Massstab wird nicht geraten."""
    out = np.where(mm_per_px > 0, width_px * mm_per_px, -1.0)
    return out.astype(np.float32)


def length_mm(points: np.ndarray, mm_per_px: np.ndarray) -> float:
    """Laenge unter ortsabhaengigem Massstab: je Abschnitt mit dem Mittel."""
    if points.shape[0] < 2 or np.any(mm_per_px <= 0):
        return -1.0
    seg = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    factor = 0.5 * (mm_per_px[:-1] + mm_per_px[1:])
    return float(np.sum(seg * factor))
