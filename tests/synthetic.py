"""Synthetische Risse mit **bekannter** Breite.

Der Kern des Dienstes ist eine Messung, und eine Messung prueft man gegen
eine bekannte Groesse. Echte Fotos taugen dafuer nicht: niemand weiss, wie
breit der Riss darauf wirklich war. Also wird er gezeichnet - Breite und
Verlauf sind dann per Konstruktion bekannt, und die Pipeline muss sie
zurueckgeben.

Ein synthetisches Bild prueft **die Geometrie und die Messung**, nicht die
Erkennung. Ob ein Netz einen echten Riss von einer Schattenfuge unterscheidet,
zeigt nur ein Testsatz echter Fotos (siehe training/).
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

RNG = np.random.default_rng(20260924)


def concrete_background(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Grobkoerniger, ungleichmaessig ausgeleuchteter Grauton."""
    rng = np.random.default_rng(seed)
    base = rng.normal(168, 12, size=(h, w)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 1.2)

    speckle = rng.normal(0, 26, size=(h, w)).astype(np.float32)
    speckle = cv2.GaussianBlur(speckle, (0, 0), 3.5)
    base += speckle

    # Lichtverlauf von links oben - genau das, was einen globalen Schwellwert
    # unbrauchbar macht.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base += 26.0 * (1.0 - (yy / h + xx / w) / 2.0)

    return np.clip(base, 0, 255).astype(np.uint8)


def draw_crack(
    canvas: np.ndarray,
    points: np.ndarray,
    width_px: float,
    darkness: int = 95,
) -> np.ndarray:
    """Zeichnet einen Riss bekannter Breite entlang ``points`` in ``(y, x)``.

    Gezeichnet wird auf einem vierfach vergroesserten Feld und danach
    verkleinert: nur so entsteht eine Kante mit Zwischenwerten, und nur dann
    ist eine Breite von 2,5 Pixeln ueberhaupt darstellbar.
    """
    factor = 4
    h, w = canvas.shape[:2]
    big = np.zeros((h * factor, w * factor), dtype=np.uint8)
    pts = (points[:, ::-1] * factor).round().astype(np.int32)  # (y,x) -> (x,y)
    thickness = max(1, int(round(width_px * factor)))
    cv2.polylines(big, [pts], False, 255, thickness, cv2.LINE_AA)

    alpha = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    out = canvas.astype(np.float32)
    out = out * (1.0 - alpha) + (out - darkness).clip(0, 255) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def straight_crack(
    h: int = 480,
    w: int = 640,
    width_px: float = 3.0,
    angle_deg: float = 0.0,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Gerader Riss durch die Bildmitte. Gibt Bild und wahre Punkte zurueck."""
    bg = concrete_background(h, w, seed)
    cy, cx = h / 2.0, w / 2.0
    half = min(h, w) * 0.4
    a = np.radians(angle_deg)
    # y nach unten, daher das Minus beim Sinus.
    dy, dx = -np.sin(a), np.cos(a)
    t = np.linspace(-half, half, 200)
    points = np.stack([cy + dy * t, cx + dx * t], axis=1).astype(np.float32)
    rgb = cv2.cvtColor(draw_crack(bg, points, width_px), cv2.COLOR_GRAY2RGB)
    return rgb, points


def branched_crack(
    h: int = 480, w: int = 640, width_px: float = 3.0, seed: int = 2
) -> np.ndarray:
    """Stamm mit einem Ast - prueft die Zerlegung des Skeletts."""
    bg = concrete_background(h, w, seed)
    trunk = np.array(
        [[80, 100], [180, 220], [260, 300], [360, 420]], dtype=np.float32
    )
    trunk = _densify(trunk)
    branch = _densify(np.array([[260, 300], [300, 480], [330, 590]], dtype=np.float32))
    img = draw_crack(bg, trunk, width_px)
    img = draw_crack(img, branch, width_px * 0.7)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def two_cracks(h: int = 480, w: int = 640, seed: int = 3) -> np.ndarray:
    """Zwei getrennte Risse verschiedener Breite - prueft die Instanzbildung."""
    bg = concrete_background(h, w, seed)
    a = _densify(np.array([[60, 40], [140, 260], [200, 560]], dtype=np.float32))
    b = _densify(np.array([[320, 60], [400, 300], [430, 600]], dtype=np.float32))
    img = draw_crack(bg, a, 5.0)
    img = draw_crack(img, b, 2.0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def blank(h: int = 480, w: int = 640, seed: int = 4) -> np.ndarray:
    """Beton ohne Riss - hier darf nichts gefunden werden."""
    return cv2.cvtColor(concrete_background(h, w, seed), cv2.COLOR_GRAY2RGB)


def with_aruco(
    rgb: np.ndarray, marker_id: int = 7, size_px: int = 120, margin: int = 20
) -> np.ndarray:
    """Legt einen ArUco-Marker in die linke untere Ecke."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)
    out = rgb.copy()
    h = out.shape[0]
    y0 = h - margin - size_px
    out[y0 : y0 + size_px, margin : margin + size_px] = marker[:, :, None]
    return out


def as_png(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def _densify(points: np.ndarray, step: float = 2.0) -> np.ndarray:
    """Stuetzstellen auffuellen, damit die Linie glatt bleibt."""
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:], strict=True):
        dist = float(np.hypot(*(b - a)))
        n = max(2, int(dist / step))
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / n))
    return np.array(out, dtype=np.float32)
