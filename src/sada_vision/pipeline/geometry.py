"""Von der Maske zum geordneten Rissverlauf.

Der Weg ist immer derselbe:

    Maske -> Zusammenhangskomponenten -> Skelett -> Graph -> Aeste -> Polylinie

Das Skelett ist die Mittellinie des Risses, ein Pixel breit. Sein Graph hat
Knoten (Enden mit einem Nachbarn, Verzweigungen mit dreien oder mehr) und
Kanten dazwischen - jede Kante ist ein Ast und wird zu einer Polylinie. Ein
Riss ohne Verzweigung hat genau einen Ast.
"""

from __future__ import annotations

import inspect
from typing import NamedTuple

import cv2
import numpy as np
from skimage.morphology import remove_small_holes, skeletonize

_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

# scikit-image hat ``area_threshold`` in 0.26 zugunsten von ``max_size``
# abgeloest. Beide Namen kommen vor, je nachdem, was das Image mitbringt.
_HOLE_SIZE_KEYWORD = (
    "max_size"
    if "max_size" in inspect.signature(remove_small_holes).parameters
    else "area_threshold"
)


def clean_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Loecher schliessen, Streusel entfernen."""
    m = mask.astype(bool)
    if min_area > 1:
        m = remove_small_holes(m, **{_HOLE_SIZE_KEYWORD: max(4, min_area // 4)})
    # Kleine Bruecken schliessen: ein Riss, den die Segmentierung an einer
    # Stelle verliert, waere sonst zwei Befunde.
    closed = cv2.morphologyEx(
        m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
    )
    return closed.astype(bool)


def components(
    mask: np.ndarray, min_area: int
) -> list[tuple[np.ndarray, tuple[int, int, int, int], float]]:
    """Zerlegt die Maske in Instanzen.

    Gibt je Instanz die Maske im Zuschnitt, den Kasten ``(y0, x0, y1, x1)``
    und die Flaeche in Pixeln zurueck, die groesste zuerst.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    out = []
    for i in range(1, num):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        sub = labels[y0 : y0 + h, x0 : x0 + w] == i
        out.append((sub, (y0, x0, y0 + h, x0 + w), area))
    out.sort(key=lambda t: t[2], reverse=True)
    return out


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(
        skel.astype(np.uint8), cv2.CV_8U, kernel, borderType=cv2.BORDER_CONSTANT
    )


class Skeleton(NamedTuple):
    """Das zerlegte Skelett einer Instanz.

    ``junctions`` sind die Mittelpunkte der Kreuzungen in ``(y, x)``. Sie
    werden gebraucht, weil die Breitenmessung dort systematisch zu gross
    ausfaellt: in eine Gabelung passt ein groesserer Kreis als in den Riss.
    """

    branches: list[np.ndarray]
    branch_count: int
    junctions: np.ndarray       # (M, 2), leer bei einem Riss ohne Verzweigung


def skeleton_branches(component: np.ndarray) -> Skeleton:
    """Skelettiert eine Instanz und zerlegt das Skelett in Aeste.

    Die Punktfolgen sind geordnet, in ``(y, x)`` und relativ zum Zuschnitt.
    """
    skel = skeletonize(component)
    if not skel.any():
        return Skeleton([], 0, np.zeros((0, 2), dtype=np.float32))

    nb = _neighbour_count(skel)
    nb = np.where(skel, nb, 0)

    node_mask = skel & (nb != 2)
    branch_pixels = skel & (nb >= 3)

    # Eine Kreuzung ist im Skelett selten *ein* Pixel: bei einem T liegen
    # zwei oder drei Pixel mit drei Nachbarn nebeneinander. Wer die einzeln
    # als Knoten zaehlt, bekommt lauter zweipixelige Scheinaeste - aus einem
    # T werden acht Aeste statt drei. Deshalb werden benachbarte Knotenpixel
    # zu *einem* Knoten zusammengefasst.
    _clusters, cluster_of = cv2.connectedComponents(
        node_mask.astype(np.uint8), connectivity=8
    )
    kreuzungen = sorted(
        {
            int(cluster_of[y, x])
            for y, x in zip(*np.nonzero(branch_pixels), strict=True)
        }
    )
    branch_count = len(kreuzungen)

    # Je Kreuzung ihr Schwerpunkt - dort ist die Breitenmessung unbrauchbar,
    # und zwar im Umkreis von etwa einer Rissbreite.
    junctions = np.array(
        [
            np.argwhere(cluster_of == kennung).mean(axis=0)
            for kennung in kreuzungen
        ],
        dtype=np.float32,
    ).reshape(-1, 2)

    h, w = skel.shape
    node_set = {(int(y), int(x)) for y, x in zip(*np.nonzero(node_mask), strict=True)}
    visited_interior: set[tuple[int, int]] = set()
    visited_direct: set[frozenset] = set()
    branches: list[np.ndarray] = []

    def neighbours(p: tuple[int, int]) -> list[tuple[int, int]]:
        y, x = p
        out = []
        for dy, dx in _NEIGHBOURS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                out.append((ny, nx))
        return out

    def cluster(p: tuple[int, int]) -> int:
        return int(cluster_of[p[0], p[1]])

    # 1. Aeste, die an einem Knoten beginnen
    for node in sorted(node_set):
        own = cluster(node)
        for first in neighbours(node):
            if first in node_set:
                if cluster(first) == own:
                    continue  # innerhalb derselben Kreuzung, kein Ast
                key = frozenset((node, first))
                if key in visited_direct:
                    continue
                visited_direct.add(key)
                branches.append(np.array([node, first], dtype=np.float32))
                continue
            if first in visited_interior:
                continue
            path = [node, first]
            visited_interior.add(first)
            prev, cur = node, first
            while True:
                nxt = [p for p in neighbours(cur) if p != prev]
                # Bei nb == 2 gibt es genau einen Nachfolger; sonst ist cur
                # ein Knoten und der Ast endet hier.
                if len(nxt) != 1:
                    break
                step = nxt[0]
                path.append(step)
                if step in node_set or step in visited_interior:
                    break
                visited_interior.add(step)
                prev, cur = cur, step
            branches.append(np.array(path, dtype=np.float32))

    # 2. Geschlossene Ringe ohne jeden Knoten
    remaining = {
        (int(y), int(x))
        for y, x in zip(*np.nonzero(skel), strict=True)
        if (int(y), int(x)) not in node_set
        and (int(y), int(x)) not in visited_interior
    }
    while remaining:
        start = remaining.pop()
        path = [start]
        prev, cur = None, start
        while True:
            nxt = [p for p in neighbours(cur) if p != prev and p in remaining]
            if not nxt:
                break
            step = nxt[0]
            remaining.discard(step)
            path.append(step)
            prev, cur = cur, step
        if len(path) >= 3:
            path.append(path[0])  # Ring schliessen
            branches.append(np.array(path, dtype=np.float32))

    return Skeleton(branches, branch_count, junctions)


def polyline_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    d = np.diff(points, axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def simplify_rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker, iterativ.

    Ein Skelett liefert einen Punkt je Pixel; ein Riss von 2.000 Pixeln
    Laenge braucht keine 2.000 Stuetzstellen. Epsilon ist der groesste
    zugelassene Abstand zur Originallinie, in Pixeln.
    """
    n = points.shape[0]
    if n < 3 or epsilon <= 0:
        return points

    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        seg = points[i0 : i1 + 1]
        a, b = points[i0], points[i1]
        ab = b - a
        norm = float(np.hypot(ab[0], ab[1]))
        if norm < 1e-9:
            dist = np.hypot(seg[:, 0] - a[0], seg[:, 1] - a[1])
        else:
            # Abstand Punkt-Gerade ueber das Kreuzprodukt
            dist = (
                np.abs(ab[0] * (a[1] - seg[:, 1]) - (a[0] - seg[:, 0]) * ab[1]) / norm
            )
        inner = dist[1:-1]
        if inner.size == 0:
            continue
        idx = int(np.argmax(inner)) + 1
        if float(dist[idx]) > epsilon:
            keep[i0 + idx] = True
            stack.append((i0, i0 + idx))
            stack.append((i0 + idx, i1))
    return points[keep]


def resample_uniform(points: np.ndarray, step_px: float) -> np.ndarray:
    """Tastet den Verlauf in **festem Abstand** ab.

    Das ist der Unterschied zwischen einer gezeichneten Linie und einer
    Messreihe. Ramer-Douglas-Peucker optimiert die *Form*: er laesst Punkte
    stehen, wo die Linie knickt, und wirft sie weg, wo sie gerade laeuft.
    Fuer das Zeichnen ist das richtig - fuer das Messen falsch, denn auf
    einem geraden Stueck bleiben dann zwei Stuetzstellen ueber vierzig
    Pixel, und die breiteste Stelle dazwischen sieht niemand.

    Abgetastet wird entlang des **rohen** Skelettpfades, nicht entlang der
    vereinfachten Linie: sonst waeren die Punkte zwar gleichmaessig
    verteilt, lagen aber neben dem Riss, weil die Vereinfachung Kurven
    abschneidet.
    """
    n = points.shape[0]
    if n < 2 or step_px <= 0:
        return points

    schritte = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    weg = np.concatenate([[0.0], np.cumsum(schritte)])
    gesamt = float(weg[-1])
    if gesamt < step_px:
        # Kuerzer als ein Schritt: Anfang und Ende genuegen.
        return points[[0, -1]]

    # Anfang und Ende bleiben immer stehen - sie sind die Enden des Risses.
    anzahl = max(2, int(round(gesamt / step_px)) + 1)
    ziel = np.linspace(0.0, gesamt, anzahl)
    return np.stack(
        [np.interp(ziel, weg, points[:, 0]), np.interp(ziel, weg, points[:, 1])],
        axis=1,
    ).astype(np.float32)


def resample(points: np.ndarray, max_points: int) -> np.ndarray:
    """Harte Obergrenze - die Antwort soll durch die Leitung passen."""
    n = points.shape[0]
    if n <= max_points:
        return points
    idx = np.linspace(0, n - 1, max_points).round().astype(int)
    return points[np.unique(idx)]


def principal_orientation(points: np.ndarray) -> float:
    """Hauptrichtung in Grad.

    0 Grad = waagerecht, positiv gegen den Uhrzeigersinn, Wertebereich
    [-90, 90). Bildkoordinaten haben y nach unten, daher das Minus.
    """
    if points.shape[0] < 2:
        return 0.0
    centred = points - points.mean(axis=0)
    if centred.shape[0] < 2:
        return 0.0
    cov = np.cov(centred.T)
    if not np.all(np.isfinite(cov)) or cov.shape != (2, 2):
        return 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    angle = float(np.degrees(np.arctan2(-major[0], major[1])))
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def tortuosity(points: np.ndarray) -> float:
    """Weglaenge geteilt durch Luftlinie. 1,0 = schnurgerade."""
    if points.shape[0] < 2:
        return 1.0
    direct = float(np.hypot(points[-1, 0] - points[0, 0], points[-1, 1] - points[0, 1]))
    if direct < 1e-6:
        return 1.0
    return max(1.0, polyline_length(points) / direct)
