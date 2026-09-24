"""Masse fuer duenne Strukturen.

Die uebliche IoU taugt fuer Risse nur bedingt. Ein Riss von drei Pixeln
Breite, um ein Pixel danebengelegt, hat eine IoU nahe null - obwohl er
gefunden wurde. Umgekehrt belohnt IoU ein Netz, das den Riss einfach dicker
malt, denn Flaeche ist billig. Beides ist genau verkehrt herum zu dem, was
dieser Dienst braucht: **die Lage** muss stimmen, und **die Breite** darf
nicht geschoent sein.

Deshalb drei Masse nebeneinander:

``iou`` / ``f1``       Das Uebliche, zum Vergleich mit der Literatur.
``tolerant_f1``        Treffer gelten innerhalb eines Toleranzbandes von
                       wenigen Pixeln. Das ist die Zahl, die zur Frage
                       "wurde der Riss gefunden" passt.
``width_bias``         Verhaeltnis der vorhergesagten zur wahren Flaeche bei
                       gleicher Skelettlaenge - schlaegt aus, wenn das Netz
                       systematisch zu breit oder zu duenn malt. Ueber 1 =
                       zu dick. Das ist die Zahl, die zur Messung passt.
"""

from __future__ import annotations

import cv2
import numpy as np

_EPS = 1e-7


def confusion(pred: np.ndarray, truth: np.ndarray) -> tuple[int, int, int]:
    p = pred.astype(bool)
    t = truth.astype(bool)
    return int(np.sum(p & t)), int(np.sum(p & ~t)), int(np.sum(~p & t))


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    tp, fp, fn = confusion(pred, truth)
    return tp / (tp + fp + fn + _EPS)


def f1(pred: np.ndarray, truth: np.ndarray) -> float:
    tp, fp, fn = confusion(pred, truth)
    return 2 * tp / (2 * tp + fp + fn + _EPS)


def tolerant_f1(pred: np.ndarray, truth: np.ndarray, tolerance: int = 2) -> float:
    """F1 mit Toleranzband: ein Treffer zaehlt, wenn er nah genug liegt.

    Die Genauigkeit wird gegen die *geweitete* Wahrheit gerechnet, die
    Trefferquote gegen die *geweitete* Vorhersage. So wird weder ein Versatz
    von einem Pixel bestraft noch eine zu dicke Linie belohnt.
    """
    if tolerance <= 0:
        return f1(pred, truth)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1)
    )
    p = pred.astype(np.uint8)
    t = truth.astype(np.uint8)
    p_wide = cv2.dilate(p, kernel)
    t_wide = cv2.dilate(t, kernel)

    precision = np.sum(p & t_wide) / (np.sum(p) + _EPS)
    recall = np.sum(t & p_wide) / (np.sum(t) + _EPS)
    if precision + recall < _EPS:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def width_bias(pred: np.ndarray, truth: np.ndarray) -> float:
    """Malt das Netz zu dick? Flaechenverhaeltnis je Laengeneinheit.

    1,0 = im Mittel richtig breit. 1,3 = dreissig Prozent zu dick, und damit
    waere jede gemeldete Rissbreite um dreissig Prozent zu gross.
    """
    from skimage.morphology import skeletonize

    t_area = float(np.sum(truth))
    p_area = float(np.sum(pred))
    if t_area < 1 or p_area < 1:
        return float("nan")

    t_len = float(np.sum(skeletonize(truth.astype(bool))))
    p_len = float(np.sum(skeletonize(pred.astype(bool))))
    if t_len < 1 or p_len < 1:
        return float("nan")

    return (p_area / p_len) / (t_area / t_len)


class Running:
    """Sammelt die Masse ueber einen ganzen Durchlauf.

    IoU und F1 werden **global** gerechnet (alle Pixel in einen Topf), nicht
    als Mittel ueber Bilder: sonst zieht ein rissfreies Bild mit einem
    einzigen falschen Pixel den Schnitt auf null und verzerrt alles.
    """

    def __init__(self, tolerance: int = 2) -> None:
        self.tp = self.fp = self.fn = 0
        self._tolerant: list[float] = []
        self._bias: list[float] = []
        self._tolerance = tolerance

    def update(self, pred: np.ndarray, truth: np.ndarray) -> None:
        tp, fp, fn = confusion(pred, truth)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        if truth.any():
            self._tolerant.append(tolerant_f1(pred, truth, self._tolerance))
            bias = width_bias(pred, truth)
            if np.isfinite(bias):
                self._bias.append(bias)

    def summary(self) -> dict:
        return {
            "iou": self.tp / (self.tp + self.fp + self.fn + _EPS),
            "f1": 2 * self.tp / (2 * self.tp + self.fp + self.fn + _EPS),
            "precision": self.tp / (self.tp + self.fp + _EPS),
            "recall": self.tp / (self.tp + self.fn + _EPS),
            "tolerant_f1": float(np.mean(self._tolerant)) if self._tolerant else 0.0,
            "width_bias": float(np.median(self._bias)) if self._bias else float("nan"),
        }
