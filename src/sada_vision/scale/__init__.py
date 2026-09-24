"""Pixel in Millimeter - drei Wege, eine Antwort.

Ohne Massstab ist eine Rissbreite eine Zahl ohne Einheit. Der Dienst rechnet
deshalb **nie** heimlich um; er meldet ``-1``, solange keiner der drei Wege
gegriffen hat, und schreibt in ``scale.source``, welcher es war.

    explicit  Der Aufrufer weiss es (Stativ, feste Entfernung, Kalibrierung).
    aruco     Ein gedruckter Marker bekannter Kantenlaenge liegt im Bild.
    lidar     Die AR-App liefert Tiefe und Brennweite.

Genauigkeit in dieser Reihenfolge: ``aruco`` ist der einzige, der auch die
Schraeglage mitnimmt, weil aus vier Eckpunkten eine Homographie folgt und der
Massstab damit ortsabhaengig wird. ``lidar`` ist gut, solange die Flaeche
einigermassen frontal steht. ``explicit`` ist so gut wie die Annahme dahinter.
"""

from .aruco import ArucoScaleResolver
from .base import resolve_scale
from .explicit import explicit_scale
from .lidar import lidar_scale

__all__ = ["ArucoScaleResolver", "explicit_scale", "lidar_scale", "resolve_scale"]
