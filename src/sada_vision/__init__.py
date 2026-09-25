"""SADA Vision - Erkennung und Vermessung von Bauteilschaeden auf Fotos."""

__version__ = "0.1.0"

# Version des Draht-Vertrags. Wird in jeder Antwort mitgeschickt.
# Aendert sich nur additiv (siehe docs/API_Vertrag.md).
#
# 1.1: touches_border und width_samples_excluded je Befund,
#      width_at_junction je Ast. Alte Aufrufer laufen unveraendert weiter.
CONTRACT_VERSION = "1.1"
