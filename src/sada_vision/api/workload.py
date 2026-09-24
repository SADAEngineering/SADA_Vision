"""Die schwere Arbeit aus der Ereignisschleife heraushalten.

Eine Rissanalyse rechnet Sekunden, und sie rechnet in numpy und
onnxruntime - also in synchronem C-Code. Stuende sie direkt in einer
``async def``-Route, hielte sie die Ereignisschleife des ganzen Prozesses
an: kein ``/health/live`` mehr, keine zweite Anfrage, nichts. Der
Container-Neustarter sieht dann einen Dienst, der nicht antwortet, und
startet einen gesunden Container neu - mitten in der Arbeit.

Also laeuft die Rechnung in einem Arbeitsfaden. Der Python-Interpreter
steht dem nicht im Weg: numpy, OpenCV und onnxruntime geben die GIL
waehrend ihrer eigenen Arbeit frei, und genau dort liegt die Zeit.

Dazu ein Zaehlwerk: **mehr gleichzeitige Rechnungen als Kerne machen alles
langsamer, nicht schneller.** Jede Bibliothek nimmt sich ohnehin schon
mehrere Kerne (``OMP_NUM_THREADS``), und zwei Laeufe nebeneinander
verdoppeln nicht den Durchsatz, sondern beide Laufzeiten. Wer wartet,
wartet in der Schlange statt im Gedraenge.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import anyio
from anyio import Semaphore

from ..config import get_settings
from ..logging_setup import get_logger

log = get_logger(__name__)

T = TypeVar("T")

_semaphore: Semaphore | None = None


def _gate() -> Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = Semaphore(max(1, get_settings().max_concurrent_analyses))
    return _semaphore


def reset_gate() -> None:
    """Nur fuer Tests."""
    global _semaphore
    _semaphore = None


async def run_heavy(work: Callable[[], T]) -> T:
    """Fuehrt ``work`` in einem Arbeitsfaden aus, hoechstens N gleichzeitig."""
    async with _gate():
        return await anyio.to_thread.run_sync(work)
