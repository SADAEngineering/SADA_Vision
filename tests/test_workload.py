"""Der Dienst muss waehrend einer Rechnung ansprechbar bleiben.

Eine Rissanalyse rechnet Sekunden in synchronem C-Code. Stuende sie direkt
in der ``async def``-Route, hielte sie die Ereignisschleife des ganzen
Prozesses an - und der Container-Neustarter saehe einen Dienst, der auf
``/health/live`` nicht antwortet, und startete einen gesunden Container
mitten in der Arbeit neu.

Das ist die Art Fehler, die unter Last auftritt und im Testlauf nie: eine
einzelne Anfrage merkt nichts davon.
"""

from __future__ import annotations

import time

import anyio
import pytest

from sada_vision.api import workload
from sada_vision.config import get_settings, reset_settings


@pytest.fixture(autouse=True)
def _clean():
    workload.reset_gate()
    reset_settings()
    yield
    workload.reset_gate()
    reset_settings()


@pytest.mark.anyio
async def test_schwere_arbeit_blockiert_die_ereignisschleife_nicht():
    """Waehrend die Rechnung laeuft, muss anderes drankommen."""
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        for _ in range(20):
            await anyio.sleep(0.01)
            ticks += 1

    def heavy():
        time.sleep(0.25)   # steht fuer numpy/onnxruntime
        return "fertig"

    async with anyio.create_task_group() as group:
        group.start_soon(heartbeat)
        result = await workload.run_heavy(heavy)

    assert result == "fertig"
    # Ohne Arbeitsfaden waeren waehrend der 0,25 s null Schlaege durchgekommen.
    assert ticks >= 10, f"nur {ticks} Schlaege - die Schleife stand"


@pytest.mark.anyio
async def test_hoechstens_so_viele_rechnungen_wie_erlaubt():
    """Mehr gleichzeitige Rechnungen als Kerne machen alles langsamer."""
    settings = get_settings()
    settings.max_concurrent_analyses = 2
    workload.reset_gate()

    laufend = 0
    hoechststand = 0

    def heavy():
        nonlocal laufend, hoechststand
        laufend += 1
        hoechststand = max(hoechststand, laufend)
        time.sleep(0.05)
        laufend -= 1
        return None

    async with anyio.create_task_group() as group:
        for _ in range(8):
            group.start_soon(workload.run_heavy, heavy)

    assert hoechststand <= 2, f"{hoechststand} Rechnungen gleichzeitig"


@pytest.fixture
def anyio_backend():
    return "asyncio"
