"""Start ohne Container: python -m sada_vision"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "sada_vision.api.app:app",
        host=os.environ.get("SADAVISION_HOST", "0.0.0.0"),
        port=int(os.environ.get("SADAVISION_PORT", "8080")),
        reload=os.environ.get("SADAVISION_RELOAD", "0") == "1",
        log_config=None,  # structlog macht das Protokoll
        # Uvicorns Zugriffsprotokoll ist Klartext. In einem Strom, der sonst
        # aus JSON-Zeilen besteht, ist das fuer jede Auswertung Muell - und
        # die Middleware protokolliert ohnehin jede Anfrage, mit
        # Korrelationskennung und Dauer.
        access_log=False,
    )


if __name__ == "__main__":
    main()
