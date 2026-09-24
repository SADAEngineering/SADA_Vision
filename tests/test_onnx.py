"""Der ONNX-Pfad - der, der im Betrieb laeuft.

Alle anderen Tests laufen gegen den Notbehelf oder ein Orakel. Im Betrieb
laeuft aber ``OnnxSegmenter``, und dort steckt die Fehlerklasse, die am
teuersten ist: **falsche Normierung**. Stehen in der Begleitdatei andere
Mittelwerte als beim Training, rechnet der Dienst mit falsch normierten
Bildern - und das faellt nicht als Fehler auf, sondern als schlechte
Erkennung, Monate spaeter.

Das Testmodell ist kein gelerntes Netz, sondern ein fester Rechenweg mit
genau der Form, die der Dienst erwartet: dunkle Pixel gelten als Riss.
Damit ist die Vorhersage vorhersagbar, und jede Abweichung ist ein Fehler
in der Kette und nicht im Modell.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from sada_vision.config import get_settings, reset_settings
from sada_vision.models import registry
from sada_vision.models.onnx_segmenter import OnnxSegmenter, _sigmoid

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    shutil.copy(FIXTURES / "dummy_dark.onnx", tmp_path / "crack_unet_r18.onnx")
    shutil.copy(FIXTURES / "dummy_dark.json", tmp_path / "crack_unet_r18.json")
    return tmp_path


@pytest.fixture(autouse=True)
def _clean():
    registry.reset_cache()
    reset_settings()
    yield
    registry.reset_cache()
    reset_settings()


def test_stabiles_sigmoid_laeuft_nicht_ueber():
    """exp(710) ist inf - und ein Logit von 800 ist bei untrainierten
    Zwischenstaenden nicht ungewoehnlich."""
    x = np.array([-900.0, -50.0, 0.0, 50.0, 900.0], dtype=np.float32)
    got = _sigmoid(x)
    assert np.all(np.isfinite(got))
    assert got[0] == pytest.approx(0.0, abs=1e-6)
    assert got[2] == pytest.approx(0.5)
    assert got[4] == pytest.approx(1.0, abs=1e-6)


def test_onnx_modell_wird_geladen_und_liefert_wahrscheinlichkeiten(model_dir):
    seg = OnnxSegmenter(model_dir / "crack_unet_r18.onnx")
    assert seg.info.kind == "onnx"
    assert seg.info.trained is True
    assert seg.info.name == "dummy-dark"
    assert seg.tile == 128

    rgb = np.full((200, 300, 3), 220, dtype=np.uint8)   # helle Flaeche
    rgb[90:110, 50:250] = 20                            # dunkler Balken
    prob = seg.probability(rgb)

    assert prob.shape == (200, 300)
    assert prob.dtype == np.float32
    assert float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0
    assert prob[100, 150] > prob[20, 150], "dunkler Balken muss hoeher liegen"


def test_begleitdatei_bestimmt_die_normierung(model_dir, tmp_path):
    """Andere Mittelwerte muessen zu anderen Ausgaben fuehren.

    Wuerde die Begleitdatei ignoriert, liefe der Dienst stillschweigend mit
    den falschen Zahlen - genau das soll hier auffallen.
    """
    rgb = np.full((128, 128, 3), 128, dtype=np.uint8)
    normal = OnnxSegmenter(model_dir / "crack_unet_r18.onnx").probability(rgb)

    abweichend = json.loads((model_dir / "crack_unet_r18.json").read_text())
    abweichend["mean"] = [0.0, 0.0, 0.0]
    abweichend["std"] = [1.0, 1.0, 1.0]
    other_dir = tmp_path / "andere"
    other_dir.mkdir()
    shutil.copy(model_dir / "crack_unet_r18.onnx", other_dir / "m.onnx")
    (other_dir / "m.json").write_text(json.dumps(abweichend), encoding="utf-8")

    anders = OnnxSegmenter(other_dir / "m.onnx").probability(rgb)
    assert not np.allclose(normal, anders)


def test_fehlende_begleitdatei_faellt_auf_imagenet_zurueck(tmp_path):
    shutil.copy(FIXTURES / "dummy_dark.onnx", tmp_path / "nackt.onnx")
    seg = OnnxSegmenter(tmp_path / "nackt.onnx")
    assert seg.info.name == "nackt"
    assert seg.probability(np.full((128, 128, 3), 100, np.uint8)).shape == (128, 128)


def test_registry_nimmt_das_onnx_wenn_es_da_ist(model_dir, monkeypatch):
    monkeypatch.setenv("SADAVISION_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("SADAVISION_CRACK_MODEL", "crack_unet_r18.onnx")
    reset_settings()
    registry.reset_cache()

    seg = registry.get_segmenter("crack")
    assert seg.info.kind == "onnx"
    assert seg.info.trained is True

    status = registry.registry_status()[0]
    assert status["weights_present"] is True
    assert status["trained"] is True


def test_registry_faellt_zurueck_wenn_der_name_nicht_passt(model_dir, monkeypatch):
    """Der haeufigste Betriebsfehler: Datei da, Name falsch gesetzt."""
    monkeypatch.setenv("SADAVISION_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("SADAVISION_CRACK_MODEL", "vertippt.onnx")
    reset_settings()
    registry.reset_cache()

    seg = registry.get_segmenter("crack")
    assert seg.info.kind == "classic"
    assert seg.info.trained is False
    assert registry.registry_status()[0]["weights_present"] is False


def test_kaputte_modelldatei_bringt_den_dienst_nicht_um(tmp_path, monkeypatch):
    (tmp_path / "crack_unet_r18.onnx").write_bytes(b"kein modell")
    monkeypatch.setenv("SADAVISION_MODEL_DIR", str(tmp_path))
    reset_settings()
    registry.reset_cache()

    seg = registry.get_segmenter("crack")
    assert seg.info.kind == "classic", "Ladefehler muss in den Notbehelf fuehren"


def test_ganze_pipeline_mit_onnx_modell(model_dir, monkeypatch):
    """Durchstich bis zum Befund - mit dem Segmentierer des Betriebs."""
    from sada_vision.domain import ScaleInfo, SourceImage
    from sada_vision.pipeline import analyse_crack

    monkeypatch.setenv("SADAVISION_MODEL_DIR", str(model_dir))
    reset_settings()
    registry.reset_cache()
    get_settings()

    rgb = np.full((300, 500, 3), 230, dtype=np.uint8)
    rgb[148:153, 60:440] = 15   # dunkler Strich, 5 px breit

    result = analyse_crack(
        SourceImage(rgb=rgb, width=500, height=300),
        ScaleInfo(source="explicit", mm_per_px=0.04, confidence=1.0),
    )

    assert result.model_kind == "onnx"
    assert result.model_trained is True
    assert len(result.instances) == 1
    inst = result.instances[0]
    assert inst.orientation_class == "horizontal"
    assert inst.width_max_mm > 0
    # 5 px bei 0,04 mm/px sind 0,2 mm
    widths = np.concatenate([p.width_mm for p in inst.paths])
    assert abs(float(np.median(widths)) - 0.2) < 0.06


def test_modelleigene_schwelle_wird_verwendet(model_dir, monkeypatch):
    """Die Schwelle aus der Begleitdatei schlaegt die Vorgabe des Dienstes.

    Sie wird beim Bewerten abgesucht und korrigiert unter anderem die
    Breitenverzerrung - ein Netz, das Risse zu breit malt, wird damit
    wieder geradegerueckt. Faellt sie weg, misst der Dienst mit einem
    fremden Wert, und zwar stillschweigend.
    """
    import json as _json

    from sada_vision.domain import ScaleInfo, SourceImage
    from sada_vision.pipeline import analyse_crack

    meta = _json.loads((model_dir / "crack_unet_r18.json").read_text())
    meta["threshold"] = 0.8      # deutlich strenger als die Vorgabe 0,5
    (model_dir / "crack_unet_r18.json").write_text(_json.dumps(meta), encoding="utf-8")

    monkeypatch.setenv("SADAVISION_MODEL_DIR", str(model_dir))
    reset_settings()
    registry.reset_cache()
    assert registry.get_segmenter("crack").info.threshold == pytest.approx(0.8)

    # Weicher Verlauf von hell nach dunkel: wo genau die Maske endet,
    # haengt allein an der Schwelle.
    rgb = np.zeros((200, 400, 3), np.uint8)
    rgb[:, :] = np.linspace(255, 0, 400).astype(np.uint8)[None, :, None]
    source = SourceImage(rgb=rgb, width=400, height=200)

    streng = analyse_crack(source, ScaleInfo(source="none"))
    locker = analyse_crack(source, ScaleInfo(source="none"), threshold=0.5)

    flaeche_streng = sum(i.area_px for i in streng.instances)
    flaeche_locker = sum(i.area_px for i in locker.instances)
    assert flaeche_streng < flaeche_locker, (
        "Die strengere Modellschwelle muss eine kleinere Maske ergeben - "
        "sonst wurde sie ignoriert"
    )


def test_ohne_eigene_schwelle_gilt_die_vorgabe(model_dir, monkeypatch):
    monkeypatch.setenv("SADAVISION_MODEL_DIR", str(model_dir))
    reset_settings()
    registry.reset_cache()
    # Die Begleitdatei des Testmodells nennt 0,5 - dasselbe wie die Vorgabe.
    assert registry.get_segmenter("crack").info.threshold == pytest.approx(0.5)
    assert get_settings().mask_threshold == pytest.approx(0.5)


def test_notbehelf_hat_keine_eigene_schwelle():
    from sada_vision.models.classic_segmenter import ClassicRidgeSegmenter

    assert ClassicRidgeSegmenter().info.threshold == -1.0
