
import time
import logging
import os
import tempfile
import shutil
import pytest
from io import StringIO
from utils import zeit_messen, pruefe_und_erstelle_ordner

# =============================
# Test für zeit_messen (log-based)
# =============================

@zeit_messen
def warte_eine_sekunde():
    time.sleep(1)
    return "fertig"

def test_zeit_messen_logging(caplog):
    """
    Testet, ob der Decorator `zeit_messen` korrekt die Ausführungszeit loggt.
    """
    caplog.set_level(logging.INFO)
    result = warte_eine_sekunde()

    assert result == "fertig", "Die Funktion gibt nicht den erwarteten Wert zurück."

    logs = "\n".join(caplog.messages).lower()
    assert "dauerte" in logs, f"Der Logeintrag fehlt: {logs}"
    assert "sekunde" in logs or "sekunden" in logs, f"Einheit fehlt im Log: {logs}"

# =============================
# Test für pruefe_und_erstelle_ordner
# =============================

def test_pruefe_und_erstelle_ordner():
    """
    Testet die Funktion `pruefe_und_erstelle_ordner`:
    - erstellt einen Ordner, wenn er nicht existiert
    - verursacht keinen Fehler, wenn der Ordner bereits existiert
    - wirft bei ungültigem Pfad eine Exception
    """
    # 1. Test: Ordner wird erstellt
    temp_dir = tempfile.mkdtemp()
    neuer_ordner = os.path.join(temp_dir, "test_unterordner")
    assert not os.path.exists(neuer_ordner)
    pruefe_und_erstelle_ordner(neuer_ordner)
    assert os.path.isdir(neuer_ordner)

    # 2. Test: Aufruf erneut → kein Fehler
    pruefe_und_erstelle_ordner(neuer_ordner)
    assert os.path.isdir(neuer_ordner)

    # 3. Test: Ungültiger Pfad (Plattformabhängig)
    with pytest.raises(Exception):
        pruefe_und_erstelle_ordner("?:/<>INVALID")  # ungültiger Pfad (Windows-Style)

    # Aufräumen
    shutil.rmtree(temp_dir)
