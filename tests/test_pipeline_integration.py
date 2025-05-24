import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import os

# Wir testen: dataframe_to_db.main → model_test.main → auswertung.main

def mock_engine():
    """Erzeugt einen mock-fähigen SQLAlchemy Engine-ähnlichen Dummy"""
    mock = MagicMock()
    mock.connect.return_value.__enter__.return_value.execute.return_value = True
    return mock

@pytest.fixture
def dummy_dbverbindung():
    with patch("dataframe_to_db.DatenbankVerbindung") as mock_db, \
         patch("model_test.DatenbankVerbindung") as mock_model_db, \
         patch("auswertung.DatenbankVerbindung") as mock_eval_db:

        dummy_engine = mock_engine()
        dummy_data = {
            "dataset_name": "dummyset",
            "beschreibung": "Testdatensatz",
            "df_tabelle": "dummy_df",
            "x_test_tabelle": "dummy_X_test",
            "x_train_tabelle": "dummy_X_train",
            "y_test_tabelle": "dummy_y_test",
            "y_train_tabelle": "dummy_y_train"
        }

        mock_db().get_engine.return_value = dummy_engine
        mock_model_db().get_engine.return_value = dummy_engine
        mock_eval_db().get_engine.return_value = dummy_engine

        # Rückgabe bei .lade_dataset_metadaten()
        mock_model_db().lade_dataset_metadaten.return_value = [dummy_data]
        mock_eval_db().lade_modelltestergebnisse.return_value = pd.DataFrame({
            "modellname": ["DummyModel"],
            "datenname": ["dummyset"],
            "durchgang": [1],
            "score": [91.2],
            "f1": [88.1],
            "dauer": [1.1],
            "laufzeit": [pd.Timestamp("2025-01-01")],
            "cm": ["[[5, 0], [1, 4]]"]
        })

        yield

@patch("builtins.open")
@patch("os.listdir", return_value=["ranking_dummyset_f1.png"])
def test_pipeline_end_to_end(mock_ls, mock_open, dummy_dbverbindung):
    """
    Führt alle Hauptfunktionen nacheinander aus → simuliert die komplette Pipeline
    """

    from dataframe_to_db import main as df_main
    from model_test import main as model_main
    from auswertung import Auswertung

    # 1. Daten vorbereiten
    df_main()

    # 2. Modelle testen
    model_main()

    # 3. HTML-Reports generieren
    auswertung = Auswertung()
    auswertung.generiere_html_report("datenname", ordner="TestExport")

    # Ergebnis: wurde open() für HTML-Datei aufgerufen?
    assert mock_open.called
