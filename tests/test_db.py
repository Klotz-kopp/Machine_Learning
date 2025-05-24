import pytest
from unittest.mock import patch, MagicMock
from db import DatenbankVerbindung
import pandas as pd

# =======================
# test_verbindung
# =======================

@patch("db.create_engine")
def test_test_verbindung_erfolgreich(mock_create_engine):
    mock_engine = MagicMock()
    mock_connect = MagicMock()
    mock_connect.__enter__.return_value.execute.return_value = True
    mock_engine.connect.return_value = mock_connect
    mock_create_engine.return_value = mock_engine

    with patch.object(DatenbankVerbindung, "__init__", return_value=None):
        dbv = DatenbankVerbindung()
        dbv.engine = mock_engine
        dbv.verbindung_erfolgreich = True

        assert dbv.test_verbindung() is True

@patch("db.create_engine")
def test_test_verbindung_fehlgeschlagen(mock_create_engine):
    mock_engine = MagicMock()
    mock_engine.connect.side_effect = Exception("Verbindungsfehler")
    mock_create_engine.return_value = mock_engine

    with patch.object(DatenbankVerbindung, "__init__", return_value=None):
        dbv = DatenbankVerbindung()
        dbv.engine = mock_engine
        dbv.verbindung_erfolgreich = True

        assert dbv.test_verbindung() is False

def test_test_verbindung_flag_false():
    with patch.object(DatenbankVerbindung, "__init__", return_value=None):
        dbv = DatenbankVerbindung()
        dbv.verbindung_erfolgreich = False

        assert dbv.test_verbindung() is False

# =======================
# lade_dataset_metadaten
# =======================

def test_lade_dataset_metadaten():
    dummy_data = [
        {"dataset_name": "iris", "beschreibung": "Blumen", "df_tabelle": "iris_df",
         "x_test_tabelle": "iris_X_test", "x_train_tabelle": "iris_X_train",
         "y_test_tabelle": "iris_y_test", "y_train_tabelle": "iris_y_train"}
    ]

    dummy_result = [MagicMock(_mapping=row) for row in dummy_data]

    mock_conn = MagicMock()
    mock_conn.execute.return_value = dummy_result

    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    with patch.object(DatenbankVerbindung, "__init__", return_value=None):
        dbv = DatenbankVerbindung()
        dbv.engine = mock_engine
        dbv.db_schema = "public"

        result = dbv.lade_dataset_metadaten()
        assert isinstance(result, list)
        assert result[0]["dataset_name"] == "iris"

# =======================
# schreibe_dataframe
# =======================

def test_schreibe_dataframe():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    mock_engine = MagicMock()
    with patch.object(DatenbankVerbindung, "__init__", return_value=None):
        dbv = DatenbankVerbindung()
        dbv.engine = mock_engine
        dbv.db_schema = "public"

        with patch.object(df, "to_sql") as mock_to_sql:
            dbv.schreibe_dataframe(df, "test_tabelle")
            mock_to_sql.assert_called_once_with(
                name="test_tabelle",
                con=mock_engine,
                if_exists="replace",
                index=False,
                schema="public"
            )

# =======================
# schreibe_metadaten
# =======================

def test_schreibe_metadaten():
    mock_engine = MagicMock()
    with patch.object(DatenbankVerbindung, "__init__", return_value=None):
        dbv = DatenbankVerbindung()
        dbv.engine = mock_engine
        dbv.db_schema = "public"

        metadaten = {
            "dataset_name": "iris",
            "beschreibung": "Testdaten",
            "df_tabelle": "iris_df",
            "x_test_tabelle": "iris_X_test",
            "x_train_tabelle": "iris_X_train",
            "y_test_tabelle": "iris_y_test",
            "y_train_tabelle": "iris_y_train"
        }

        with patch("pandas.DataFrame.to_sql") as mock_to_sql:
            dbv.schreibe_metadaten(metadaten)
            mock_to_sql.assert_called_once()
            args, kwargs = mock_to_sql.call_args
            assert kwargs["name"] == "dataframe"
            assert kwargs["schema"] == "public"
