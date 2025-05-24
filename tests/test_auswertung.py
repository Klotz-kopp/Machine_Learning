import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from auswertung import Auswertung

# Dummy-Daten
DUMMY_DF = pd.DataFrame({
    "modellname": ["ModelA", "ModelA", "ModelB"],
    "datenname": ["Dataset1", "Dataset1", "Dataset1"],
    "durchgang": [1, 2, 1],
    "score": [91.5, 92.7, 88.3],
    "f1": [89.1, 90.2, 87.0],
    "dauer": [1.1, 1.0, 0.8],
    "laufzeit": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
    "cm": ["[[5, 0], [1, 4]]"] * 3,
})

# ------------------------
# Setup für Tests
# ------------------------
@pytest.fixture
def auswertung_instanz():
    with patch("auswertung.DatenbankVerbindung") as mock_db:
        mock_db().lade_modelltestergebnisse.return_value = DUMMY_DF.copy()
        return Auswertung()

# ------------------------
# Tests
# ------------------------

def test_init_laedt_daten(auswertung_instanz):
    assert isinstance(auswertung_instanz.df, pd.DataFrame)
    assert not auswertung_instanz.df.empty

@patch("pandas.DataFrame.to_csv")
@patch("auswertung.pruefe_und_erstelle_ordner")
def test_speichere_gruppierte_ergebnisse_csv(mock_ordner, mock_csv, auswertung_instanz):
    auswertung_instanz.speichere_gruppierte_ergebnisse_csv("modellname", ordner="TestExport")
    assert mock_csv.called
    assert mock_ordner.called

@patch("matplotlib.pyplot.savefig")
@patch("auswertung.plt.bar")
@patch("auswertung.plt.figure")
def test_plot_beste_scores(mock_fig, mock_bar, mock_save, auswertung_instanz):
    auswertung_instanz.plot_beste_scores(wert="score", gruppiert_nach="modellname", ordner="TestPlot")
    assert mock_bar.called
    assert mock_save.called

@patch("matplotlib.pyplot.savefig")
@patch("auswertung.plt.bar")
@patch("auswertung.plt.figure")
def test_plot_schnellste_durchlaeufe(mock_fig, mock_bar, mock_save, auswertung_instanz):
    auswertung_instanz.plot_schnellste_durchlaeufe(gruppiert_nach="modellname", ordner="TestPlot")
    assert mock_bar.called
    assert mock_save.called

@patch("matplotlib.pyplot.savefig")
@patch("auswertung.sns.scatterplot")
@patch("auswertung.plt.figure")
def test_ranking_plot(mock_fig, mock_scatter, mock_save, auswertung_instanz):
    auswertung_instanz.ranking_plot(wert="f1", gruppiert_nach="modellname", ordner="TestPlot")
    assert mock_scatter.called
    assert mock_save.called

@patch("builtins.open")
@patch("os.listdir", return_value=["ranking_modellname_f1.png"])
@patch("auswertung.pruefe_und_erstelle_ordner")
def test_generiere_html_report(mock_ordner, mock_ls, mock_open, auswertung_instanz):
    auswertung_instanz.generiere_html_report(gruppiert_nach="modellname", ordner="TestExport")
    assert mock_open.called
