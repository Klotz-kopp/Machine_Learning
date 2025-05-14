#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
from db import DatenbankVerbindung
db = DatenbankVerbindung()
df = db.lade_modelltestergebnisse()

"""Beste Ergebnisse als Plot
Schnellste Durchgänge als Plot
Ergebnisse als CSV speichern
Visualisierung der besten Modelle nach Score
Visualisierung der schnellsten Modelle nach Dauer
Übersicht mit Ergebnissen und Grafiken je Modell
Übersicht mit Ergebnissen und Grafiken je Dataset"""

class Auswertung:
    def __init__(self):
        self.db = DatenbankVerbindung()
        self.df = self.db.lade_modelltestergebnisse()

    def speichere_csv(self, pfad="Modell_Ergebnisse.csv"):
        pass

    def plot_beste_ergebnisse(self):
        pass

    def plot_schnellste_ergebnisse(self):
        pass

    def plot_score_ranking(self):
        pass

    def plot_dauer_ranking(self):
        pass

    def uebersicht_je_modell(self):
        pass

    def uebersicht_je_dataset(self):
        pass
