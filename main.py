#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
from db import DatenbankVerbindung
import pandas as pd

db = DatenbankVerbindung()
df = db.lade_modelltestergebnisse()
df = df.drop(columns='id')
df['laufzeit'] = pd.to_datetime(df['laufzeit'])

modelname = []
datenname = []
modellnamen = df['modellname'].unique().tolist()
datennamen = df['datenname'].unique().tolist()

def main():
    menu()
    pass


def menu():
    # Dataframe (Vorbereitungen)
    # TODO Ausgabe der hardcodierten Datasets
    # TODO vorhandene Dataframes abfragen
    # TODO Dataframe erzeugen
    # Modelle testen
    # TODO Modelltest mit einem Dataframe starten
    # TODO Modelltest mit allen Dataframes starten
    # TODO Auswertungen anzeigen
    pass
