#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
"""
dataframe_to_db.py

Dieses Skript lädt vordefinierte Datasets, bereitet sie auf und speichert sie in einer PostgreSQL-Datenbank.
Die zugehörigen Metadaten werden ebenfalls in einer zentralen Tabelle abgelegt.
"""
import time
from sklearn.model_selection import train_test_split
from utils import printf
from db import DatenbankVerbindung
from datasets import datasets  # auslagern für Übersicht
import pandas as pd


# DONE iteration über alle Datasets {name : (Import, Target_Spalte, Beschreibung)} um vollautomatisch die die Dataframes zu erstellen und in der pg Datenbank speichern.
# DONE prüfen ob eine Aufteilung auf mehrere Dateien sinnvoll ist.
# DONE eigenes skript erstellen um die config.cfg zu erstellen

class DatasetPipeline:
    """
    Verarbeitet ein Dataset: importiert, bereinigt, splittet, speichert es in die Datenbank und pflegt Metadaten.
    """
    def __init__(self, name, df, zielspalte, beschreibung, db: DatenbankVerbindung):
        self.name = name
        self.df = df.dropna()
        self.zielspalte = zielspalte
        self.beschreibung = beschreibung
        self.db = db
        self.df_name = name + "_df"

    def preprocess(self):
        """Optional: spezielle Aufbereitung für bestimmte Datasets."""
        if self.name == 'penguins':
            self.df['sex'] = self.df['sex'].map({'Male': 0, 'Female': 1})
            self.df['island'] = self.df['island'].map({'Torgersen': 1, 'Biscoe': 2, 'Dream': 3})
            self.df['species'] = self.df['species'].map({'Adelie': 1, 'Chinstrap': 2, 'Gentoo': 3})

    def split_and_store(self):
        """Splitte DataFrame in Trainings-/Testdaten und speichere alle Tabellen in die Datenbank."""
        start = time.time()
        X = self.df.drop(columns=[self.zielspalte])
        y = self.df[self.zielspalte]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=119)

        self.db.write_dataframe(self.df, self.df_name)
        self.db.write_dataframe(X_train, self.name + '_X_train')
        self.db.write_dataframe(X_test, self.name + '_X_test')
        self.db.write_dataframe(y_train, self.name + '_y_train')
        self.db.write_dataframe(y_test, self.name + '_y_test')

        dauer = time.time() - start
        printf(f"{self.name}: Split & Save abgeschlossen in {dauer:.2f} Sekunden.")

    def schreibe_metadaten(self):
        """Speichert Metadaten zur Datenstruktur in einer zentralen Tabelle."""
        meta = {
            'dataset_name': self.name,
            'beschreibung': self.beschreibung,
            'df_tabelle': self.df_name,
            'x_test_tabelle': self.name + '_X_test',
            'x_train_tabelle': self.name + '_X_train',
            'y_test_tabelle': self.name + '_y_test',
            'y_train_tabelle': self.name + '_y_train'
        }
        self.db.schreibe_metadaten(meta)


def main():
    db = DatenbankVerbindung()

    if not db.test_verbindung():
        print("❌ Fehler bei der Datenbankverbindung.")
        return

    for name, (df, zielspalte, beschreibung) in datasets.items():
        printf(f"\n=== Verarbeite Dataset: {name} ===")
        total_start = time.time()

        pipeline = DatasetPipeline(name, df, zielspalte, beschreibung, db)
        pipeline.preprocess()
        pipeline.split_and_store()
        pipeline.schreibe_metadaten()

        total_dauer = time.time() - total_start
        printf(f"✔ {name}: Komplett abgeschlossen in {total_dauer:.2f} Sekunden.")


if __name__ == "__main__":
    main()
