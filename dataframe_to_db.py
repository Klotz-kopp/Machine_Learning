#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
"""
dataframe_to_db.py

Dieses Skript lädt vordefinierte Datasets, bereitet sie auf und speichert sie in einer PostgreSQL-Datenbank.
Die zugehörigen Metadaten werden ebenfalls in einer zentralen Tabelle abgelegt.
"""
import time
from sklearn.model_selection import train_test_split
import pandas as pd
from db import DatenbankVerbindung
from datasets import datasets  # auslagern für Übersicht
import logging


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
        logging.info(f"DatasetPipeline für Dataset '{name}' initialisiert.") # Logging

    def preprocess(self):
        """Optional: spezielle Aufbereitung für bestimmte Datasets."""
        if self.name == 'malware_detect':
            self.df['Label'] = self.df['Label'].astype(int)
            logging.info(f"Preprocessing für Dataset '{self.name}' abgeschlossen.") # Logging

    def split_and_save(self):
        """
        Teilt das Dataset in Trainings- und Testdaten auf und speichert diese in separaten Tabellen.
        """
        X = self.df.drop(self.zielspalte, axis=1)
        y = self.df[self.zielspalte]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Dataframes für die Datenbank
        X_train_df = pd.DataFrame(X_train)
        X_test_df = pd.DataFrame(X_test)
        y_train_df = pd.DataFrame(y_train)
        y_test_df = pd.DataFrame(y_test)

        start = time.time()
        self.db.write_dataframe(X_train_df, self.name + '_X_train')
        self.db.write_dataframe(X_test_df, self.name + '_X_test')
        self.db.write_dataframe(y_train_df, self.name + '_y_train')
        self.db.write_dataframe(y_test_df, self.name + '_y_test')

        dauer = time.time() - start
        logging.info(f"Split & Save für Dataset '{self.name}' abgeschlossen in {dauer:.2f} Sekunden.") # Logging

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
        logging.info(f"Metadaten für Dataset '{self.name}' geschrieben.") # Logging


def main():
    db = DatenbankVerbindung()

    if not db.test_verbindung():
        print("❌ Fehler bei der Datenbankverbindung.")
        logging.critical("Fehler bei der Datenbankverbindung. Das Programm wird beendet.") # Logging
        return

    for name, (df, zielspalte, beschreibung) in datasets.items():
        logging.info(f"=== Verarbeite Dataset: {name} ===") # Logging
        total_start = time.time()
        try:
            pipeline = DatasetPipeline(name, df, zielspalte, beschreibung, db)
            pipeline.preprocess()
            pipeline.split_and_save()
            pipeline.schreibe_metadaten()
            total_dauer = time.time() - total_start
            logging.info(f"Verarbeitung von Dataset '{name}' abgeschlossen in {total_dauer:.2f} Sekunden.") # Logging
        except Exception as e:
            logging.error(f"Fehler bei der Verarbeitung von Dataset '{name}': {e}") # Logging


# -------------------------------
# Einstiegspunkt
# -------------------------------
if __name__ == "__main__":
    main()
