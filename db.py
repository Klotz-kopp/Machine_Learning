#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
# db.py
import configparser
import create_config
import sys
import psycopg2
from base64 import b64decode
from sqlalchemy import create_engine, exc
import pandas as pd
import logging # Import Logging

class DatenbankVerbindung:
    """
        Klasse zur Verwaltung der Verbindung zur PostgreSQL-Datenbank und zur Interaktion mit ihr.
        Liest Konfigurationsdaten aus einer .cfg-Datei, stellt die Verbindung her und stellt Hilfsmethoden bereit.
    """

    def __init__(self, config_pfad='db_config.cfg'):
        self.config = configparser.ConfigParser()
        files_read = self.config.read(config_pfad)

        if not files_read:
            print("Keine Config-Datei gefunden!")
            logging.error("Keine Config-Datei gefunden!") # Logging
            if input("Soll jetzt eine Config-Datei erzeugt werden? (j/n)").lower() == 'j':
                create_config.create_config()
                self.config.read(config_pfad)
                try:
                    create_config.create_dataframe_tabelle(self.config)
                except Exception as e:
                    print(f"Fehler beim Erstellen der Tabelle: {e}")
                    logging.critical(f"Fehler beim Erstellen der Tabelle: {e}") # Logging
                    sys.exit()
                print("Config-Datei und Tabelle wurden erstellt.")
                logging.info("Config-Datei und Tabelle wurden erstellt.") # Logging
            else:
                print("Das Skript wird beendet.")
                logging.critical("Das Skript wird beendet.") # Logging
                sys.exit()

        try:
            self.db_user = self.config['DEFAULT']['db_user']
            self.db_password = b64decode(self.config['SAVE']['db_password']).decode('utf-8')
            self.db_host = self.config['DEFAULT']['db_host']
            self.db_port = self.config['DEFAULT']['db_port']
            self.db_name = self.config['DEFAULT']['db_name']
            self.db_schema = self.config['DEFAULT']['db_schema']

            self.db_url = f"postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
            self.engine = create_engine(self.db_url)
            self.verbindung_erfolgreich = True # Flag setzen
            logging.info("Datenbankverbindung erfolgreich hergestellt.") # Logging
        except Exception as e:
            print(f"Fehler beim Verbindungsaufbau zur Datenbank: {e}")
            logging.critical(f"Fehler beim Verbindungsaufbau zur Datenbank: {e}") # Logging
            self.engine = None
            self.verbindung_erfolgreich = False
            sys.exit() # Skript beenden, da keine DB-Verbindung


    def test_verbindung(self):
        """
        Prüft, ob die Verbindung zur Datenbank besteht.
        :return: True, wenn die Verbindung besteht, False, wenn nicht.
        """
        if not self.verbindung_erfolgreich:
            return False  # Verbindung besteht nicht
        try:
            with self.engine.connect() as conn:
                conn.execute(f"SELECT 1")  # Einfache Abfrage, um die Verbindung zu testen
            logging.info("Datenbankverbindungstest erfolgreich.")  # Logging
            return True
        except exc.DBAPIError as e:
            print(f"Fehler beim Testen der Datenbankverbindung: {e}")
            logging.error(f"Fehler beim Testen der Datenbankverbindung: {e}")  # Logging
            return False
        except Exception as e:
            print(f"Unerwarteter Fehler beim Testen der Datenbankverbindung: {e}")
            logging.critical(f"Unerwarteter Fehler beim Testen der Datenbankverbindung: {e}")  # Logging
            return False

    def test_tabelle(self, tabellenname):
        """
        Prüft, ob eine Tabelle in der Datenbank existiert.
        :param tabellenname: Der Name der Tabelle, die geprüft werden soll.
        :return: True, wenn die Tabelle existiert, False, wenn nicht.
        """
        try:
            with self.engine.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = %s AND table_name = %s)",
                        (self.db_schema, tabellenname)
                    )
                    return cur.fetchone()[0]
        except Exception as e:
            print(f"Fehler bei test_tabelle('{tabellenname}'): {e}")
            logging.error(f"Fehler bei test_tabelle('{tabellenname}'): {e}") # Logging
            return False

    def lade_dataset_metadaten(self):
        """
        Lädt alle Datensätze (Metadaten) aus der Tabelle 'dataframe'.
        :return: Liste von Dictionaries
        """
        try:
            with self.engine.connect() as conn:
                query = f"SELECT dataset_name, beschreibung, df_tabelle, x_test_tabelle, x_train_tabelle, y_test_tabelle, y_train_tabelle FROM {self.db_schema}.dataframe;"
                result = conn.execute(query)
                return [dict(row._mapping) for row in result]  # SQLAlchemy RowProxy → dict
        except Exception as e:
            print(f"Fehler beim Laden der Metadaten: {e}")
            logging.error(f"Fehler beim Laden der Metadaten: {e}") # Logging
            return [] # Leere Liste bei Fehler


    def lade_modelltestergebnisse(self) -> pd.DataFrame:
        """
        Lädt alle Modell-Test-Ergebnisse aus der Tabelle 'modell_tests' und gibt sie als DataFrame zurück.
        :return:
        """
        try:
            sql_query = f"SELECT * FROM {self.db_schema}.modell_tests"
            return pd.read_sql_query(sql_query, self.engine)
        except Exception as e:
            print(f"Fehler beim Laden der Modelltestergebnisse: {e}")
            logging.error(f"Fehler beim Laden der Modelltestergebnisse: {e}") # Logging
            return pd.DataFrame()  # Leerer DataFrame bei Fehler

    def schreibe_dataframe(self, df: pd.DataFrame, tabellenname: str):
        """
        Speichert einen Pandas DataFrame in der Datenbank. Überschreibt die Tabelle, falls sie existiert.

        :param df: Der Pandas DataFrame, der gespeichert werden soll.
        :param tabellenname: Der Name der Datenbanktabelle.
        """
        try:
            df.to_sql(name=tabellenname, con=self.engine, if_exists='replace', index=False, schema=self.db_schema)
            logging.info(f"DataFrame erfolgreich in Tabelle '{tabellenname}' geschrieben.") # Logging
        except Exception as e:
            print(f"Fehler beim Schreiben des DataFrames in die Tabelle '{tabellenname}': {e}")
            logging.error(f"Fehler beim Schreiben des DataFrames in die Tabelle '{tabellenname}': {e}") # Logging

    def schreibe_metadaten(self, metadaten: dict):
        """
        Schreibt Metadaten über einen Datensatz in die Tabelle 'dataframe'.

        :param metadaten: Ein Dictionary mit den Metadaten.
        """
        try:
            df = pd.DataFrame([metadaten])
            df.to_sql(name='dataframe', con=self.engine, if_exists='append', index=False, schema=self.db_schema)
            logging.info(f"Metadaten für Dataset '{metadaten['dataset_name']}' erfolgreich geschrieben.") # Logging
        except Exception as e:
            print(f"Fehler beim Schreiben der Metadaten für Dataset '{metadaten['dataset_name']}': {e}")
            logging.error(f"Fehler beim Schreiben der Metadaten für Dataset '{metadaten['dataset_name']}': {e}") # Logging
