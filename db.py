#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
# db.py
import configparser
import create_config
import sys
import psycopg2
from base64 import b64decode
from sqlalchemy import create_engine
import pandas as pd
import logging

class DatenbankVerbindung:
    """
    Klasse zur Verwaltung der Verbindung zur PostgreSQL-Datenbank und zur Interaktion mit ihr.
    Liest Konfigurationsdaten aus einer .cfg-Datei, stellt die Verbindung her und stellt Hilfsmethoden bereit.
    """

    def __init__(self, config_pfad='db_config.cfg'):
        self.config = configparser.ConfigParser()
        files_read = self.config.read(config_pfad)

        if not files_read:
            logging.warning("Keine Config-Datei gefunden!")  # Verwende logging
            if input("Soll jetzt eine Config-Datei erzeugt werden? (j/n)").lower() == 'j':
                create_config.create_config()
                self.config.read(config_pfad)
                create_config.create_dataframe_tabelle(self.config)
                logging.info("Config-Datei und Tabelle wurden erstellt.")  # Verwende logging
            else:
                logging.critical("Das Skript wird beendet.")  # Verwende logging
                sys.exit()

        try:
            self.db_user = self.config['DEFAULT']['db_user']
            self.db_password = b64decode(self.config['SAVE']['db_password']).decode('utf-8')
            self.db_host = self.config['DEFAULT']['db_host']
            self.db_port = self.config['DEFAULT']['db_port']
            self.db_name = self.config['DEFAULT']['db_name']
            self.db_schema = self.config['DEFAULT']['db_schema']

            self.engine = create_engine(
                f"postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
            )

        except KeyError as e:
            logging.error(
                f"Fehler beim Lesen der Config-Datei: {e}. Bitte überprüfen Sie die Struktur.")  # Verwende logging
            sys.exit()

    def test_verbindung(self) -> bool:
        """
        Testet, ob eine Verbindung zur Datenbank hergestellt werden kann.
        :return: True bei Erfolg, False bei Misserfolg
        """
        try:
            with psycopg2.connect(
                    host=self.db_host,
                    database=self.db_name,
                    user=self.db_user,
                    password=self.db_password,
                    port=self.db_port
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return True
        except psycopg2.Error as e:
            logging.error(f"Fehler bei der Datenbankverbindung: {e}")  # Verwende logging
            return False

    def schreibe_metadaten(self, meta: dict):
        """
        Schreibt die Metadaten eines Datasets in die Tabelle 'dataframe'.
        :param meta: Dictionary mit Metadaten
        """
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    f"""
                    INSERT INTO {self.db_schema}.dataframe (dataset_name, beschreibung, df_tabelle, x_test_tabelle, x_train_tabelle, y_test_tabelle, y_train_tabelle)
                    VALUES (%(dataset_name)s, %(beschreibung)s, %(df_tabelle)s, %(x_test_tabelle)s, %(x_train_tabelle)s, %(y_test_tabelle)s, %(y_train_tabelle)s)
                    ON CONFLICT (dataset_name) DO UPDATE
                    SET beschreibung = %(beschreibung)s,
                        df_tabelle = %(df_tabelle)s,
                        x_test_tabelle = %(x_test_tabelle)s,
                        x_train_tabelle = %(x_train_tabelle)s,
                        y_test_tabelle = %(y_test_tabelle)s,
                        y_train_tabelle = %(y_train_tabelle)s
                    """,
                    meta
                )
        except Exception as e:
            logging.error(f"Fehler beim Schreiben der Metadaten: {e}")  # Verwende logging

    def write_dataframe(self, df: pd.DataFrame, tabellenname: str):
        """
        Schreibt einen Pandas DataFrame in die Datenbank.
        :param df: Der zu schreibende DataFrame
        :param tabellenname: Der Name der Tabelle
        """
        try:
            df.to_sql(
                name=tabellenname,
                con=self.engine,
                if_exists='replace',  # oder 'append', je nach Bedarf
                index=False,
                schema=self.db_schema,
                chunksize=1000,
                method='multi'
            )
            logging.info(
                f"DataFrame erfolgreich in Tabelle '{tabellenname}' geschrieben.")  # Verwende logging
        except Exception as e:
            logging.error(
                f"Fehler beim Schreiben des DataFrames in Tabelle '{tabellenname}': {e}")  # Verwende logging

    def test_tabelle(self, tabellenname: str) -> bool:
        """
        Prüft, ob eine Tabelle in der Datenbank existiert.
        :param tabellenname: Der Name der Tabelle
        :return: True, wenn die Tabelle existiert, False sonst
        """
        try:
            with psycopg2.connect(
                    host=self.db_host,
                    database=self.db_name,
                    user=self.db_user,
                    password=self.db_password,
                    port=self.db_port
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = %s AND table_name = %s)",
                        (self.db_schema, tabellenname)
                    )
                    return cur.fetchone()[0]
        except Exception as e:
            logging.error(f"Fehler bei test_tabelle('{tabellenname}'): {e}")  # Verwende logging
            return False

    def lade_dataset_metadaten(self):
        """
        Lädt alle Datensätze (Metadaten) aus der Tabelle 'dataframe'.
        :return: Liste von Dictionaries
        """
        with self.engine.connect() as conn:
            query = f"SELECT dataset_name, beschreibung, df_tabelle, x_test_tabelle, x_train_tabelle, y_test_tabelle, y_train_tabelle FROM {self.db_schema}.dataframe;"
            result = conn.execute(query)
            return [dict(row._mapping) for row in result]  # SQLAlchemy RowProxy → dict

    def lade_modelltestergebnisse(self) -> pd.DataFrame:
        """
        Lädt alle Modell-Test-Ergebnisse aus der Tabelle 'modell_tests' und gibt sie als DataFrame zurück.
        """

        return pd.read_sql_table('modell_tests', con=self.engine, schema=self.db_schema)