#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
# db.py
import configparser
import create_config
import sys
import psycopg2
from base64 import b64decode
from sqlalchemy import create_engine

class DatenbankVerbindung:
    def __init__(self, config_pfad='db_config.cfg'):
        self.config = configparser.ConfigParser()
        files_read = self.config.read(config_pfad)

        if not files_read:
            print("Keine Config-Datei gefunden!")
            if input("Soll jetzt eine Config-Datei erzeugt werden? (j/n)").lower() == 'j':
                create_config.create_config()
                self.config.read(config_pfad)
                create_config.create_dataframe_tabelle(self.config)
                print("Config-Datei und Tabelle wurden erstellt.")
            else:
                print("Das Skript wird beendet.")
                sys.exit()

        try:
            self.db_user = self.config['DEFAULT']['db_user']
            self.db_password = b64decode(self.config['SAVE']['db_password']).decode('utf-8')
            self.db_host = self.config['DEFAULT']['db_host']
            self.db_port = int(self.config['DEFAULT']['db_port'])
            self.db_name = self.config['DEFAULT']['db_name']
            self.db_schema = self.config['DEFAULT']['db_schema']
        except KeyError as e:
            print(f"Fehlender Config-Wert: {e}")
            sys.exit()

        # Engine erzeugen
        db_url = f"postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        self.engine = create_engine(db_url)


    def get_engine(self):
        return self.engine


    def get_schema(self):
        return self.db_schema


    def get_config(self):
        return self.config


    def write_dataframe(self, df, name, replace=True):
        """Hilfsfunktion zum Schreiben eines DataFrames in die Datenbank."""
        df.to_sql(
            name=name,
            con=self.engine,
            schema=self.db_schema,
            if_exists='replace' if replace else 'append',
            index=False,
            method='multi',
            chunksize=1000
        )

    def schreibe_metadaten(self, metadaten: dict):
        """
        Fügt die Metadaten in die Tabelle dataframe ein.
        Bereits vorhandene Dataset-Einträge (gleicher Primary Key) werden ignoriert.
        """
        with self.engine.begin() as conn:
            # INSERT Statement mit Konfliktbehandlung
            conn.execute(
                f"""
                INSERT INTO {self.db_schema}.dataframe
                    (dataset_name, beschreibung, df_tabelle, x_test_tabelle,
                     x_train_tabelle, y_test_tabelle, y_train_tabelle)
                VALUES (%(dataset_name)s, %(beschreibung)s, %(df_tabelle)s, %(x_test_tabelle)s,
                        %(x_train_tabelle)s, %(y_test_tabelle)s, %(y_train_tabelle)s)
                ON CONFLICT (dataset_name) DO NOTHING;
                """,
                metadaten
            )

        print(f"Metadaten für '{metadaten['dataset_name']}' gespeichert (oder übersprungen, falls bereits vorhanden).")


    def test_verbindung(self):
        """Testet, ob eine Verbindung zur PostgreSQL-Datenbank aufgebaut werden kann."""
        try:
            with psycopg2.connect(
                    host=self.db_host,
                    database=self.db_name,
                    user=self.db_user,
                    password=self.db_password,
                    port=self.db_port
            ) as conn:
                return True
        except psycopg2.Error as e:
            print(f"Postgres-Verbindungsfehler: {e}")
            return False


    def test_tabelle(self, tabellenname: str):
        """Prüft, ob eine Tabelle im angegebenen Schema existiert."""
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
            print(f"Fehler bei test_tabelle('{tabellenname}'): {e}")
            return False