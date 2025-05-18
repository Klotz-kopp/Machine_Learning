#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
# Imports
import configparser
from base64 import b64encode, b64decode
from sqlalchemy import create_engine
import logging

def create_config():
    """
    Erstellt eine Konfigurationsdatei (db_config.cfg) für die Datenbankverbindung.
    """
    config = configparser.ConfigParser()

    db_user = input("Bitte den Usernamen für die Datenbank angeben: ")
    db_password = b64encode(input("Bitte das Passwort angeben: ").encode('utf-8')).decode('utf-8')
    db_host = input("Bitte die Serveradresse (IP, Domain, localhost) angeben: ")
    db_port = input("Bitte den Port zur Datenbank angeben: ")
    db_name = input("Bitte den Namen der Datenbank angeben: ")
    db_schema = input("Bitte das Schema angeben: ")

    config['DEFAULT'] = {
        'db_user': db_user,
        'db_host': db_host,
        'db_port': db_port,
        'db_name': db_name,
        'db_schema': db_schema
    }
    config['SAVE'] = {
        'db_password': db_password
    }

    with open('db_config.cfg', 'w') as configfile:
        config.write(configfile)
    logging.info("Config-Datei erfolgreich erstellt!") # Logging


def create_dataframe_tabelle(config):
    """
    Erstellt die Tabelle 'dataframe' in der Datenbank, falls sie noch nicht existiert.
    """
    db_user = config['DEFAULT']['db_user']
    db_password = b64decode(config['SAVE']['db_password']).decode('utf-8')
    db_host = config['DEFAULT']['db_host']
    db_port = config['DEFAULT']['db_port']
    db_name = config['DEFAULT']['db_name']
    db_schema = config['DEFAULT']['db_schema']

    db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(db_url)

    try:
        with engine.begin() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {db_schema}.dataframe (
                    dataset_name TEXT PRIMARY KEY,
                    beschreibung TEXT,
                    df_tabelle TEXT,
                    x_test_tabelle TEXT,
                    x_train_tabelle TEXT,
                    y_test_tabelle TEXT,
                    y_train_tabelle TEXT
                )
            """)
        logging.info("Tabelle 'dataframe' erfolgreich geprüft/erstellt.") # Logging
    except Exception as e:
        logging.error(f"Fehler beim Erstellen der Tabelle 'dataframe': {e}") # Logging
        raise
