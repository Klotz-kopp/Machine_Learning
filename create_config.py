#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
# Imports
import configparser
from base64 import b64encode


def create_config():
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
    print("Config-Datei erfolgreich erstellt!")
