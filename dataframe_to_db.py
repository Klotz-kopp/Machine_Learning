#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
import sys
import pandas as pd
import psycopg2
import time
import configparser
import seaborn as sns
import sklearn.datasets
import create_config

from funktionen import printf
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sqlalchemy import create_engine
from base64 import b64decode

config = configparser.ConfigParser()
files_read = config.read('db_config.cfg')
if not files_read:
    print("Keine Config-Datei gefunden!")
    if input("Soll jetzt eine Config-Datei erzeugt werden? (j/n)").lower() == 'j':
        create_config.create_config()  # Config erstellen
        print("Config-Datei wurde erstellt.")
        # Nach der Erstellung die Config-Datei neu laden
        config.read('db_config.cfg')
        print("Config-Datei neu geladen. Das Skript läuft jetzt weiter...")
    else:
        print("Das Skript wird beendet.")
        sys.exit()  # Skript beenden, falls der Benutzer 'n' eingibt


# Verbindungsdaten aus der Config lesen
try:
    db_user = config['DEFAULT']['db_user']
    db_password = b64decode(config['SAVE']['db_password']).decode('utf-8')
    db_host = config['DEFAULT']['db_host']
    db_port = int(config['DEFAULT']['db_port'])
    db_name = config['DEFAULT']['db_name']
    db_schema = config['DEFAULT']['db_schema']
except:
    print("Fehler beim lesen der Config!")
    sys.exit()

# Verbindungs-URL zusammenbauen
db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# Engine erstellen
engine = create_engine(db_url)

# TODO iteration über alle Datasets {name : (Import, Target_Spalte)} um vollautomatisch die die Dataframes zu erstellen und in der pg Datenbank speichern.
# TODO prüfen ob eine Aufteilung auf mehrere Dateien sinnvoll ist.
# TODO eigenes skript erstellen um die config.cfg zu erstellen

datasets = {
    'malware_detect' : (pd.read_csv('../Datasets/TUANDROMD.csv'), 'Label'),
    'penguins': (sns.load_dataset('penguins'),'species' ),
    'brustkrebs' : (sklearn.datasets.load_breast_cancer(), 'Diagnosis'),
    'iris' : (sklearn.datasets.load_iris(), 'class')
}




datenname = 'malware_detect'
df_name = datenname + '_df'
split_name = datenname + '_split.pkl'


def dataframe_erzeugen():
    df = dataframe_bauen()
    df.to_sql(
        name=df_name,
        con=engine,
        if_exists='replace',
        index=False,
        schema=db_schema,
    method='multi',
    chunksize=1000
    )

def test_train_erzeuge():
    df = pd.read_sql_table(df_name, con=engine, schema=db_schema)
    X_train, X_test, y_train, y_test = dataframe_praeparieren(df)
    X_train.to_sql(
        name=datenname + "_X_train",
        con=engine,
        if_exists='replace',
        index=False,
        schema=db_schema,
    method='multi',
    chunksize=1000
    )
    X_test.to_sql(
        name=datenname + "_X_test",
        con=engine,
        if_exists='replace',
        index=False,
        schema=db_schema,
    method='multi',
    chunksize=1000
    )
    y_train.to_sql(
        name=datenname + "_y_train",
        con=engine,
        if_exists='replace',
        index=False,
        schema=db_schema,
    method='multi',
    chunksize=1000
    )
    y_test.to_sql(
        name=datenname + "_y_test",
        con=engine,
        if_exists='replace',
        index=False,
        schema=db_schema,
    method='multi',
    chunksize=1000
    )


def dataframe_bauen():
    # Dataframe definieren
    start = time.time()  # Startzeitpunkt
    df = pd.read_csv('../Datasets/TUANDROMD.csv')
    # Null-Zeilen entfernen
    df = df.dropna()
    if datenname == 'penguins':
        # umwandeln von nicht nummerischen Werten in nummerische Werte
        # Geschlecht männlich in 0 und Geschlecht weiblich in 1 umwandeln
        df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
        # Umwandeln der Inselnamen in Nummern 1 bis 3
        df['island'] = df['island'].map({'Torgersen': 1, 'Biscoe': 2, 'Dream': 3})
        df['species'] = df['species'].map({'Adelie': 1, 'Chinstrap': 2, 'Gentoo': 3})
    dauer = time.time() - start
    printf(f"Dataframe in {dauer:.3f} Sekunden erzeugt.")
    df.info()
    return df



def dataframe_praeparieren(df):
    start = time.time()  # Startzeitpunkt
    # Features definiert
    X = df.drop(["Label"], axis=1)
    # Labels definiert
    y = df["Label"]
    # Dataframe in Test und Trainingsdaten aufgeteilt
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=119)
    dauer = time.time() - start
    printf(f"Dataframe präpariert in {dauer:.3f} Sekunden.")
    return X_train, X_test, y_train, y_test


def postgres_test():
    try:
        with psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        ) as conn:
            return True
    except psycopg2.Error as e:
        print(f"Postgres-Verbindungsfehler: {e}")
        return False


def df_test():
    try:
        with psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = %s AND table_name = %s)",
                    (db_schema, df_name)
                )
                exists = cur.fetchone()[0]
                return exists
    except Exception as e:
        print(f"Fehler bei df_test: {e}")
        return False


def main():
    if postgres_test():
        total_start = time.time()  # Startzeitpunkt
        dataframe_erzeugen()
        if df_test():
            test_train_erzeuge()
            total_dauer = time.time() - total_start
            printf(f"Alle Aufgaben in {total_dauer:.3f} Sekunden abgearbeitet.")
        else:
            print("Dataframe Tabelle nicht gefunden.")
    else:
        print("Fehler bei der Datenbankverbindung.")

if __name__ == "__main__":
    main()
