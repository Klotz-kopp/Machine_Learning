#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
import pandas as pd
import time
import seaborn as sns

from utils import printf
from db import DatenbankVerbindung
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris


# TODO iteration über alle Datasets {name : (Import, Target_Spalte, Beschreibung)} um vollautomatisch die die Dataframes zu erstellen und in der pg Datenbank speichern.
# TODO prüfen ob eine Aufteilung auf mehrere Dateien sinnvoll ist.
# DONE eigenes skript erstellen um die config.cfg zu erstellen

datasets = {
    'malware_detect' : (pd.read_csv('Datasets/TUANDROMD.csv'), 'Label', "The target attribute for classification is a category (malware vs goodware)."),
    'penguins': (sns.load_dataset('penguins'),'species', "The goal of palmerpenguins is to provide a great dataset for data exploration & visualization, as an alternative to iris." ),
    'brustkrebs' : (load_breast_cancer(), 'Diagnosis', "Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image."),
    'iris' : (load_iris(), 'class', "The famous Iris database, first used by Sir R.A. Fisher")
}

# Objekt erzeugen
db = DatenbankVerbindung()

# Engine/Schema wie gewohnt
engine = db.get_engine()
db_schema = db.get_schema()
datenname = 'malware_detect'
df_name = datenname + '_df'
split_name = datenname + '_split.pkl'


def dataframe_erzeugen():
    df = dataframe_bauen()
    db.write_dataframe(df, df_name)


def test_train_erzeuge():
    # Lade den vollständigen DataFrame aus der Datenbank
    df = pd.read_sql_table(df_name, con=db.get_engine(), schema=db.get_schema())

    # Split in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = dataframe_praeparieren(df)

    # Schreibe alle vier Teildatensätze über die zentrale DB-Methode
    db.write_dataframe(X_train, name=datenname + "_X_train")
    db.write_dataframe(X_test, name=datenname + "_X_test")
    db.write_dataframe(y_train, name=datenname + "_y_train")
    db.write_dataframe(y_test, name=datenname + "_y_test")



def dataframe_bauen():
    # Dataframe definieren
    start = time.time()  # Startzeitpunkt
    df = pd.read_csv('Datasets/TUANDROMD.csv')
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


def sammle_metadaten(dataset_name: str, beschreibung: str) -> dict:
    """
    Erstellt ein Dictionary mit Metadaten für das gegebene Dataset.
    Die Tabellennamen folgen dem Namensschema: datasetname_X_train usw.
    """
    return {
        'dataset_name': dataset_name,
        'beschreibung': beschreibung,
        'df_tabelle': dataset_name + "_df",
        'x_test_tabelle': dataset_name + "_X_test",
        'x_train_tabelle': dataset_name + "_X_train",
        'y_test_tabelle': dataset_name + "_y_test",
        'y_train_tabelle': dataset_name + "_y_train"
    }



def main():
    if db.test_verbindung():
        total_start = time.time()  # Startzeitpunkt
        dataframe_erzeugen()
        if db.test_tabelle(df_name):
            test_train_erzeuge()
            total_dauer = time.time() - total_start
            printf(f"Alle Aufgaben in {total_dauer:.3f} Sekunden abgearbeitet.")
        else:
            print("Dataframe Tabelle nicht gefunden.")
    else:
        print("Fehler bei der Datenbankverbindung.")


if __name__ == "__main__":
    main()
