#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.

import pandas as pd
from funktionen import printf
import seaborn as sns
import pickle
import time
from sklearn.model_selection import train_test_split

datenname = 'malware_detect'
df_name = datenname + '_df.pkl'
split_name = datenname + '_split.pkl'


def dataframe_erzeugen():
    # DataFrame einmalig erzeugen und speichern
    df = dataframe_bauen()
    with open(df_name, "wb") as f:
        pickle.dump(df, f)


def test_train_erzeuge():
    with open(df_name, "rb") as f:
        df = pickle.load(f)
    X_train, X_test, y_train, y_test = dataframe_praeparieren(df)
    with open(split_name, "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)


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
    print(df.info)
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


total_start = time.time()  # Startzeitpunkt
dataframe_erzeugen()
test_train_erzeuge()
total_dauer = time.time() - total_start
printf(f"Alle Aufgaben in {total_dauer:.3f} Sekunden abgearbeitet.")
