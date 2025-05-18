#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
# model_test.py
# Führt das Training & die Evaluierung aller ML-Modelle für alle gespeicherten Datasets durch.
# Ergebnisse werden in der PostgreSQL-Datenbank gespeichert – Auswertungen erfolgen in separatem Modul.

import sys
import time
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime

from db import DatenbankVerbindung
from utils import printf, zeit_messen
import logging # Import Logging

# -------------------------------
# Klasse für ein ML-Modell mit dynamischer Parametrisierung (z.B. Nachbarn, max_iter etc.)
# -------------------------------


class MLModell:
    def __init__(self, name, model_func):
        self.name = name
        self.model_func = model_func
        self.model = None
        self.score = 0

    def train(self, X_train, y_train, i):
        try:
            self.model = self.model_func(i)  # Modellinstanz mit aktuellem Parameter erzeugen
            self.model.fit(X_train, y_train)
        except Exception as e:
            raise RuntimeError(f"Fehler beim Training des Modells {self.name}: {e}") from e

    def testen(self, X_test, y_test, i):
        try:
            predictions = self.model.predict(X_test)
            self.score = accuracy_score(y_test, predictions) * 100
            return self.score
        except Exception as e:
            raise RuntimeError(f"Fehler beim Testen des Modells {self.name}: {e}") from e


# -------------------------------
# Liste aller zu testenden ML-Modelle
# -------------------------------
Modelle = [
    MLModell('KNeighbors', lambda i: KNeighborsClassifier(n_neighbors=i)),
    MLModell('RandomForest', lambda i: RandomForestClassifier(n_estimators=i)),
    MLModell('LogisticRegression', lambda i: LogisticRegression(max_iter=1000 * i)),
    MLModell('SVC', lambda i: SVC()),
    MLModell('DecisionTree', lambda i: DecisionTreeClassifier()),
    MLModell('GradientBoost', lambda i: GradientBoostingClassifier()),
    MLModell('GaussianNB', lambda i: GaussianNB()),
    MLModell('MLPClassifier', lambda i: MLPClassifier(max_iter=1000 * i))
]


# -------------------------------
# Hauptfunktion: Trainings- und Testlauf für alle Datasets und Modelle
# -------------------------------
@zeit_messen
def main():
    db = DatenbankVerbindung()
    if not db.test_verbindung():
        print("Keine Verbindung zur Datenbank. Programm wird beendet.")
        logging.critical("Keine Verbindung zur Datenbank. Programm wird beendet.")
        return

    engine = db.get_engine()
    schema = db.get_schema()
    metadaten_liste = db.lade_dataset_metadaten()

    for eintrag in metadaten_liste:
        datenname = eintrag['dataset_name']
        printf(f"\n=== Starte Modellvergleich für Dataset: {datenname} ===")
        logging.info(f"=== Starte Modellvergleich für Dataset: {datenname} ===")

        # Lade Trainings- und Testdaten aus der Datenbank
        try:
            X_train = pd.read_sql_table(eintrag['x_train_tabelle'], con=engine, schema=schema)
            X_test = pd.read_sql_table(eintrag['x_test_tabelle'], con=engine, schema=schema)
            y_train = pd.read_sql_table(eintrag['y_train_tabelle'], con=engine, schema=schema)
            y_test = pd.read_sql_table(eintrag['y_test_tabelle'], con=engine, schema=schema)
        except Exception as e:
            logging.error(f"Fehler beim Laden der Daten für Dataset {datenname}: {e}")
            print(f"Fehler beim Laden der Daten für Dataset {datenname}: {e}")
            continue  # Gehe zum nächsten Dataset über

        # Initialisiere leeres Ergebnis-Dictionary für dieses Dataset
        Ergebnisse = {modell.name + '_ergebnis': [] for modell in Modelle}

        # 10 Durchläufe pro Modell mit steigender Komplexität (z.B. Nachbarn, max_iter)
        for i in range(1, 11):
            for modell in Modelle:
                start = time.time()
                try:
                    modell.train(X_train, y_train, i)
                    score = modell.testen(X_test, y_test, i)
                    dauer = time.time() - start
                    Ergebnisse[modell.name + '_ergebnis'].append((i, score, dauer))
                    logging.info(f"Modell {modell.name}, Durchgang {i}, Score: {score:.2f}, Dauer: {dauer:.2f}s")
                except RuntimeError as e:
                    logging.error(f"Fehler bei Modell {modell.name}, Durchgang {i}: {e}")
                    print(f"Fehler bei Modell {modell.name}, Durchgang {i}: {e}")
                    Ergebnisse[modell.name + '_ergebnis'].append((i, 0, time.time() - start)) #Ergebnis festhalten
                    continue  # Gehe zum nächsten Modell über
            printf(f"Durchgang {i} abgeschlossen.")
            logging.info(f"Durchgang {i} abgeschlossen.")

        # Ergebnisse in zentrale Tabelle schreiben
        try:
            speichere_ergebnisse_in_datenbank(Ergebnisse, datenname, db)
        except Exception as e:
            logging.error(f"Fehler beim Speichern der Ergebnisse für Dataset {datenname}: {e}")
            print(f"Fehler beim Speichern der Ergebnisse für Dataset {datenname}: {e}")



# -------------------------------
# Speichert alle Testergebnisse (je Modell, Durchgang) in PostgreSQL-Tabelle `modell_tests`
# -------------------------------
@zeit_messen
def speichere_ergebnisse_in_datenbank(Ergebnisse, datenname, db):
    engine = db.get_engine()
    schema = db.get_schema()

    # Sicherstellen, dass die Tabelle existiert
    try:
        erstelle_modell_tests_tabelle(engine, schema)
    except Exception as e:
        logging.critical(f"Fehler beim Erstellen der Tabelle 'modell_tests': {e}")
        print(f"Fehler beim Erstellen der Tabelle 'modell_tests': {e}")
        raise  # Re-raise, damit die übergeordnete Funktion es auch mitbekommt



    # Alle Ergebnisse in flache Liste für DataFrame umwandeln
    aktuelle_zeit = datetime.now()
    datensaetze = []

    for key, werte in Ergebnisse.items():
        modellname = key.replace('_ergebnis', '')
        for durchgang, score, dauer in werte:
            datensaetze.append({
                'modellname': modellname,
                'score': score,
                'durchgang': durchgang,
                'dauer': dauer,
                'datenname': datenname,
                'laufzeit': aktuelle_zeit
            })

    df_gesamt = pd.DataFrame(datensaetze)

    # In Datenbank anhängen
    try:
        df_gesamt.to_sql(
            name='modell_tests',
            con=engine,
            if_exists='append',
            index=False,
            schema=schema,
            chunksize=1000,
            method='multi'
        )
        printf(f"Testergebnisse für '{datenname}' gespeichert.")
        logging.info(f"Testergebnisse für '{datenname}' gespeichert.")
    except Exception as e:
        logging.error(f"Fehler beim Schreiben der Testergebnisse in die Datenbank für Dataset {datenname}: {e}")
        print(f"Fehler beim Schreiben der Testergebnisse in die Datenbank für Dataset {datenname}: {e}")
        raise  # Re-raise, damit die übergeordnete Funktion es auch mitbekommt



# -------------------------------
# Erstellt Tabelle `modell_tests` falls sie noch nicht existiert
# -------------------------------
def erstelle_modell_tests_tabelle(engine, db_schema):
    try:
        with engine.begin() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {db_schema}.modell_tests (
                    id SERIAL PRIMARY KEY,
                    modellname TEXT NOT NULL,
                    score FLOAT NOT NULL,
                    durchgang INT NOT NULL,
                    dauer FLOAT NOT NULL,
                    datenname TEXT NOT NULL,
                    laufzeit TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        print("Tabelle 'modell_tests' geprüft/erstellt.")
        logging.info("Tabelle 'modell_tests' geprüft/erstellt.")
    except Exception as e:
        print(f"Fehler beim Erstellen der Tabelle modell_tests: {e}")
        logging.error(f"Fehler beim Erstellen der Tabelle modell_tests: {e}")
        raise



# -------------------------------
# Einstiegspunkt
# -------------------------------
if __name__ == "__main__":
    # Logging Konfigurieren
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='model_test.log',  # Optional: Log in eine Datei
        filemode='a'
    )
    main()
