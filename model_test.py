#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
import os
import time

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime

from utils import printf
from db import DatenbankVerbindung

# DONE Auswertungen visualisieren und speichern
# DONE Auswertungen zusätzlich in pgSQL speichern
# TODO Auswertungen tabellarisch + Visuell kombiniert speichern (.html?, dynamisch mittels Python?)
# DONE Lade alle verfügbaren dataframes aus der Tabelle dataframe und iteriere über all diese dataframes
# TODO trennen nach Modell Test, Auswertung, Ausgabe / speichern der Ergebnisse


class MLModell:
    def __init__(self, name, model_func):
        self.name = name
        self.model_func = model_func  # Funktion, die ein Modell zurückgibt
        self.model = None
        self.score = 0

    def train(self, X_train, y_train, i):
        self.model = self.model_func(i)  # Neues Modell mit Parameter i erzeugen
        self.model.fit(X_train, y_train)

    def testen(self, X_test, y_test, i):
        predictions = self.model.predict(X_test)
        self.score = accuracy_score(y_test, predictions) * 100
        return self.score


# Liste der Modelle mit Parametrisierung über lambdas
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

# datenname = 'malware_detect'
db = DatenbankVerbindung()
metadaten_liste = db.lade_dataset_metadaten()


def main():
    engine = db.get_engine()
    schema = db.get_schema()
    for eintrag in metadaten_liste:
        datenname = eintrag['dataset_name']
        X_train = pd.read_sql_table(eintrag['x_train_tabelle'], con=engine, schema=schema)
        X_test = pd.read_sql_table(eintrag['x_test_tabelle'], con=engine, schema=schema)
        y_train = pd.read_sql_table(eintrag['y_train_tabelle'], con=engine, schema=schema)
        y_test = pd.read_sql_table(eintrag['y_test_tabelle'], con=engine, schema=schema)
        # Dictionary zum Speichern der Ergebnisse: Key = Modellname, Value = Liste der Scores
        Ergebnisse = {modell.name + '_ergebnis': [] for modell in Modelle}
        printf(f"\n=== Starte Modellvergleich für Dataset: {datenname} ===")
        for i in range(1, 11):
            for modell in Modelle:
                start = time.time()  # Startzeitpunkt
                modell.train(X_train, y_train, i)
                score = modell.testen(X_test, y_test, i)
                dauer = time.time() - start  # Differenz = Laufzeit in Sekunden

                # printf(f"Durchgang {i} - {modell.name}: {score:.2f} % (Dauer: {dauer:.3f} s)")

                key = modell.name + '_ergebnis'
                Ergebnisse[key].append((i, score, dauer))
            printf(f"Durchgang {i} abgeschlossen.")
    # Dict → lange Liste von Zeilen für DataFrame vorbereiten
        alle_ergebnisse = []

        for key, werte in Ergebnisse.items():
            modell_name = key.replace('_ergebnis', '')
            for durchgang, score, dauer in werte:
                alle_ergebnisse.append({
                    'Modell': modell_name,
                    'Durchgang': durchgang,
                    'Score (%)': round(score, 2),
                    'Dauer (s)': round(dauer, 3)
                })

        # DataFrame erstellen
        df_alle = pd.DataFrame(alle_ergebnisse)

        # Optional: nach Modell und Durchgang sortieren
        df_alle = df_alle.sort_values(by=['Modell', 'Durchgang'])
        ordner = os.path.join("Auswertung", datenname)
        csv_pfad = os.path.join(ordner, f"{datenname}_Modell_Ergebnisse.csv")

        if not os.path.exists(ordner):
            os.makedirs(ordner)
        df_alle.to_csv(csv_pfad, index=False)
        # Anzeige
        print(df_alle.to_string(index=False))

        modell_vergleich(Ergebnisse, datenname)


def erstelle_modell_tests_tabelle(engine, db_schema):
    """Erstellt die Tabelle modell_tests, falls sie noch nicht existiert."""
    with engine.connect() as conn:
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
    print("Tabelle modell_tests geprüft/erstellt.")


def modell_vergleich(Ergebnisse, datenname):
    # ermittle höchste Genauigkeit je Modell
    # Ergebnisliste vorbereiten
    beste_ergebnisse = {}

    for key, werte in Ergebnisse.items():
        max_score = max(w[1] for w in werte)
        kandidaten = [w for w in werte if w[1] == max_score]
        best = min(kandidaten, key=lambda x: x[2])  # kleinste Dauer bei gleichem Score
        beste_ergebnisse[key] = best

    df_beste = pd.DataFrame.from_dict(beste_ergebnisse, orient='index', columns=['Durchgang', 'Score (%)', 'Dauer (s)'])
    df_beste.index.name = 'Modell'
    df_beste.reset_index(inplace=True)
    df_beste['Modell'] = df_beste['Modell'].str.replace('_ergebnis', '')
    df_beste = df_beste.sort_values(by=['Score (%)', 'Dauer (s)'], ascending=[False, True])

    printf("Hier eine Übersicht der besten Ergebnisse je Modell:")
    print(df_beste.to_string(index=False))

    # -----------------
    # Beste Ergebnisse als Plot
    # -----------------

    # Ordner prüfen
    if not os.path.exists(f"Auswertung/{datenname}/"):
        os.makedirs(f"Auswertung/{datenname}/")

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Score (%)", y="Modell", data=df_beste, palette="Blues_d")
    plt.title("Beste Genauigkeit je Modell")
    plt.tight_layout()
    plt.savefig(f"Auswertung/{datenname}/Beste_Modelle_{datenname}.png")
    plt.close()

    # -----------------
    # Schnellste Durchgänge je Modell
    # -----------------

    schnellste_ergebnisse = []

    for key, werte in Ergebnisse.items():
        best = min(werte, key=lambda x: x[2])  # schnellste Dauer finden
        eintrag = {
            'Modell': key.replace('_ergebnis', ''),
            'Bester Score (%)': round(best[1], 2),
            'Durchgang': best[0],
            'Dauer (s)': round(best[2], 3)
        }
        schnellste_ergebnisse.append(eintrag)

    df_schnell = pd.DataFrame(schnellste_ergebnisse)
    df_schnell = df_schnell.sort_values(by='Dauer (s)', ascending=True)

    printf("Hier eine Übersicht über die jeweils schnellsten Durchgänge je Modell.")
    print(df_schnell.to_string(index=False))

    # Schnellste Durchgänge als Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Dauer (s)", y="Modell", data=df_schnell, palette="Greens_d")
    plt.title("Schnellster Durchgang je Modell")
    plt.tight_layout()
    plt.savefig(f"Auswertung/{datenname}/Schnellste_Modelle_{datenname}.png")
    plt.close()

    # -----------------
    # Ergebnisse als CSV speichern
    # -----------------

    df_beste.to_csv(f"Auswertung/{datenname}/Auswertung_Beste_Modelle_{datenname}.csv", index=False)
    df_schnell.to_csv(f"Auswertung/{datenname}/Auswertung_Schnellste_Modelle_{datenname}.csv", index=False)

    # -----------------
    # Ergebnisse in PostgreSQL speichern
    # -----------------

    engine = db.get_engine()
    schema = db.get_schema()
    # -----------------
    # Alle Einzelergebnisse in zentrale Tabelle modell_tests schreiben
    # -----------------

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

    # Tabelle modell_tests anlegen (falls noch nicht vorhanden)
    erstelle_modell_tests_tabelle(engine, schema)

    # Ergebnisse anhängen
    df_gesamt.to_sql(name='modell_tests', con=engine, if_exists='append', index=False, schema=schema, method='multi')
    visualisiere_auswertung(df_beste, df_schnell, datenname)
    printf("Alle Testergebnisse wurden erfolgreich gespeichert (CSV, PNG, Datenbank).")


def visualisiere_auswertung(df_beste, df_schnell, datenname):
    # Visualisierung der besten Modelle nach Score
    plt.figure(figsize=(10, 6))
    plt.barh(df_beste['Modell'], df_beste['Score (%)'], color='skyblue')
    plt.xlabel('Score (%)')
    plt.title('Beste Genauigkeit je Modell')
    plt.gca().invert_yaxis()  # Höchster Score oben
    plt.tight_layout()
    plt.savefig(f"Auswertung/{datenname}/Beste_Modelle_Score.png")
    plt.close()

    # Visualisierung der schnellsten Modelle nach Dauer
    plt.figure(figsize=(10, 6))
    plt.barh(df_schnell['Modell'], df_schnell['Dauer (s)'], color='lightcoral')
    plt.xlabel('Dauer (s)')
    plt.title('Schnellste Modelle je Durchgang')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"Auswertung/{datenname}/Schnellste_Modelle_Dauer.png")
    plt.close()


if __name__ == "__main__":
    main()
