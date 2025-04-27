#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
# Testen verschiedener Modelle am selben Dataframe um zu prüfen, welches Modell die besten Ergebnisse liefert.
import os
import pickle
import time

import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from funktionen import printf


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

datenname = 'malware_detect'

def main():
    # Dictionary zum Speichern der Ergebnisse: Key = Modellname, Value = Liste der Scores
    Ergebnisse = {modell.name + '_ergebnis': [] for modell in Modelle}
    if os.path.exists(datenname + "_df.pkl"):
        with open(datenname + "_df.pkl", "rb") as f:
            df = pickle.load(f)
    else:
        df = dataframe_bauen()
        with open(datenname + "_df.pkl", "wb") as f:
            pickle.dump(df, f)
    if os.path.exists(datenname + "_split.pkl"):
        with open(datenname + "_split.pkl", "rb") as f:
            X_train, X_test, y_train, y_test = pickle.load(f)
    else:
        X_train, X_test, y_train, y_test = dataframe_praeparieren(df)
        with open(datenname + "_split.pkl", "wb") as f:  # <- RICHTIG
            pickle.dump((X_train, X_test, y_train, y_test), f)

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
    ordner = f"Auswertung/{datenname}/"

    if not os.path.exists(ordner):
        os.makedirs(ordner)  # Erstellt alle benötigten Unterordner
    df_alle.to_csv("Auswertung/" + datenname + "/" + datenname + "_Modell_Ergebnisse.csv", index=False)
    # Anzeige
    print(df_alle.to_string(index=False))

    modell_vergleich(Ergebnisse)


def modell_vergleich(Ergebnisse):
    # DONE bestes Modell je Durchgang finden
    # DONE bestes Modell über alle Durchgänge finden
    # DONE beste Kombi aus Modell und Durchgang finden
    # DONE die Dauer mit berücksichtigen
    # ermittle höchste Genauigkeit je Modell
    # Ergebnisliste vorbereiten
    beste_ergebnisse = {}

    for key, werte in Ergebnisse.items():
        # Erst den höchsten Score finden
        max_score = max(w[1] for w in werte)

        # Alle Einträge mit diesem Max-Score herausfiltern
        kandidaten = [w for w in werte if w[1] == max_score]

        # Aus den Kandidaten den mit der geringsten Dauer nehmen
        best = min(kandidaten, key=lambda x: x[2])  # x[2] ist Dauer

        # Als bestes Ergebnis abspeichern
        beste_ergebnisse[key] = best

    df_beste = pd.DataFrame.from_dict(beste_ergebnisse, orient='index', columns=['Durchgang', 'Score (%)', 'Dauer (s)'])
    df_beste.index.name = 'Modell'
    df_beste.reset_index(inplace=True)
    df_beste['Modell'] = df_beste['Modell'].str.replace('_ergebnis', '')
    df_beste = df_beste.sort_values(by=['Score (%)', 'Dauer (s)'], ascending=[False, True])
    printf(
        f"Hier eine Übersicht der besten Ergebnisse je Modell (Beste Ergebnisse = höchste Genauigkeit in kürzester Dauer): ")
    print(df_beste.to_string(index=False))

    # ermittle schnellsten Durchgang je Modell
    # Ergebnisliste vorbereiten
    schnellste_ergebnisse = []

    for key, werte in Ergebnisse.items():
        # niedrigste Dauer finden (Index 2 ist die Dauer)
        best = min(werte, key=lambda x: x[2])

        # Dictionary für DataFrame
        eintrag = {
            'Modell': key.replace('_ergebnis', ''),
            'Bester Score (%)': round(best[1], 2),
            'Durchgang': best[0],
            'Dauer (s)': round(best[2], 3)
        }
        schnellste_ergebnisse.append(eintrag)

    # DataFrame erzeugen
    df_schnell = pd.DataFrame(schnellste_ergebnisse)

    # Nach Score sortieren
    df_schnell = df_schnell.sort_values(by='Dauer (s)', ascending=True)

    # Anzeige
    print("Hier eine Übersicht über die jeweils schnellsten Durchgänge je Modell.")
    print(df_schnell.to_string(index=False))
    # Ergebnisse exportieren
    df_beste.to_csv("Auswertung/" + datenname + "/Auswertung_Beste_Modelle_" + datenname + ".csv", index=False)
    df_schnell.to_csv("Auswertung/" + datenname + "/" + "Auswertung_Schnellste_Modelle" + datenname + ".csv", index=False)

def dataframe_bauen():
    # Dataframe definieren
    df = sns.load_dataset("penguins")
    # Null-Zeilen entfernen
    df = df.dropna()
    # umwandeln von nicht nummerischen Werten in nummerische Werte
    # Geschlecht männlich in 0 und Geschlecht weiblich in 1 umwandeln
    df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
    # Umwandeln der Inselnamen in Nummern 1 bis 3
    df['island'] = df['island'].map({'Torgersen': 1, 'Biscoe': 2, 'Dream': 3})
    # df['species'] = df['species'].map({'Adelie': 1, 'Chinstrap': 2, 'Gentoo': 3})
    return df


def dataframe_praeparieren(df):
    # Features definiert
    X = df.drop(["species"], axis=1)
    # Labels definiert
    y = df["species"]
    # Dataframe in Test und Trainingsdaten aufgeteilt
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=119)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()
