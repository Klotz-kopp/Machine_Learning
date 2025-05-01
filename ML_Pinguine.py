#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
from utils import printf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():
    df = dataframe_bauen()
    KNeighbors_ergebnis = []
    RandomForest_ergebnis = []
    for i in range(1, 11):
        hoechster_genauigkeitswert_KNeighbors = vorhersage_KNeigbors(df)
        KNeighbors_ergebnis.append(hoechster_genauigkeitswert_KNeighbors)
        hoechster_genauigkeitswert_RandomForest = vorhersage_RandomForest(df)
        RandomForest_ergebnis.append(hoechster_genauigkeitswert_RandomForest)
    max_genauigkeitswert_KNeighbors = max(KNeighbors_ergebnis, key=lambda x: x[1])
    max_genauigkeitswert_RandomForest = max(RandomForest_ergebnis, key=lambda x: x[1])
    printf(f"KNeighors Ergebnisse: {KNeighbors_ergebnis}")
    printf(f"RandomForest Ergebnisse: {RandomForest_ergebnis}")
    vergleich_KNeighbors_Randomforest(max_genauigkeitswert_KNeighbors, max_genauigkeitswert_RandomForest)


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
    df['species'] = df['species'].map({'Adelie': 1, 'Chinstrap': 2, 'Gentoo': 3})
    return df


def vorhersage_KNeigbors(df):
    # Features definiert
    X = df.drop(["sex"], axis=1)
    # Labels definiert
    y = df["sex"]
    # Dataframe in Test und Trainingsdaten aufgeteilt
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=119)
    # Wir prüfen die Genauigkeit der Vorhersagen mit 1-10 Nachbarn
    genauigkeitswerte_KNeighbors = []
    for i in range(1, 11):
        # Classifier definiert
        clf = KNeighborsClassifier(n_neighbors=i)
        # Classifier mit Trainingsdaten traininert
        clf.fit(X_train, y_train)
        # Vorhersagen mit dem Classifier auf die Testdaten durchgeführt
        predictions = clf.predict(X_test)
        # Vorhersagegenauigkeit berechnet, indem die Vorhersagen mit der realität verglichen wurde
        # printf(f"Bei der Verwendung des Wert {i} erreichen wir eine Genauigkeit von {accuracy_score(predictions, y_test) * 100:.2f}%.")
        genauigkeitswerte_KNeighbors.append((i, accuracy_score(predictions, y_test) * 100))
    hoechster_genauigkeitswert_KNeighbors = max(genauigkeitswerte_KNeighbors, key=lambda x: x[1])
    # printf(f"Der höchste Genauigkeitswert der erreicht wurde, liegt bei {hoechster_genauigkeitswert_KNeighbors[1]:.2f}%, bei der betrachtung von {hoechster_genauigkeitswert_KNeighbors[0]} Nachbarn.")
    return hoechster_genauigkeitswert_KNeighbors


def vorhersage_RandomForest(df):
    # Features definiert
    X = df.drop(["species"], axis=1)
    # Labels definiert
    y = df["species"]
    # Dataframe in Test und Trainingsdaten aufgeteilt
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=119)
    genauigkeitswerte_RandomForest = []
    for i in range(1, 11):
        # Classifier definiert
        clf1 = RandomForestClassifier(n_estimators=i)
        # Classifier mit Trainingsdaten traininert
        clf1.fit(X_train, y_train)
        # Vorhersagen mit dem Classifier auf die Testdaten durchgeführt
        predictions = clf1.predict(X_test)
        # Vorhersagegenauigkeit berechnet, indem die Vorhersagen mit der realität verglichen wurde
        # printf(f"Bei der Verwendung des Wert {i} erreichen wir eine Genauigkeit von {accuracy_score(predictions, y_test) * 100:.2f} %.")
        genauigkeitswerte_RandomForest.append((i, accuracy_score(predictions, y_test) * 100))
    hoechster_genauigkeitswert_RandomForest = max(genauigkeitswerte_RandomForest, key=lambda x: x[1])
    # printf(f"Der höchste Genauigkeitswert der erreicht wurde, liegt bei {hoechster_genauigkeitswert_RandomForest[1]:.2f} %, bei der betrachtung von {hoechster_genauigkeitswert_RandomForest[0]} Nachbarn.")
    return hoechster_genauigkeitswert_RandomForest


def vergleich_KNeighbors_Randomforest(max_genauigkeitswert_KNeighbors, max_genauigkeitswert_RandomForest):
    if max_genauigkeitswert_KNeighbors[1] > max_genauigkeitswert_RandomForest[1]:
        printf(f"Der höchste Genauigkeitswert von {max_genauigkeitswert_KNeighbors[1]:.2f} % wurde mit KNeighbors erreicht bei der verwendung von {max_genauigkeitswert_KNeighbors[0]} Nachbarn.\n Die Genauigkeit liegt {max_genauigkeitswert_KNeighbors[1]} - {max_genauigkeitswert_RandomForest[1]:.2f} % höher als bei RandomForest.")
    else:
        printf(f"Der höchste Genauigkeitswert von {max_genauigkeitswert_RandomForest[1]:.2f} % wurde mit RandomForest erreicht bei der verwendung von {max_genauigkeitswert_RandomForest[0]} Estimators.\n Die Genauigkeit liegt {max_genauigkeitswert_RandomForest[1] - max_genauigkeitswert_KNeighbors[1]:.2f} % höher als bei KNeighbors.")


main()
