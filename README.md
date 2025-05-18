# 🧠 Modellvergleich mit Scikit-Learn und PostgreSQL

Dieses Python-Projekt ermöglicht dir einen vollständigen Machine-Learning-Workflow:
- verschiedene Datensätze vorbereiten
- ML-Modelle trainieren und testen
- Ergebnisse speichern und vergleichen
- Visualisierungen und HTML-Reports erzeugen

---

## 🔧 Features

- ✅ Laden & Vorverarbeiten beliebiger CSV- oder OpenDataset-Daten
- ✅ Speichern der Daten in PostgreSQL
- ✅ Automatischer Vergleich mehrerer Klassifikations-Modelle
- ✅ Speicherung der Ergebnisse (CSV, Plots, Datenbank)
- ✅ Visualisierung der Ergebnisse je Modell & je Datensatz
- ✅ Generierung interaktiver HTML-Berichte
- ✅ Modularer und erweiterbarer Aufbau

---

## 🗂️ Projektstruktur
````tree
Machine Learning/
│
├── auswertung.py # Auswertung & Visualisierung der Ergebnisse
├── create_config.py # Erstellt Konfigurationsdatei für Datenbankzugriff
├── dataframe_to_db.py # Lädt und speichert vorbereitete Datasets in PostgreSQL
├── db.py # Klasse DatenbankVerbindung für alle DB-Operationen
├── model_test.py # Trainiert alle Modelle auf allen Datasets
├── utils.py # Hilfsfunktionen (z.B. printf, Zeitmessung)
├── main.py # Einstiegspunkt: führt alles der Reihe nach aus
├── db_config.cfg # Wird automatisch erzeugt (nicht im Repo enthalten)
├── /Datasets/ # Enthält externe .csv-Dateien
└── /Auswertung/ # Ergebnisse (CSV, Plots, HTML)
`````

---

## 📦 Verwendete Modelle

- `KNeighborsClassifier`
- `RandomForestClassifier`
- `LogisticRegression`
- `DecisionTreeClassifier`
- `GradientBoostingClassifier`
- `SVC`
- `GaussianNB`
- `MLPClassifier`

---

## ⚙️ Voraussetzungen

- Python ≥ 3.9
- PostgreSQL (lokal oder remote)

### Installation

```bash
# Repository klonen
git clone https://github.com/Klotz-kopp/python_tutorial.git

# In das Projektverzeichnis wechseln
cd python_tutorial/Machine\ Learning/

# Virtuelle Umgebung erstellen (optional, empfohlen)
python -m venv .venv
source .venv/bin/activate  # oder .venv\Scripts\activate auf Windows

# Abhängigkeiten installieren
pip install -r requirements.txt
```
🚀 Nutzung
```bash
# Hauptskript starten
python main.py
```
Beim ersten Start wirst du aufgefordert, eine Datenbank-Konfiguration zu erstellen (db_config.cfg).

Danach läuft:

    Laden & Speichern aller Datasets

    Testen aller Modelle

    Speichern & Visualisierung der Ergebnisse

📊 Beispielergebnisse

Alle Ergebnisse findest du im Ordner Auswertung/, z.B.:

    CSV-Dateien mit den besten Scores

    Plots der schnellsten Modelle

    Interaktive HTML-Berichte je Modell und Dataset

🧠 Erweiterungsmöglichkeiten

    Weitere Modelle (z.B. XGBoost, LightGBM) hinzufügen

    Weitere Metriken wie Precision, Recall einbauen

    Klassifizierte Labels ebenfalls in der DB speichern

    Webinterface mit Dash, Streamlit oder Flask

👨‍💻 Autor

Dieses Projekt wurde entwickelt von Klotz-kopp zum Lernen und Testen moderner Machine-Learning-Pipelines.

📜 Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe [LICENSE](LICENSE) für Details.