#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
import logging  # Importiere das Logging Modul
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from db import DatenbankVerbindung
from utils import pruefe_und_erstelle_ordner


class Auswertung:
    def __init__(self):
        self.db = DatenbankVerbindung()
        try:
            self.df = self.db.lade_modelltestergebnisse()
            self.df = self.df.drop(columns='id', errors='ignore')
            self.df['laufzeit'] = pd.to_datetime(self.df['laufzeit'])
            logging.info("Auswertungsobjekt initialisiert und Daten geladen.")  # Logging
        except Exception as e:
            logging.error(f"Fehler beim Initialisieren der Auswertung: {e}")
            raise  # Kritischer Fehler: Programmabbruch

    def _speichere_plot(self, pfad: str):
        """
        Hilfsmethode zum Speichern eines aktuellen Plots
        """
        try:
            plt.tight_layout()
            plt.savefig(pfad, dpi=300)
            plt.close()
            logging.info(f"[Plot gespeichert] {pfad}")  # Logging
        except Exception as e:
            logging.error(f"Fehler beim Speichern des Plots '{pfad}': {e}")
            # Nicht kritisch: Fortfahren
            print(f"Fehler beim Speichern des Plots. Details im Log.")

    def speichere_gruppierte_ergebnisse_csv(self, gruppiert_nach: str = 'modellname', ordner: str = 'Auswertung'):
        """
        Speichert für jede Gruppe (Modell oder Dataset) eine CSV mit allen zugehörigen Ergebnissen.
        """
        try:
            gruppen = self.df[gruppiert_nach].unique()
            logging.info(f"Starte Speicherung gruppierter Ergebnisse für Gruppierung: {gruppiert_nach}")  # Logging

            for gruppe in gruppen:
                try:
                    df_gruppe = self.df[self.df[gruppiert_nach] == gruppe]
                    ordner_pfad = os.path.join(ordner, gruppiert_nach)
                    pruefe_und_erstelle_ordner(ordner_pfad)
                    dateipfad = os.path.join(ordner_pfad, f"{gruppe}.csv")
                    df_gruppe.to_csv(dateipfad, index=False)
                    logging.info(f"CSV-Datei gespeichert: {dateipfad}")  # Logging
                except Exception as e:
                    logging.error(f"Fehler beim Speichern der CSV-Datei für Gruppe '{gruppe}': {e}")
                    print(f"Fehler beim Speichern der CSV-Datei für Gruppe '{gruppe}'. Details im Log.")
                    # Nicht kritisch: Fortfahren mit der nächsten Gruppe
        except Exception as e:
            logging.error(f"Fehler in speichere_gruppierte_ergebnisse_csv: {e}")
            print(f"Unerwarteter Fehler beim Speichern der CSV-Dateien. Details im Log.")
            # Nicht kritisch: Fortfahren

    def plot_beste_scores(self,wert: str = 'score', gruppiert_nach: str = 'modellname', ordner: str = 'Auswertung'):
        """
        Erstellt einen Balkenplot der besten Scores für jede Gruppe.
        """
        try:
            logging.info(f"Erstelle Balkenplot der besten Scores für Gruppierung: {gruppiert_nach}")  # Logging
            gruppen = self.df[gruppiert_nach].unique()
            plt.figure(figsize=(10, 6))
            for gruppe in gruppen:
                try:
                    df_gruppe = self.df[self.df[gruppiert_nach] == gruppe]
                    bester_durchgang = df_gruppe.loc[df_gruppe[wert].idxmax()]
                    plt.bar(gruppe, bester_durchgang[wert])
                except Exception as e:
                    logging.error(f"Fehler beim Erstellen des Balkenplots für Gruppe '{gruppe}': {e}")
                    print(f"Fehler beim Erstellen des Balkenplots für Gruppe '{gruppe}'. Details im Log.")
                    # Nicht kritisch: Fortfahren mit der nächsten Gruppe
            plt.xlabel(gruppiert_nach)
            if wert == 'score':
                plt.ylabel('Bester Score')
            else:
                plt.ylabel('Bester F1-Score')
            plt.title(f'Beste Scores nach {gruppiert_nach}')
            ordner_pfad = os.path.join(ordner, gruppiert_nach)
            pruefe_und_erstelle_ordner(ordner_pfad)
            dateipfad = os.path.join(ordner_pfad, f"beste_scores_{gruppiert_nach}_{wert}.png")
            self._speichere_plot(dateipfad)
        except Exception as e:
            logging.error(f"Fehler in plot_beste_scores: {e}")
            print(f"Unerwarteter Fehler beim Erstellen der Balkenplots. Details im Log.")
            # Nicht kritisch: Fortfahren

    def plot_schnellste_durchlaeufe(self, gruppiert_nach: str = 'modellname', ordner: str = 'Auswertung'):
        """
        Erstellt einen Balkenplot der schnellsten Durchläufe für jede Gruppe.
        """
        try:
            logging.info(f"Erstelle Balkenplot der schnellsten Durchläufe für Gruppierung: {gruppiert_nach}")  # Logging
            gruppen = self.df[gruppiert_nach].unique()
            plt.figure(figsize=(10, 6))
            for gruppe in gruppen:
                try:
                    df_gruppe = self.df[self.df[gruppiert_nach] == gruppe]
                    schnellster_durchgang = df_gruppe.loc[df_gruppe['dauer'].idxmin()]
                    plt.bar(gruppe, schnellster_durchgang['dauer'])
                except Exception as e:
                    logging.error(f"Fehler beim Erstellen des Balkenplots für Gruppe '{gruppe}': {e}")
                    print(f"Fehler beim Erstellen des Balkenplots für Gruppe '{gruppe}'. Details im Log.")
                    # Nicht kritisch: Fortfahren mit der nächsten Gruppe
            plt.xlabel(gruppiert_nach)
            plt.ylabel('Schnellste Durchlaufzeit (Sekunden)')
            plt.title(f'Schnellste Durchläufe nach {gruppiert_nach}')
            ordner_pfad = os.path.join(ordner, gruppiert_nach)
            pruefe_und_erstelle_ordner(ordner_pfad)
            dateipfad = os.path.join(ordner_pfad, f"schnellste_durchlaeufe_{gruppiert_nach}.png")
            self._speichere_plot(dateipfad)
        except Exception as e:
            logging.error(f"Fehler in plot_schnellste_durchlaeufe: {e}")
            print(f"Unerwarteter Fehler beim Erstellen der Balkenplots. Details im Log.")
            # Nicht kritisch: Fortfahren

    def ranking_plot(self, wert: str = 'score', gruppiert_nach: str = 'modellname', ordner: str = "Auswertung"):
        """
        Erstellt einen Scatterplot, der die Scores und Durchlaufzeiten der Modelle/Datasets vergleicht.
        """
        try:
            logging.info(f"Erstelle Ranking-Plot für Gruppierung: {gruppiert_nach}")  # Logging
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=self.df, x='dauer', y=wert, hue=gruppiert_nach)
            plt.xlabel('Durchlaufzeit (Sekunden)')
            if wert == 'score':
                plt.ylabel('Score')
            else:
                plt.ylabel('f1_Score')
            plt.title(f'Ranking nach {gruppiert_nach}')
            ordner_pfad = os.path.join(ordner, gruppiert_nach)
            pruefe_und_erstelle_ordner(ordner_pfad)
            dateipfad = os.path.join(ordner_pfad, f"ranking_{gruppiert_nach}_{wert}.png")
            self._speichere_plot(dateipfad)
        except Exception as e:
            logging.error(f"Fehler in ranking_plot: {e}")
            print(f"Unerwarteter Fehler beim Erstellen des Ranking-Plots. Details im Log.")
            # Nicht kritisch: Fortfahren

    def generiere_html_report(self, gruppiert_nach: str = 'modellname', ordner: str = 'Auswertung'):
        """
        Generiert einen HTML-Report für jede Gruppe (Modell oder Dataset).
        """
        try:
            logging.info(f"Generiere HTML-Report für Gruppierung: {gruppiert_nach}")
            gruppen = self.df[gruppiert_nach].unique()

            for gruppe in gruppen:
                try:
                    df_gruppe = self.df[self.df[gruppiert_nach] == gruppe].copy()
                    ordner_pfad = os.path.join(ordner, gruppiert_nach)
                    pruefe_und_erstelle_ordner(ordner_pfad)
                    dateipfad = os.path.join(ordner_pfad, f"{gruppe}.html")

                    # Weitere Reports verlinken
                    links_html = "<h3>Weitere Auswertungen</h3><ul>"
                    for andere_gruppe in gruppen:
                        if andere_gruppe != gruppe:
                            links_html += f'<li><a href="{andere_gruppe}.html">{andere_gruppe}</a></li>'
                    links_html += "</ul>"

                    # === Confusion Matrix als HTML-Tabelle formatieren ===
                    def format_cm(cm_text):
                        try:
                            cm = eval(cm_text) if isinstance(cm_text, str) else cm_text
                            html = "<table style='border-collapse:collapse;'>"
                            for row in cm:
                                html += "<tr>" + "".join(
                                    f"<td style='border:1px solid #ccc; padding:4px; text-align:center;'>{v}</td>"
                                    for v in row
                                ) + "</tr>"
                            html += "</table>"
                            return html
                        except Exception as e:
                            logging.warning(f"Fehler beim Parsen der CM: {e}")
                            return "<i>Fehler beim Parsen der Confusion Matrix</i>"

                    df_gruppe['ConfusionMatrix'] = df_gruppe['cm'].apply(format_cm)

                    # Auswahl der Spalten für Anzeige
                    df_anzeige = df_gruppe[
                        ['modellname', 'datenname', 'durchgang', 'score', 'f1', 'dauer', 'ConfusionMatrix']]
                    tabelle_html = df_anzeige.to_html(index=False, escape=False, border=1, classes="tabelle")

                    # === Plots anzeigen ===
                    plots_html = "<h3>Visualisierungen</h3>"
                    for bild in os.listdir(ordner_pfad):
                        if bild.endswith(".png"):
                            plots_html += f'<div><img src="{bild}" alt="{bild}" style="max-width:800px; margin:10px 0;"></div>'

                    # === HTML zusammenbauen ===
                    html_inhalt = f"""
                    <html>
                    <head>
                        <meta charset="utf-8">
                        <title>Auswertung – {gruppe}</title>
                        <style>
                            body {{ font-family: sans-serif; padding: 20px; }}
                            .tabelle {{ border-collapse: collapse; width: 100%; }}
                            .tabelle th, .tabelle td {{ border: 1px solid #ccc; padding: 5px; }}
                            .tabelle th {{ background-color: #f5f5f5; }}
                        </style>
                    </head>
                    <body>
                        <h1>Auswertung: {gruppe}</h1>
                        {links_html}
                        <h2>Ergebnisse</h2>
                        {tabelle_html}
                        {plots_html}
                    </body>
                    </html>
                    """

                    # Speichern
                    with open(dateipfad, "w", encoding="utf-8") as f:
                        f.write(html_inhalt)
                    logging.info(f"HTML-Report generiert: {dateipfad}")

                except Exception as e:
                    logging.error(f"Fehler beim Generieren des HTML-Reports für Gruppe '{gruppe}': {e}")
                    print(f"Fehler beim Generieren des HTML-Reports für Gruppe '{gruppe}'. Details im Log.")

        except Exception as e:
            logging.error(f"Fehler in generiere_html_report: {e}")
            print(f"Unerwarteter Fehler beim Generieren der HTML-Reports. Details im Log.")
            # Nicht kritisch: Fortfahren
