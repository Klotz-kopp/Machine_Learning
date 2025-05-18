#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from db import DatenbankVerbindung
from utils import pruefe_und_erstelle_ordner


class Auswertung:
    def __init__(self):
        self.db = DatenbankVerbindung()
        self.df = self.db.lade_modelltestergebnisse()
        self.df = self.df.drop(columns='id', errors='ignore')
        self.df['laufzeit'] = pd.to_datetime(self.df['laufzeit'])

    def _speichere_plot(self, pfad: str):
        """
        Hilfsmethode zum Speichern eines aktuellen Plots
        """
        plt.tight_layout()
        plt.savefig(pfad, dpi=300)
        plt.close()
        print(f"[Plot gespeichert] {pfad}")

    def speichere_gruppierte_ergebnisse_csv(self, gruppiert_nach: str = 'modellname', ordner: str = 'Auswertung'):
        """
        Speichert für jede Gruppe (Modell oder Dataset) eine CSV mit allen zugehörigen Ergebnissen.
        """
        gruppen = self.df[gruppiert_nach].unique()

        for gruppe in gruppen:
            df_gruppe = self.df[self.df[gruppiert_nach] == gruppe]
            df_gruppe = df_gruppe.sort_values(by='score', ascending=False)

            zielordner = os.path.join(ordner, gruppe)
            pruefe_und_erstelle_ordner(zielordner)

            pfad = os.path.join(zielordner, f"{gruppe}_Ergebnisse.csv")
            df_gruppe.to_csv(pfad, index=False)
            print(f"[CSV gespeichert] {pfad}")

    def plot_beste_scores(self, gruppiert_nach: str = 'modellname', top_n: int = 1, ordner: str = "Auswertung"):
        """
        Erstellt Balkendiagramm der besten Scores je Gruppe (Modell oder Dataset) und speichert dieses.
        """
        grouped = self.df.groupby(gruppiert_nach)
        beste = grouped.apply(lambda g: g.sort_values(by='score', ascending=False).head(top_n)).reset_index(drop=True)

        plt.figure(figsize=(10, 6))
        hue_col = 'datenname' if gruppiert_nach == 'modellname' else 'modellname'
        sns.barplot(data=beste, x='score', y=gruppiert_nach, hue=hue_col)

        plt.title(f"Beste Ergebnisse je {'Modell' if gruppiert_nach == 'modellname' else 'Dataset'}")
        plt.xlabel("Score (%)")
        plt.ylabel(gruppiert_nach.capitalize())
        plt.legend(title=hue_col.capitalize())

        zielordner = os.path.join(ordner, gruppiert_nach)
        pruefe_und_erstelle_ordner(zielordner)
        pfad = os.path.join(zielordner, f"{gruppiert_nach}_beste_scores.png")

        self._speichere_plot(pfad)

    def plot_schnellste_durchlaeufe(self, gruppiert_nach: str = 'modellname', ordner: str = "Auswertung"):
        """
        Erstellt Balkendiagramm der jeweils schnellsten Durchläufe je Gruppe und speichert dieses.
        """
        grouped = self.df.groupby(gruppiert_nach)
        schnellste = grouped.apply(lambda g: g.sort_values(by='dauer').head(1)).reset_index(drop=True)

        plt.figure(figsize=(10, 6))
        hue_col = 'datenname' if gruppiert_nach == 'modellname' else 'modellname'
        sns.barplot(data=schnellste, x='dauer', y=gruppiert_nach, hue=hue_col)

        plt.title(f"Schnellste Durchläufe je {'Modell' if gruppiert_nach == 'modellname' else 'Dataset'}")
        plt.xlabel("Dauer (s)")
        plt.ylabel(gruppiert_nach.capitalize())
        plt.legend(title=hue_col.capitalize())

        zielordner = os.path.join(ordner, gruppiert_nach)
        pruefe_und_erstelle_ordner(zielordner)
        pfad = os.path.join(zielordner, f"{gruppiert_nach}_schnellste_durchlaeufe.png")

        self._speichere_plot(pfad)

    def ranking_plot(self, wert: str = 'score', gruppiert_nach: str = 'modellname', ordner: str = "Auswertung"):
        """
        Erstellt ein Ranking-Diagramm (Durchschnittswerte) und speichert es als Bild.
        wert = 'score' oder 'dauer'
        """
        df_avg = self.df.groupby(gruppiert_nach)[wert].mean().reset_index().sort_values(by=wert, ascending=(wert == 'dauer'))

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_avg, x=wert, y=gruppiert_nach)

        plt.title(f"Durchschnittlicher {wert.capitalize()} je {'Modell' if gruppiert_nach == 'modellname' else 'Dataset'}")
        plt.xlabel(wert.capitalize())
        plt.ylabel(gruppiert_nach.capitalize())

        zielordner = os.path.join(ordner, gruppiert_nach)
        pruefe_und_erstelle_ordner(zielordner)
        pfad = os.path.join(zielordner, f"{gruppiert_nach}_ranking_{wert}.png")

        self._speichere_plot(pfad)

    def generiere_html_report(self, gruppiert_nach: str = 'modellname', ordner: str = "Auswertung"):
        """
        Erstellt je Gruppe (Modell oder Dataset) eine HTML-Datei mit:
        - Tabelle aller Ergebnisse
        - eingebetteten Plots aus dem jeweiligen Ordner
        - Links zu den anderen Gruppen
        """
        gruppen = sorted(self.df[gruppiert_nach].unique())

        for gruppe in gruppen:
            zielordner = os.path.join(ordner, gruppe)
            dateipfad = os.path.join(zielordner, f"{gruppe}.html")

            # Ergebnisse dieser Gruppe
            df_gruppe = self.df[self.df[gruppiert_nach] == gruppe].copy()
            df_gruppe = df_gruppe.sort_values(by='score', ascending=False)

            # Links zu anderen Gruppen
            links_html = "<h3>Weitere Reports:</h3><ul>"
            for andere in gruppen:
                if andere != gruppe:
                    rel_path = os.path.relpath(os.path.join('..', andere, f"{andere}.html"), start=zielordner)
                    links_html += f'<li><a href="{rel_path}">{andere}</a></li>'
            links_html += "</ul>"

            # Ergebnisse als HTML-Tabelle
            tabelle_html = df_gruppe.to_html(index=False, border=1, classes="tabelle")

            # Plots aus Ordner einbinden
            plots_html = "<h3>Visualisierungen</h3>"
            for bild in os.listdir(zielordner):
                if bild.endswith(".png"):
                    plots_html += f'<div><img src="{bild}" alt="{bild}" style="max-width:800px; margin:10px 0;"></div>'

            # HTML zusammenbauen
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

            # Datei schreiben
            with open(dateipfad, "w", encoding="utf-8") as f:
                f.write(html_inhalt)

            print(f"[HTML] Report gespeichert: {dateipfad}")
