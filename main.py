import dataframe_to_db
import model_test
from auswertung import Auswertung
from utils import zeit_messen  # <-- WICHTIG


@zeit_messen
def main():
    dataframe_to_db.main()
    model_test.main()

    auswertung = Auswertung()  # <-- Instanz erzeugen

    for gruppierung in ["modellname", "datenname"]:
        auswertung.speichere_gruppierte_ergebnisse_csv(gruppierung)
        auswertung.plot_beste_scores(gruppierung)
        auswertung.plot_schnellste_durchlaeufe(gruppierung)
        auswertung.ranking_plot(gruppierung)
        auswertung.generiere_html_report(gruppierung)
