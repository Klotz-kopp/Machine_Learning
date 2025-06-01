import logging  # Import Logging
import sys

import dataframe_to_db
import model_test
from auswertung import Auswertung
from utils import zeit_messen  # <-- WICHTIG
from utils import log_start
@log_start
@zeit_messen
def main():
    try:
        dataframe_to_db.main()
    except Exception as e:
        logging.error(f"Fehler in dataframe_to_db.main(): {e}")
        print(f"Fehler beim Ausführen von dataframe_to_db.main(): {e}")
        # Entscheidung: Programm hier beenden oder fortsetzen?
        # Für Abbruch:
        sys.exit(1)
        # Für Fortsetzung (mit Loggen):
        # pass

    try:
        model_test.main()
    except Exception as e:
        logging.error(f"Fehler in model_test.main(): {e}")
        print(f"Fehler beim Ausführen von model_test.main(): {e}")
        # Entscheidung: Programm hier beenden oder fortsetzen?
        # Für Abbruch:
        sys.exit(1)
        # Für Fortsetzung (mit Loggen):
        # pass



    try:
        auswertung = Auswertung()  # <-- Instanz erzeugen

        for gruppierung in ["modellname", "datenname"]:
            try:
                auswertung.speichere_gruppierte_ergebnisse_csv(gruppierung)
                auswertung.plot_beste_scores('score', gruppierung)
                auswertung.plot_beste_scores('f1', gruppierung)
                auswertung.plot_schnellste_durchlaeufe(gruppierung)
                auswertung.ranking_plot('score', gruppierung)
                auswertung.ranking_plot('f1', gruppierung)
                auswertung.generiere_html_report(gruppierung)
            except Exception as e:
                logging.error(f"Fehler bei der Auswertung für Gruppierung '{gruppierung}': {e}")
                print(f"Fehler bei der Auswertung für Gruppierung '{gruppierung}': {e}")
                # Abbruch der Schleife?  Kommt drauf an, ob die anderen Durchläufe noch sinnvoll sind.
                # break # Wenn ja.
                pass # Wenn nein
    except Exception as e:
        logging.critical(f"Schwerwiegender Fehler in main(): {e}")
        print(f"Schwerwiegender Fehler beim Erstellen oder Ausführen der Auswertung: {e}")
        sys.exit(1) # Programm Abbruch


# -------------------------------
# Einstiegspunkt
# -------------------------------
if __name__ == "__main__":
    # Eigener Logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Alles durchlassen

    # Bestehende Handler entfernen (z. B. von basicConfig)
    if logger.hasHandlers():
        logger.handlers.clear()

    # --------- INFO & WARNING Handler ---------
    class InfoWarningFilter(logging.Filter):
        def filter(self, record):
            return record.levelno in (logging.INFO, logging.WARNING)

    info_handler = logging.FileHandler("main.info.log", mode='a')
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(InfoWarningFilter())  # Nur INFO & WARNING
    info_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # --------- ERROR & CRITICAL Handler ---------
    error_handler = logging.FileHandler("main.error.log", mode='a')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Handler hinzufügen
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)

    # Start der Anwendung
    main()
