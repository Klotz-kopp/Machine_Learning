import dataframe_to_db
import model_test
from auswertung import Auswertung
from utils import zeit_messen  # <-- WICHTIG
import logging # Import Logging

@zeit_messen
def main():
    try:
        dataframe_to_db.main()
    except Exception as e:
        logging.error(f"Fehler in dataframe_to_db.main(): {e}")
        print(f"Fehler beim Ausführen von dataframe_to_db.main(): {e}")
        # Entscheidung: Programm hier beenden oder fortsetzen?
        # Für Abbruch:
        # sys.exit(1)
        # Für Fortsetzung (mit Loggen):
        pass

    try:
        model_test.main()
    except Exception as e:
        logging.error(f"Fehler in model_test.main(): {e}")
        print(f"Fehler beim Ausführen von model_test.main(): {e}")
        # Entscheidung: Programm hier beenden oder fortsetzen?
        # Für Abbruch:
        # sys.exit(1)
        # Für Fortsetzung (mit Loggen):
        pass



    try:
        auswertung = Auswertung()  # <-- Instanz erzeugen

        for gruppierung in ["modellname", "datenname"]:
            try:
                auswertung.speichere_gruppierte_ergebnisse_csv(gruppierung)
                auswertung.plot_beste_scores(gruppierung)
                auswertung.plot_schnellste_durchlaeufe(gruppierung)
                auswertung.ranking_plot(gruppierung)
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
    # Konfiguration des Loggings ZUERST, damit es überall verfügbar ist.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='main.log',  # Optional: Log in eine Datei
        filemode='a'
    )
    main()
