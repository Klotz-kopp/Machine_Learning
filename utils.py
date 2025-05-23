#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
import logging  # Import Logging
import os
import re
from functools import wraps
from time import time

Farben = {
    'Schwarz': '\033[30m',
    'Rot': '\033[31m',
    'Grün': '\033[32m',
    'Gelb': '\033[33m',
    'Blau': '\033[34m',
    'Magenta': '\033[35m',
    'Cyan': '\033[36m',
    'Weiß': '\033[37m',
    'Fett': '\033[1m',
    'Unterstrich': '\033[4m',
    'Reset': '\033[0m'
}


def printf(*args, sep=' ', end='\n'):
    BLUE = "\033[34m"
    RESET = "\033[0m"

    def einfaerben(text):
        # Ersetzt alle Zahlen durch farbige Zahlen
        return re.sub(r'\b\d+\b', lambda m: f"{BLUE}{m.group()}{RESET}", text)

    # Alles in Strings umwandeln & einfärben
    output = [einfaerben(str(arg)) for arg in args]

    print(sep.join(output), end=end)


def pruefe_und_erstelle_ordner(pfad: str):
    """Erstellt den Ordner, falls er noch nicht existiert."""
    try:
        if not os.path.exists(pfad):
            os.makedirs(pfad)
            logging.info(f"Ordner erstellt: {pfad}")
    except Exception as e:
        print(f"Fehler beim Erstellen des Ordners '{pfad}': {e}")
        logging.error(f"Fehler beim Erstellen des Ordners '{pfad}': {e}")
        raise  # решаем, что ошибка при создании директории - критическая


def zeit_messen(func):
    """Decorator, der die Ausführungszeit einer Funktion misst und loggt."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        try:
            result = func(*args, **kwargs)
            dauer = time() - start
            if dauer > 60:
                minuten, sekunden = zeit_umrechnen(dauer)
                logging.info(f"Funktion '{func.__name__}' dauerte {minuten} Minuten und {sekunden:.3f} Sekunden.")
            else:
                logging.info(f"Funktion '{func.__name__}' dauerte {dauer:.3f} Sekunden.")
            return result
        except Exception as e:
            logging.error(f"Fehler bei der Ausführung von Funktion '{func.__name__}': {e}")
            print(f"Fehler bei der Ausführung von Funktion '{func.__name__}': {e}")
            raise  # Re-raise, damit der Aufrufer die Exception auch sieht
    return wrapper


def zeit_umrechnen(dauer):
    minuten = int(dauer // 60)
    sekunden = dauer % 60
    return minuten, sekunden
