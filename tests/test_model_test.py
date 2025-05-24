import pytest
from sklearn.dummy import DummyClassifier
from model_test import MLModell
import pandas as pd

# Dummy-Daten für Tests
X_train = pd.DataFrame({
    "feature1": [0, 1, 0, 1],
    "feature2": [1, 1, 0, 0]
})
y_train = pd.Series([0, 1, 0, 1])
X_test = pd.DataFrame({
    "feature1": [0, 1],
    "feature2": [1, 0]
})
y_test = pd.Series([0, 1])

# ==============================
# Tests für MLModell
# ==============================

def test_train_success():
    modell = MLModell("Dummy", lambda i: DummyClassifier(strategy="most_frequent"))
    modell.train(X_train, y_train, i=1)
    assert modell.model is not None
    assert hasattr(modell.model, "predict")

def test_testen_success():
    modell = MLModell("Dummy", lambda i: DummyClassifier(strategy="most_frequent"))
    modell.train(X_train, y_train, i=1)
    score, f1, cm = modell.testen(X_test, y_test, i=1)
    assert 0 <= score <= 100
    assert 0 <= f1 <= 100
    assert cm.shape[0] == cm.shape[1]

def test_train_failure():
    modell = MLModell("Dummy", lambda i: DummyClassifier(strategy="most_frequent"))
    with pytest.raises(RuntimeError):
        modell.train(None, None, i=1)

def test_testen_failure():
    modell = MLModell("Dummy", lambda i: DummyClassifier(strategy="most_frequent"))
    # absichtlich ohne Training aufrufen
    with pytest.raises(RuntimeError):
        modell.testen(X_test, y_test, i=1)
