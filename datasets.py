#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.

import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_iris


# Hilfsfunktion für sklearn Datasets
def sklearn_to_df(dataset):
    return pd.DataFrame(data=dataset.data, columns=dataset.feature_names).assign(target=dataset.target)


# Datasets: name → (DataFrame, Zielspalte, Beschreibung)
datasets = {
    'malware_detect': (
        pd.read_csv('Datasets/TUANDROMD.csv'),
        'Label',
        "The target attribute for classification is a category (malware vs goodware)."
    ),
    'penguins': (
        sns.load_dataset('penguins'),
        'species',
        "Dataset for data exploration and visualization (alternative to Iris)."
    ),
    'brustkrebs': (
        sklearn_to_df(load_breast_cancer()),
        'target',
        "FNA-derived features for breast cancer classification (0=malignant, 1=benign)."
    ),
    'iris': (
        sklearn_to_df(load_iris()),
        'target',
        "Famous Iris flower dataset. Target: 0=setosa, 1=versicolor, 2=virginica."
    )
}
