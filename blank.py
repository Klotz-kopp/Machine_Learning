#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.

datasets = {
    'malware_detect' : (pd.read_csv('Datasets/TUANDROMD.csv'), 'Label'),
    'penguins': (sns.load_dataset('penguins'),'species' ),
    'brustkrebs' : (sklearn.datasets.load_breast_cancer(), 'Diagnosis'),
    'iris' : (sklearn.datasets.load_iris(), 'class')
}