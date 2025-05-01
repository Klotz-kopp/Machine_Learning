#  Copyright (c) 2025. Diese Python Skripte wurden von mir erstellt und können als Referenz von anderen genutzt und gelesen werden.
# flake8: noqa
datasets = {
    'malware_detect' : (pd.read_csv('Datasets/TUANDROMD.csv'), 'Label', "The target attribute for classification is a category (malware vs goodware)."),
    'penguins': (sns.load_dataset('penguins'),'species', "The goal of palmerpenguins is to provide a great dataset for data exploration & visualization, as an alternative to iris." ),
    'brustkrebs' : (load_breast_cancer(), 'Diagnosis', "Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image."),
    'iris' : (load_iris(), 'class', "The famous Iris database, first used by Sir R.A. Fisher")
}
