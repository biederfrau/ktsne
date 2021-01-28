#!/usr/bin/env python3
#!/usr/bin/env python3

from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import numpy as np
import pandas as pd
from sys import exit

import random
from collections import defaultdict
import json

from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

datasets = []
methods = ['ktsne', 'bhtsne', 'fitsne', 'umap']
percentages = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]

labels = np.ravel(pd.read_csv('../../data/larger_dblp_data/labels.txt', header=None).values)
results = defaultdict(lambda: defaultdict(list))

random.seed(666)
np.random.seed(666)

repeats = 100

for percentage in percentages:
    for method in methods:
        print(f"{'='*8} p = {percentage}, method = {method} {'='*8}")
        embedding = pd.read_csv(f"../emb/{method}_128d_dblp.emb", sep=" ", header=None).values

        scores = defaultdict(list)
        for _ in range(repeats):
            X_train, X_test, y_train, y_test = train_test_split(embedding, labels, train_size=percentage)

            svc = LinearSVC(dual=False).fit(X_train, y_train)
            y_pred = svc.predict(X_test)

            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_micro = f1_score(y_test, y_pred, average='micro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            accuracy = accuracy_score(y_test, y_pred)

            scores['f1_macro'].append(f1_macro)
            scores['f1_micro'].append(f1_micro)
            scores['f1_weighted'].append(f1_weighted)
            scores['accuracy'].append(accuracy)

        results[method]['f1_macro'].append((percentage, np.average(scores['f1_macro'])))
        results[method]['f1_micro'].append((percentage, np.average(scores['f1_micro'])))
        results[method]['f1_weighted'].append((percentage, np.average(scores['f1_weighted'])))
        results[method]['accuracy'].append((percentage, np.average(scores['accuracy'])))

        print(f"f1 macro = {results[method]['f1_macro'][-1][-1]}\n" \
              f"f1 micro = {results[method]['f1_micro'][-1][-1]}\n" \
              f"f1 weighted = {results[method]['f1_weighted'][-1][-1]}\n" \
              f"accuracy = {results[method]['accuracy'][-1][-1]}")

with open('results.json', 'w+') as fh:
    json.dump(results, fp=fh)

