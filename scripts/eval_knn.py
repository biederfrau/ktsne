#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys

import re

from sklearn.neighbors import NearestNeighbors

if len(sys.argv) < 2:
    print("please specify filename")
    sys.exit(-1)

f = sys.argv[1]
df = pd.read_csv(f, header=None if "bhtsne" in f else 0)

dataset = re.search('([^_]*)_(.*)_d', f).group(2)

label_f = os.path.join('..', 'data', dataset, dataset + '_labels.txt')
labels = pd.read_csv(label_f, header=None)

X = df.to_numpy()
y = labels.to_numpy()

k = 5 if len(sys.argv) < 3 else int(sys.argv[2])
nn = NearestNeighbors(n_neighbors=k).fit(X)

_, indices = nn.kneighbors(X)
matches = []

for row in indices:
    truth_label = y[row[0]]
    match = 0

    for i in row[1:]:
        if y[i] == truth_label:
            match += 1

    match /= k-1
    matches.append(match)

print(np.average(matches))
