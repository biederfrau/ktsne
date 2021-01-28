#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys

from sklearn.neighbors import NearestNeighbors

if len(sys.argv) < 2:
    print("please specify filename")
    sys.exit(-1)

f = sys.argv[1]
df = pd.read_csv(f)

X = df.loc[:, df.columns != "label"].to_numpy()
y = df["label"].to_numpy()

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
