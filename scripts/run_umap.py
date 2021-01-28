#!/usr/bin/env python3

import numpy as np
import pandas as pd

import sys
import os
import umap

if len(sys.argv) < 2:
    print("please specify filename")
    sys.exit(-1)

f = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) == 3 else 666
df = pd.read_csv(f)

n_neighbors = 50
min_dist = 0.1
n_components = 2
metric = 'euclidean'

X = df.values
fit = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    n_components=n_components,
    metric=metric,
    n_epochs=1000,
    random_state=seed
)

Z = fit.fit_transform(X)

df_ = pd.DataFrame(data=Z, columns=[f"x{i+1}" for i in range(Z.shape[1])])
df_.to_csv("umap_{}_s_{}.csv".format(os.path.splitext(os.path.basename(f))[0], seed), index=None)
