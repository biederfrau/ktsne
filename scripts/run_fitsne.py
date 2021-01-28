#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os

import sys; sys.path.append('../FIt-SNE/')
from fast_tsne import fast_tsne

if len(sys.argv) < 2:
    print("please specify filename")
    sys.exit(-1)

f = sys.argv[1]
df = pd.read_csv(f)
seed = int(sys.argv[2]) if len(sys.argv) == 3 else 666

X = df.to_numpy()
Z = fast_tsne(X, perplexity=50, max_iter=1000, learning_rate=200, nthreads=8)

df_ = pd.DataFrame(data=Z, columns=[f"x{i+1}" for i in range(Z.shape[1])])
df_.to_csv("fitsne_{}_s_{}.csv".format(os.path.splitext(os.path.basename(f))[0], seed), index=None)
