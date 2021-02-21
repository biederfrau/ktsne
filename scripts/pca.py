#!/usr/bin/env python3

import os
from sys import argv, exit
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if len(argv) < 3:
    print("please specify target dimensions")
    exit(-1)

f = argv[1]
d = int(argv[2])

_, extname = os.path.splitext(f)
if extname == ".npz":
    import scipy.sparse
    X = scipy.sparse.load_npz(f)
elif extname == ".csv":
    df = pd.read_csv(f)
    X = df.to_numpy()
elif extname == ".emb":
    df = pd.read_csv(f, header=None, sep=" ")
    X = df.to_numpy()
else:
    print("unknown extension:", extname)
    exit(-1)

print(f"original shape = {X.shape}")

if extname == ".npz":
    from sklearn.decomposition import TruncatedSVD
    X_ = TruncatedSVD(n_components=d).fit_transform(X)
else:
    X_std = StandardScaler().fit_transform(X)
    X_ = PCA(n_components=d).fit_transform(X)

print(f"shape after dimensionality reduction = {X_.shape}")

df_ = pd.DataFrame(data=X_, columns=[f"x{i+1}" for i in range(d)])
df_.to_csv("{}_d_{}.csv".format(os.path.splitext(f)[0], d), index=None)
