#!/usr/bin/env python3

import pandas as pd
import numpy as np

d = 64
ns = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]

for n in ns:
    X = np.random.rand(n, d)
    cols = [f"x{i}" for i in range(d)]

    df = pd.DataFrame(X, columns=cols)
    df.to_csv(f"../data/random_n_{n}_{d}.csv", index=None)
