#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

from sys import argv, exit

if len(argv) == 1:
    print("no filename(s) provided")
    exit(-1)

for fname in argv[1:]:
    df = pd.read_csv(fname)

    ns = np.linspace(100, len(df), 15, dtype='int')
    for n in ns:
        df_sampled = df.sample(n)

        splt = os.path.splitext(fname)
        df_sampled.to_csv(f"{splt[0]}_n_{n}{splt[1]}", index=None)
