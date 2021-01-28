#!/usr/bin/env python3
import subprocess
from multiprocessing.dummy import Pool as ThreadPool
import time

def run(n):
    start = time.time()
    proc = subprocess.Popen(f"../ktsne ../data/random/random_n_{n}_64.csv 2> /dev/null", shell=True).communicate()
    end = time.time()

    return (n, end - start)

pr = 64
ns = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]

pool = ThreadPool(pr)

print("n,t")
for n, t in pool.map(run, ns):
    print(f"{n},{t}")
