#!/usr/bin/env python3

import re
import sys
import os
import subprocess
from multiprocessing.dummy import Pool as ThreadPool
import time

def run(cmd):
    start = time.time()
    FNULL = open(os.devnull, 'w')
    proc = subprocess.Popen(cmd, shell=True, stderr=FNULL, stdout=FNULL).communicate()
    end = time.time()

    replace = ['python3', '..', '/bhtsne/', '/', 'run_', 'scripts', '.py']
    cmd_ = ''.join(cmd.split()[:2])
    if cmd.startswith('..'): cmd_ = ''.join(cmd.split()[:1])

    for thingy in replace:
        cmd_ = cmd_.replace(thingy, '')

    try:
        dataset = re.search('../data/(.*)/(.*.csv)', cmd).group(2)
    except AttributeError:
        print("invalid command, could not parse dataset:", cmd)
        dataset = "???"

    print(cmd.strip(), "took", end - start, "seconds", file=sys.stderr)
    return (cmd_.strip(), dataset.strip(), end - start)

pr = 16
pool = ThreadPool(pr)

if len(sys.argv) < 1:
    print("give me a command file")
    sys.exit(-1)

with open(sys.argv[1]) as f:
    cmds = f.readlines()

print("algo,data,t")
for cmd, dataset, t in pool.map(run, cmds):
    print(f"{cmd},{dataset},{t}")
