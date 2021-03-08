#! /usr/bin/env python
"""
jobber.py
Summarize status of running jobs under SLURM scheduler.
"""

from __future__ import print_function

import os
import sys
import subprocess as sp
from collections import defaultdict
from datetime import timedelta


## parse runtime as reported by `squeue`, return as hours (floating-point)
def parse_runtime(x):
    pieces = x.split(":")
    if len(pieces) == 2:
        d = 0
        h = 0
        m, s = [int(_) for _ in pieces]
    elif len(pieces) == 3:
        h, m, s = pieces
        if "-" in h:
            d, h = [int(_) for _ in h.split("-")]
            m, s = [int(_) for _ in [m, s]]
        else:
            d = 0
            h, m, s = [int(_) for _ in [h, m, s]]
    hh = 24 * d + h + m / 60 + s / (60 * 60)
    return hh


def reformat_time(x):
    return str(timedelta(hours=x))


## query scheduler for running jobs
cmd = os.path.expandvars("squeue -u $USER")
piper = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
STAT_CODE = ["PD", "R", "CG", "S", "ST"]
STAT_DESC = ["pending", "running", "compl", "susp", "stop"]

## slurp off header line
jobs = iter(piper.stdout.readline, "")
_ = next(jobs)

## loop on jobs
counts = defaultdict(int)
runtimes = defaultdict(list)
for line in jobs:
    pieces = line.decode().strip().split()
    if not len(pieces):
        break
    counts[pieces[4]] += 1
    runtimes[pieces[4]].append(parse_runtime(pieces[5]))

## print summary
print("STATUS", "NJOBS", "MINTIME", "MAXTIME", sep="\t")
for status, desc in zip(STAT_CODE, STAT_DESC):
    max_time, min_time = "--", "--"
    if status in runtimes:
        max_time = reformat_time(max(runtimes[status]))
        min_time = reformat_time(min(runtimes[status]))
    print(desc, counts[status], min_time, max_time, sep="\t")
