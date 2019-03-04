import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Avg results from multiple runs')
parser.add_argument('--logdir', type=str, default=None, help="Directory contains multiple csvs")
args = parser.parse_args()

logdir = args.logdir
assert os.path.exists(logdir)

model_stats = {}
col_names = []
for fname in os.listdir(logdir):
    found_delta = 0
    fname = os.path.join(logdir, fname)
    lines = [line.strip().split(',') for line in open(fname, 'r')]
    for idx, line in enumerate(lines):
        if idx == 0 and not col_names:
            col_names = line
        if line[0] == 'Model(delta)':
            found_delta = 1
            continue
        if not found_delta:
            continue
        model = line[0]
        if model not in model_stats:
            model_stats[model] = [[float(x.strip()) for x in line[1:]]]
        else:
            model_stats[model].append([float(x.strip()) for x in line[1:]])

for k, v in model_stats.items():
    v = np.array(v)
    mean = np.mean(v, axis=0)
    std = np.std(v, axis=0)
    out_str = k + ' & '
    for m, s in zip(mean, std):
        out_str += '%.2f$_{[%.2f]}$ & ' % (m, s)
    print(out_str)