import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import collections
import os
import pickle as pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})
_since_last_flush_idx = collections.defaultdict(lambda: {})

plt.style.use('ggplot')

def tick(names):
    if type(names) == list:
        for each in names:
            _since_last_flush_idx[each] += 1
    else:
        _since_last_flush_idx[names] += 1

def plot(name, value):
    if name not in _since_last_flush_idx:
        _since_last_flush_idx[name] = 0
    _since_last_flush[name][_since_last_flush_idx[name]] = value

def flush(path='',verbose=False):
    os.makedirs(path,exist_ok=True)
    prints = []

    for name, vals in list(_since_last_flush.items()):
        prints.append("{}\t{}".format(name, np.mean(list(vals.values()))))
        _since_beginning[name].update(vals)

        x_vals = np.sort(list(_since_beginning[name].keys()))
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.savefig(os.path.join(path,name.replace(' ', '_')+'.png'))

    if verbose: print("\t".join(prints))
    _since_last_flush.clear()

    with open(os.path.join(path,'log.pkl'), 'wb') as f:
        pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)