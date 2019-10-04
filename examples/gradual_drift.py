# -*- coding: utf-8 -*-
"""
==========================
Incremental drift
==========================
This example shows a basic stream processing using WAE algorithm.

"""

# Authors: Paweł Ksieniewicz <pawel.ksieniewicz@pwr.edu.pl>
# License: MIT

from strlearn.generators import DriftedStream
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


stream = DriftedStream(
    n_chunks=100,
    chunk_size=100,
    random_state=105,
    n_features=2,
    n_classes=2,
    n_drifts=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    sigmoid_spacing=5,
    n_clusters_per_class=1,
)

checkpoints = np.linspace(0, stream.n_chunks - 1, 8).astype(int)

fig = plt.figure(constrained_layout=True, figsize=(8, 4))

gs = GridSpec(3, len(checkpoints), figure=fig)

# Scatter plots
a, b = [], []
for i in range(100):
    X, y = stream.get_chunk()

    start, end = (stream.chunk_size * i, stream.chunk_size * i + stream.chunk_size)
    cs = stream.concept_selector[start:end]
    a.append(np.sum(cs == 0))
    b.append(np.sum(cs == 1))

    if i in checkpoints:
        index = np.where(checkpoints == i)[0][0]
        ax = fig.add_subplot(gs[2, index])
        ax.scatter(X[:, 0], X[:, 1], c=y, s=10, alpha=0.5, cmap="bwr")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xticks([])
        ax.set_yticks([])

# Concept presence
ax = fig.add_subplot(gs[1, :])
ax.set_title("Concept presence")
ax.plot(a, c="black", ls=":")
ax.plot(b, c="black", ls="--")
ax.set_ylim(-10, stream.chunk_size + 10)
ax.set_xticks(checkpoints)

# Periodical sigmoid
ax = fig.add_subplot(gs[0, :])
ax.set_title("Periodical sigmoid (ss=%.1f)" % stream.sigmoid_spacing)
ax.plot(stream.period_sigmoid, lw=1, c="black")


plt.savefig("gradual_drift.png")
