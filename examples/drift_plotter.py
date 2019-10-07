# -*- coding: utf-8 -*-
"""
==========================
Incremental drift
==========================
This example shows a basic stream processing using WAE algorithm.

"""

# Authors: Paweł Ksieniewicz <pawel.ksieniewicz@pwr.edu.pl>
# License: MIT


import numpy as np
from strlearn.streams import StreamGenerator
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


mcargs = {
    "n_classes": 2,
    "n_chunks": 100,
    "chunk_size": 100,
    "random_state": 105,
    "n_features": 2,
    "n_informative": 2,
    "n_redundant": 0,
    "n_repeated": 0,
    "n_features": 2,
    "n_clusters_per_class": 1,
}

streams = {
    "0_stationary": StreamGenerator(**mcargs),
    "2_sudden_drift": StreamGenerator(n_drifts=1, **mcargs),
    "1_incremental_drift": StreamGenerator(
        n_drifts=1, concept_sigmoid_spacing=5, **mcargs
    ),
}

for stream_name in streams:
    print(stream_name)
    stream = streams[stream_name]

    checkpoints = np.linspace(0, stream.n_chunks - 1, 8).astype(int)

    fig = plt.figure(constrained_layout=True, figsize=(8, 4))

    gs = GridSpec(3, len(checkpoints), figure=fig)

    # Scatter plots
    a, b = [], []
    for i in range(100):
        X, y = stream.get_chunk()

        start, end = (stream.chunk_size * i, stream.chunk_size * i + stream.chunk_size)

        if hasattr(stream, "concept_selector"):
            cs = stream.concept_selector[start:end]
            a.append(np.sum(cs == 0))
            b.append(np.sum(cs == 1))
        else:
            a.append(stream.chunk_size)

        if i in checkpoints:
            index = np.where(checkpoints == i)[0][0]
            ax = fig.add_subplot(gs[2, index])
            ax.scatter(X[:, 0], X[:, 1], c=y, s=10, alpha=0.5, cmap="brg")
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
    if hasattr(stream, "concept_sigmoid"):
        if stream.concept_sigmoid_spacing is not None:
            ax.set_title(
                "Concept sigmoid (ss=%.1f, n_drifts=%i)"
                % (stream.concept_sigmoid_spacing, stream.n_drifts)
            )
        else:
            ax.set_title("Concept sigmoid (n_drifts=%i)" % (stream.n_drifts))
        ax.plot(stream.concept_sigmoid, lw=1, c="black")
    else:
        ax.set_title("No concept sigmoid")
    ax.set_ylim(-0.05, 1.05)

    plt.savefig("plots/%s.png" % stream_name)
