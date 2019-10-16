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
    "n_classes": 3,
    "n_chunks": 100,
    "chunk_size": 500,
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
    "1_sudden": StreamGenerator(n_drifts=1, **mcargs),
    "2_incremental": StreamGenerator(n_drifts=1, concept_sigmoid_spacing=5, **mcargs),
    "3_reocurring": StreamGenerator(
        n_drifts=2, concept_sigmoid_spacing=5, reocurring=True, **mcargs
    ),
    "4_nonreocurring": StreamGenerator(
        n_drifts=2, concept_sigmoid_spacing=5, reocurring=False, **mcargs
    ),
}

mcargs.update({"n_classes": 2})
streams.update(
    {
        "5_balanced": StreamGenerator(**mcargs),
        "6_static_imbalanced": StreamGenerator(**mcargs, weights=[0.3, 0.7]),
        "7_dynamic_imbalanced": StreamGenerator(**mcargs, weights=(2, 5, 0.9)),
        "8_it_all": StreamGenerator(
            **mcargs,
            weights=(2, 5, 0.9),
            n_drifts=1,
            concept_sigmoid_spacing=5,
            reocurring=True
        ),
    }
)

for stream_name in streams:
    print(stream_name)
    stream = streams[stream_name]

    checkpoints = np.linspace(0, stream.n_chunks - 1, 8).astype(int)

    fig = plt.figure(constrained_layout=True, figsize=(8, 6))

    gs = GridSpec(5, len(checkpoints), figure=fig)

    # Scatter plots
    a, b, c = [], [], []
    A, B, C = [], [], []
    for i in range(100):
        X, y = stream.get_chunk()

        start, end = (stream.chunk_size * i, stream.chunk_size * i + stream.chunk_size)

        if hasattr(stream, "concept_selector"):
            cs = stream.concept_selector[start:end]
            a.append(np.sum(cs == 0))
            b.append(np.sum(cs == 1))
            c.append(np.sum(cs == 2))
        else:
            a.append(stream.chunk_size)

        if hasattr(stream, "class_selector"):
            cs = stream.class_selector[start:end]
            A.append(np.sum(cs == 0))
            B.append(np.sum(cs == 1))
            C.append(np.sum(cs == 2))

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
    ax.plot(a, c="red", ls=":", label="0")
    ax.plot(b, c="green", ls=":", label="1")
    ax.plot(c, c="blue", ls=":", label="2")
    ax.legend()
    ax.set_ylim(-10, stream.chunk_size + 10)
    ax.set_xticks(checkpoints)

    # Class presence
    ax = fig.add_subplot(gs[3, :])
    ax.set_title("Class presence")
    ax.plot(A, c="red", ls="-", label="0")
    ax.plot(B, c="green", ls="-", label="1")
    ax.plot(C, c="blue", ls="-", label="2")
    ax.legend()
    ax.set_ylim(-10, stream.chunk_size + 10)
    ax.set_xticks(checkpoints)

    # Concept Periodical sigmoid
    ax = fig.add_subplot(gs[0, :])
    if hasattr(stream, "concept_probabilities"):
        if stream.concept_sigmoid_spacing is not None:
            ax.set_title(
                "Concept probabilities (ss=%.1f, n_drifts=%i)"
                % (stream.concept_sigmoid_spacing, stream.n_drifts)
            )
        else:
            ax.set_title("Concept probabilities (n_drifts=%i)" % (stream.n_drifts))
        ax.plot(stream.concept_probabilities, lw=1, c="black")
    else:
        ax.set_title("No concept probabilities")
        ax.set_xlim(0, mcargs["n_chunks"] * mcargs["chunk_size"])
    ax.set_ylim(-0.05, 1.05)

    # Class Periodical sigmoid
    ax = fig.add_subplot(gs[4, :])
    if hasattr(stream, "class_probabilities"):
        ax.set_title(
            "Class probabilities (ss=%.1f, n_drifts=%i, ba=%.1f)"
            % (
                stream.class_sigmoid_spacing,
                stream.n_balance_drifts,
                stream.balance_amplitude,
            )
        )
        ax.plot(stream.class_probabilities, lw=1, c="black")
    else:
        ax.set_title("No class probabilities")
        ax.set_xlim(0, mcargs["n_chunks"] * mcargs["chunk_size"])
    ax.set_ylim(-0.05, 1.05)

    plt.savefig("plots/%s.png" % stream_name)
