# -*- coding: utf-8 -*-
"""
==========================
Incremental drift
==========================
This example shows a basic stream processing using WAE algorithm.

"""

# Authors: Paweł Ksieniewicz <pawel.ksieniewicz@pwr.edu.pl>
# License: MIT


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from strlearn.streams import StreamGenerator

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
    "2_gradual": StreamGenerator(n_drifts=1, concept_sigmoid_spacing=5, **mcargs),
    "3_incremental": StreamGenerator(
        n_drifts=1, concept_sigmoid_spacing=5, incremental=True, **mcargs
    ),
    "4_reocurring": StreamGenerator(
        n_drifts=2, concept_sigmoid_spacing=5, reocurring=True, **mcargs
    ),
    "5_nonreocurring": StreamGenerator(
        n_drifts=2, concept_sigmoid_spacing=5, reocurring=False, **mcargs
    ),
}
# streams={}
mcargs.update({"n_classes": 2, "random_state": 5})
streams.update(
    {
        "6_balanced": StreamGenerator(**mcargs),
        "7_static_imbalanced": StreamGenerator(**mcargs, weights=[0.3, 0.7]),
        "8_dynamic_imbalanced": StreamGenerator(**mcargs, weights=(2, 5, 0.9)),
        "9_disco": StreamGenerator(
            **mcargs,
            weights=(8, 5, 0.9),
            n_drifts=4,
            concept_sigmoid_spacing=5,
            reocurring=True,
            incremental=True
        ),
    }
)

for stream_name in tqdm(streams):
    print(stream_name)
    stream = streams[stream_name]

    checkpoints = np.array(list(range(mcargs["n_chunks"])))

    # Scatter plots
    for i in tqdm(range(mcargs["n_chunks"])):
        X, y = stream.get_chunk()
        if i in checkpoints:
            index = np.where(checkpoints == i)[0][0]

            plt.figure(figsize=(2, 2))
            plt.title("chunk %i" % i, fontsize=8)
            # ax = fig.add_subplot(gs[2, index])
            plt.scatter(X[:, 0], X[:, 1], c=y, s=10, alpha=0.5, cmap="brg")
            plt.xlim(-5, 5)
            plt.ylim(-5, 5)
            # ax.set_xticks([])
            # ax.set_yticks([])
            plt.axis("off")
            plt.grid(color="r", linestyle="-", linewidth=2)
            plt.tight_layout()

            plt.savefig("plots/keyframes/%s-%03i.png" % (stream_name, i))
            plt.close()
