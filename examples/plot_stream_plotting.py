"""
=============================
Plotting example data streams
=============================

Id sunt magna sint doctrina quo de enim ullamco, quorum proident adipisicing,
nam summis quid magna probant. O incurreret transferrem, ne anim elit do
nescius. De velit mandaremus, do ubi minim ingeniis.Amet singulis ita sunt dolor
ut quis reprehenderit incurreret veniam cupidatat. Culpa aut id amet proident
quo te te fugiat dolor quis, commodo sed ingeniis non tamen est laborum o quae.
Probant dolore occaecat senserit ea fugiat de senserit ne nulla. Laboris eram
pariatur ullamco, an export excepteur fidelissimae.

"""


import strlearn as sl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

cm = LinearSegmentedColormap.from_list(
    "lokomotiv", colors=[(0.3, 0.7, 0.3), (0.7, 0.3, 0.3)]
)

n_chunks = 100
chunks_plotted = np.linspace(0, n_chunks - 1, 8).astype(int)


def plot_stream(stream, filename="foo", title=""):
    fig, ax = plt.subplots(1, len(chunks_plotted), figsize=(14, 2.5))

    j = 0
    for i in range(n_chunks):
        X, y = stream.get_chunk()
        if i in chunks_plotted:
            ax[j].set_title("Chunk %i" % i)
            ax[j].scatter(X[:, 0], X[:, 1], c=y, cmap=cm, s=10, alpha=0.5)
            ax[j].set_ylim(-4, 4)
            ax[j].set_xlim(-4, 4)
            ax[j].set(aspect="equal")
            j += 1

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig("%s.png" % filename, transparent=True)


##############################################################################
# Processing
##############################################################################

concept_kwargs = {
    "n_chunks": n_chunks,
    "chunk_size": 500,
    "n_classes": 2,
    "random_state": 106,
    "n_features": 2,
    "n_informative": 2,
    "n_redundant": 0,
    "n_repeated": 0,
}

##############################################################################
# Stationary stream
##############################################################################

##############################################################################
# inne

stream = sl.streams.StreamGenerator(**concept_kwargs)

plot_stream(stream, "stationary", "Stationary stream")

##############################################################################
# Sudden drift
##############################################################################

##############################################################################
# inne

stream = sl.streams.StreamGenerator(**concept_kwargs, n_drifts=1)

plot_stream(stream, "sudden", "Stream with sudden drift")

##############################################################################
# Gradual drift
##############################################################################

##############################################################################
# inne

stream = sl.streams.StreamGenerator(
    **concept_kwargs, n_drifts=1, concept_sigmoid_spacing=5
)

plot_stream(stream, "gradual", "Stream with gradual drift")

##############################################################################
# Incremental drift
##############################################################################

##############################################################################
# inne

stream = sl.streams.StreamGenerator(
    **concept_kwargs, n_drifts=1, concept_sigmoid_spacing=5, incremental=True
)

plot_stream(stream, "incremental", "Stream with incremental drift")

##############################################################################
# Recurrent
##############################################################################

##############################################################################
# inne

stream = sl.streams.StreamGenerator(**concept_kwargs, n_drifts=2, recurring=True)

plot_stream(stream, "recurring", "Data stream with recurring drift")

##############################################################################
# Non-recurrent
##############################################################################

##############################################################################
# inne

stream = sl.streams.StreamGenerator(**concept_kwargs, n_drifts=2, recurring=False)

plot_stream(stream, "nonrecurring", "Data stream with non-recurring drift")

##############################################################################
# Static-imbalanced
##############################################################################

##############################################################################
# inne

stream = sl.streams.StreamGenerator(**concept_kwargs, weights=[0.3, 0.7])

plot_stream(stream, "static-imbalanced", "Data stream with statically imbalanced drift")


##############################################################################
# Dynamic-imbalanced
##############################################################################

##############################################################################
# inne

stream = sl.streams.StreamGenerator(**concept_kwargs, weights=(2, 5, 0.9))

plot_stream(
    stream, "dynamic-imbalanced", "Data stream with dynamically imbalanced drift"
)

##############################################################################
# DISCO
##############################################################################

##############################################################################
# inne

stream = sl.streams.StreamGenerator(
    **concept_kwargs,
    weights=(2, 5, 0.9),
    n_drifts=3,
    concept_sigmoid_spacing=5,
    recurring=True,
    incremental=True
)

plot_stream(
    stream, "disco", "Dynamically Imbalanced Stream with Concept Oscillation (DISCO)"
)
