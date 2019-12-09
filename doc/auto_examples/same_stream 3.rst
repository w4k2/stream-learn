.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_same_stream.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_same_stream.py:


==========================
Incremental drift
==========================
This example shows a basic stream processing using WAE algorithm.




.. code-block:: python


    # Authors: Paweł Ksieniewicz <pawel.ksieniewicz@pwr.edu.pl>
    # License: MIT


    import numpy as np
    from strlearn.streams import StreamGenerator
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec


    mcargs = {
        "n_classes": 2,
        "n_chunks": 100,
        "chunk_size": 500,
        "random_state": 5,
        "n_features": 2,
        "n_informative": 2,
        "n_redundant": 0,
        "n_repeated": 0,
        "n_features": 2,
        "n_clusters_per_class": 1,
    }

    streams = {
        "9_first": StreamGenerator(
            n_drifts=3,
            concept_sigmoid_spacing=5,
            reocurring=True,
            incremental=True,
            weights=(2, 5, 0.9),
            **mcargs
        ),
        "9_second": StreamGenerator(n_drifts=1, concept_sigmoid_spacing=5, **mcargs),
    }

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
                ax.axis("off")
                ax.grid(color="r", linestyle="-", linewidth=2)

        # Concept presence
        ax = fig.add_subplot(gs[1, :])
        ax.set_title("Incremental drift", fontsize=8)
        if not stream.incremental:
            ax.set_title("Concept presence", fontsize=8)
            ax.plot(a, c="red", ls=":", label="A")
        if stream.n_drifts > 0 and not stream.incremental:
            ax.plot(b, c="green", ls=":", label="B")
        if stream.n_drifts > 1 and not stream.reocurring and not stream.incremental:
            ax.plot(c, c="blue", ls=":", label="C")
        ax.legend(frameon=False, loc=5)
        ax.set_ylim(-10, stream.chunk_size + 10)
        ax.set_xlim(0, stream.n_chunks - 1)
        ax.set_xticks(checkpoints)
        ax.set_yticks([0, 250, 500])
        ax.grid(color="k", linestyle=":", linewidth=0.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Class presence
        ax = fig.add_subplot(gs[3, :])
        ax.set_title("Class presence", fontsize=8)
        ax.plot(A, c="red", ls="-", label="0")
        ax.plot(B, c="green", ls="-", label="1")
        if stream.n_classes > 2:
            ax.plot(C, c="blue", ls="-", label="2")
        ax.legend(frameon=False, loc=5)
        ax.set_ylim(-10, stream.chunk_size + 10)
        ax.set_xlim(0, stream.n_chunks - 1)
        ax.set_xticks(checkpoints)
        ax.set_yticks([0, 250, 500])
        ax.grid(color="k", linestyle=":", linewidth=0.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Concept Periodical sigmoid
        ax = fig.add_subplot(gs[0, :])
        if hasattr(stream, "concept_probabilities"):
            if stream.concept_sigmoid_spacing is not None:
                ax.set_title(
                    "Concept probabilities (ss=%.1f, n_drifts=%i)"
                    % (stream.concept_sigmoid_spacing, stream.n_drifts),
                    fontsize=8,
                )
            else:
                ax.set_title(
                    "Concept probabilities (n_drifts=%i)" % (stream.n_drifts), fontsize=8
                )
            ax.plot(stream.concept_probabilities, lw=1, c="black")
        else:
            ax.set_title("No concept probabilities", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, mcargs["n_chunks"] * mcargs["chunk_size"])
        ax.grid(color="k", linestyle=":", linewidth=0.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Class Periodical sigmoid
        ax = fig.add_subplot(gs[4, :])
        if hasattr(stream, "class_probabilities"):
            ax.set_title(
                "Class probabilities (ss=%.1f, n_drifts=%i, ba=%.1f)"
                % (
                    stream.class_sigmoid_spacing,
                    stream.n_balance_drifts,
                    stream.balance_amplitude,
                ),
                fontsize=8,
            )
            ax.plot(stream.class_probabilities, lw=1, c="black")
        else:
            ax.set_title("No class probabilities", fontsize=8)
            ax.set_xlim(0, mcargs["n_chunks"] * mcargs["chunk_size"])
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, mcargs["n_chunks"] * mcargs["chunk_size"])
        ax.grid(color="k", linestyle=":", linewidth=0.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.savefig("../plots/%s.png" % stream_name)

**Total running time of the script:** ( 0 minutes  0.000 seconds)


.. _sphx_glr_download_auto_examples_same_stream.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: same_stream.py <same_stream.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: same_stream.ipynb <same_stream.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
