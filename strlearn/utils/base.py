import numpy as np

def scores_to_cummean(scores):
    """
    Convert evaluator scores to accumulative mean.

    It's the best way to make reader capable to understand anything from your results.

    :type scores: array-like, shape (n_estimators, n_chunks, n_metrics)
    :param scores: Evaluation scores.

    :rtype: array-like, shape (n_estimators, n_chunks, n_metrics)
    :returns: Evaluation scores in format possible to read for human being.

    .. image:: plots/cummean.png
    """
    divider = np.arange(1,scores.shape[1]+1)
    cs = np.cumsum(scores, axis=1)
    return cs / divider[np.newaxis, :, np.newaxis]
