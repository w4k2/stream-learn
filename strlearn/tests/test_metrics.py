"""Metric tests."""

import sys
import strlearn as sl
from sklearn.datasets import make_classification

sys.path.insert(0, "../..")


def test_precision_recall():
    "Calculating matrics"
    chunk_size = 100
    n_chunks = 10
    X, y = make_classification(
        n_samples=chunk_size * n_chunks, n_features=5, n_classes=2, random_state=1410
    )

    clf = sl.classifiers.AccumulatedSamplesClassifier()

    previous_chunk = None
    for chunk_id in range(n_chunks):
        print("Chunk ", chunk_id)
        chunk = (
            X[chunk_size * chunk_id : chunk_size * chunk_id + chunk_size],
            y[chunk_size * chunk_id : chunk_size * chunk_id + chunk_size],
        )

        if previous_chunk is not None:
            X_test, y_test = previous_chunk
            y_pred = clf.predict(X_test)

            precision = sl.metrics.precision(y_test, y_pred)
            recall = sl.metrics.recall(y_test, y_pred)
            gmean1 = sl.metrics.geometric_mean_score_1(y_test, y_pred)
            gmean2 = sl.metrics.geometric_mean_score_2(y_test, y_pred)
            f1 = sl.metrics.f1_score(y_test, y_pred)

        clf.partial_fit(chunk[0], chunk[1])

        previous_chunk = chunk
