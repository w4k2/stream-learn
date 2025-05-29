import numpy as np
from sklearn.metrics import balanced_accuracy_score


class TriggeredRebuildPartiallyUnsupervised:
    def __init__(self, metrics=(balanced_accuracy_score,), labeling_delay=10, partial=True):
        self.metrics = metrics
        self.labeling_delay = labeling_delay
        self.partial = partial

    def process(self, stream, det, clf):
        self.scores = []
        self.label_request_chunks = []
        self.training_chunks = []

        pending_label_requests = []

        for chunk_id, (X, y) in enumerate(stream):
            if chunk_id == 0:
                if self.partial:
                    clf.partial_fit(X, y, np.unique(y))
                else:
                    clf.fit(X, y)
                pending_label_requests.append(chunk_id)
                self.label_request_chunks.append(chunk_id)
                continue

            if len(pending_label_requests) != 0:
                if chunk_id-self.labeling_delay in pending_label_requests:
                    start = stream.chunk_size * (chunk_id-self.labeling_delay)
                    end = stream.chunk_size * (chunk_id-self.labeling_delay) + stream.chunk_size

                    past_X = stream.X[start:end]
                    past_y = stream.y[start:end]

                    if self.partial:
                        clf.partial_fit(past_X, past_y, np.unique(past_y))
                    else:
                        clf.fit(past_X, past_y)

                    pending_label_requests.remove(chunk_id-self.labeling_delay)
                    self.training_chunks.append(chunk_id)

                    det.process(past_X, past_y)
                    if det._is_drift:
                        pending_label_requests.append(chunk_id)
                        self.label_request_chunks.append(chunk_id)

                else:
                    try:
                        det.empty_process()  # For Oracle detector
                    except:
                        pass

            else:
                det.process(X)  # det.process(X, np.zeros(X.shape[0]))  <- why this was with np.zeros instead of no labels like in unsupervised?
                # TODO: MD3 zmienić w process y=None
                if det._is_drift:
                    pending_label_requests.append(chunk_id)
                    self.label_request_chunks.append(chunk_id)

            preds = clf.predict(X)
            self.scores.append(metric(y, preds) for metric in self.metrics)
