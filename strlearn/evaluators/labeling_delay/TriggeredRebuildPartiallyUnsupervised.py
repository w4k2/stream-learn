import numpy as np

from .Evaluator import Evaluator


class TriggeredRebuildPartiallyUnsupervised(Evaluator):

    def process(self, stream, clf, det):
        self.scores = []

        for chunk_id, (X, y) in enumerate(stream):
            if chunk_id == 0:
                self.train_model(clf, X, y)
                self.labeling_process.request_annotation(X, y)
                continue

            self.labeling_process.update_time()
            if self.labeling_process.peding_labeling():
                if self.labeling_process.labels_avaliable():
                    past_X, past_y = self.labeling_process.retrive_annotated()
                    self.train_model(clf, X, y)
                    det.feed(past_X, past_y)
                    if det._is_drift:
                        self.labeling_process.request_annotation(X, y)

                else:
                    try:
                        det.empty_process()  # For Oracle detector
                    except:
                        pass

            else:
                det.feed(X)
                if det._is_drift:
                    self.labeling_process.request_annotation(X, y)

            preds = clf.predict(X)
            self.scores.append([metric(y, preds) for metric in self.metrics])

        self.scores = np.array(self.scores)
