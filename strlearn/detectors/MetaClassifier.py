from sklearn.base import BaseEstimator, ClassifierMixin, clone

class MetaClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_clf, detector):
        self.base_clf = base_clf
        self.detector = detector

        self.clf = clone(base_clf)
        self.prev_pred = None

    def partial_fit(self, X, y, classes):

        # If first chunk just partial fit & return
        if self.prev_pred is None:
            self.clf.partial_fit(X, y, classes)
            return self

        # Feed X, y and previous predictions to detector
        self.detector.feed(X, y, self.prev_pred)

        # If drift reset classifier
        if self.detector.drift[-1]==2:
            self.clf = clone(self.base_clf)
            
        # Partial fit
        self.clf.partial_fit(X, y, classes)

        return self

    def predict(self, X):
        self.prev_pred = self.clf.predict(X)
        return self.prev_pred