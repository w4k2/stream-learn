"""Drift detectors tests."""

from sklearn.naive_bayes import GaussianNB
import strlearn as sl


def get_stream():
    return sl.streams.StreamGenerator(n_chunks=50, n_drifts=2)


def test_MetaClassifier_ADWIN_TestThanTrain():
    stream = get_stream()
    detector = sl.detectors.ADWIN()
    clf = sl.detectors.MetaClassifier(base_clf=GaussianNB(), detector=detector)
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)
    assert(len(clf.detector.drift)==49)


def test_MetaClassifier_DDM_TestThanTrain():
    stream = get_stream()
    detector = sl.detectors.DDM()
    clf = sl.detectors.MetaClassifier(base_clf=GaussianNB(), detector=detector)
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)
    assert(len(clf.detector.drift)==49)

    
def test_MetaClassifier_EDDM_TestThanTrain():
    stream = get_stream()
    detector = sl.detectors.EDDM()
    clf = sl.detectors.MetaClassifier(base_clf=GaussianNB(), detector=detector)
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)
    assert(len(clf.detector.drift)==49)


def test_MetaClassifier_SDDE_TestThanTrain():
    stream = get_stream()
    detector = sl.detectors.SDDE()
    clf = sl.detectors.MetaClassifier(base_clf=GaussianNB(), detector=detector)
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)
    assert(len(clf.detector.drift)==49)


def test_dderror():
    drifts=[1,3,90,180]    
    detections=[1,3,90,180]
    err = sl.detectors.utils.dderror(drifts, detections)
    assert(err==(0, 0, 0))
