import numpy as np

def dderror(drifts_idx, detections_idx):
    
    drifts_idx = np.array(drifts_idx)
    detections_idx = np.array(detections_idx)

    if len(detections_idx) == 0: # no detections
        return np.inf, np.inf, np.inf

    n_detections = len(detections_idx)
    n_drifts = len(drifts_idx)

    ddm = np.abs(drifts_idx[:, np.newaxis] - detections_idx[np.newaxis,:])

    cdri = np.min(ddm, axis=0)
    cdec = np.min(ddm, axis=1)

    d1metric = np.mean(cdri)
    d2metric = np.mean(cdec)
    cmetric = np.abs((n_drifts/n_detections)-1)

    return d1metric, d2metric, cmetric
    # d1 - detection from nearest drift
    # d2 - drift from nearest detection