import numpy as np

def expected_calibration_error(confidences, correct, n_bins=10):
    confidences=np.asarray(confidences); correct=np.asarray(correct).astype(float)
    bins=np.linspace(0,1,n_bins+1); ece=0.0
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(confidences>lo)&(confidences<=hi)
        if m.any(): ece += m.mean()*abs(correct[m].mean()-confidences[m].mean())
    return float(ece)
