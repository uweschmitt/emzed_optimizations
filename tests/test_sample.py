import numpy as np
from emzed_optimizations import chromatogram, sample_image
import cPickle


def _load():
    # we doent want to import emzed here for avoiding
    # circular dependencies, so we fake objects
    # from emzed.core.data_types.ms_types for testing:

    import os
    here = os.path.dirname(os.path.abspath(__file__))
    spectra = cPickle.load(open(os.path.join(here, "peakmap.bin"), "rb"))

    class Holder(object):
        pass

    pm = Holder()
    pm.spectra = []
    for (rt, peaks) in spectra:
        s = Holder()
        s.rt = rt
        s.peaks = peaks
        s.msLevel = 1
        pm.spectra.append(s)

    return pm


def test_chromatogram():

    pm = _load()

    rtmin = pm.spectra[0].rt
    rtmax = pm.spectra[-1].rt
    mzmin = min(min(s.peaks[:, 0]) for s in pm.spectra)
    mzmax = max(max(s.peaks[:, 0]) for s in pm.spectra)


    rts, chromo = chromatogram(pm, 0, 0, 0, 0, 1)
    assert len(rts) == 0
    assert len(chromo) == 0

    rts, chromo = chromatogram(pm, 0, 10000, 0, 10000, 2)
    assert len(rts) == 0
    assert len(chromo) == 0

    rts, chromo = chromatogram(pm, 0, 1000, rtmin, rtmin, 1)
    assert len(rts) == 1
    assert len(chromo) == 1

    rts, chromo = chromatogram(pm, 0, 1000, rtmax, rtmax, 1)
    assert len(rts) == 1
    assert len(chromo) == 1

    rts, chromo = chromatogram(pm, mzmin - 1e-5, mzmin - 1e-5, rtmin, rtmax, 1)
    assert len(rts) == 3042
    assert len(chromo) == 3042
    assert sum(chromo) == 0.0

    rts, chromo = chromatogram(pm, mzmax + 1e-5, mzmax + 1e-5, rtmin, rtmax, 1)
    assert len(rts) == 3042
    assert len(chromo) == 3042
    assert sum(chromo) == 0.0

    rts, chromo = chromatogram(pm, 0, 1000, 41, 41.5, 1)
    assert len(rts) == 1
    assert len(chromo) == 1

    assert abs(rts[0] - 41.0816) < 1e-4, rts[0]
    assert abs(chromo[0] - 951676.41) < 1e-2, chromo[0]

    rts, chromo = chromatogram(pm, 0, 1000, 41, 3000, 1)
    assert len(rts) == 3042
    assert len(chromo) == 3042

    assert abs(rts[1000] - 1003.81995) < 1e-5, rts[1000]
    assert abs(chromo[1000] - 3183594.05) < 1e-2, chromo[1000]


def py_sample(pm, rtmin, rtmax, mzmin, mzmax, w, h):
    rtmin = float(rtmin)
    rtmax = float(rtmax)
    mzmin = float(mzmin)
    mzmax = float(mzmax)
    img = np.zeros((h, w), dtype=np.float64)
    for s in pm.spectra:
        if s.rt < rtmin:
            continue
        if s.rt > rtmax:
            continue
        if s.msLevel != 1:
            continue
        x_bin = int((s.rt - rtmin) / (rtmax - rtmin) * (w - 1))

        peaks = s.peaks
        ix = (peaks[:, 0] >= mzmin) & (peaks[:, 0] <= mzmax)
        mzs = peaks[ix, 0]
        iis = peaks[ix, 1]
        mz_bin = np.floor((mzs - mzmin) / (mzmax - mzmin) * (h - 1)).astype(int)
        upd = np.bincount(mz_bin, iis)
        i1 = max(mz_bin) + 1
        img[:i1, x_bin] = img[:i1, x_bin] + upd

    return img


def test_sample():
    pm = _load()

    t0 = 41.0
    m0 = 202.0

    pm, rtmin, rtmax, mzmin, mzmax, w, h = (pm, t0, t0 + 300.0, m0, m0 + 500, 200, 400)
    img_optim = sample_image(pm, rtmin, rtmax, mzmin, mzmax, w, h)
    img_py = py_sample(pm, rtmin, rtmax, mzmin, mzmax, w, h)

    diff = np.max(np.abs(img_optim - img_py))
    assert diff == 0.0, diff



"""
@profile
def sample_mz_python():
    pm = _load()
    rtmin = pm.spectra[0].rt
    rtmax = pm.spectra[-1].rt
    mzmin = min(min(s.peaks[:, 0]) for s in pm.spectra)
    mzmax = max(max(s.peaks[:, 0]) for s in pm.spectra)

    spectra = pm.spectra
    spectra = [s for s in spectra if s.msLevel == 1]
    spectra = [ s for s in spectra if rtmin <= s.rt ]
    spectra = [ s for s in spectra if rtmax >= s.rt ]

    for s in spectra:
        mzs = s.peaks[:,0]
        imin = mzs.searchsorted(mzmin)
        imax = mzs.searchsorted(mzmax, side='right')
        s.peaks = s.peaks[imin:imax]

    return np.vstack([s.peaks for s in spectra])

@profile
def test_sample_mz_peaks():
    pm = _load()
    rtmin = pm.spectra[0].rt
    rtmax = pm.spectra[-1].rt
    mzmin = min(min(s.peaks[:, 0]) for s in pm.spectra)
    mzmax = max(max(s.peaks[:, 0]) for s in pm.spectra)

    all_peaks = sample_mz_peaks(pm, rtmin, rtmax, mzmin, mzmax)
    assert all_peaks.shape == (798730, 2), all_peaks.shape

sample_mz_python()
test_sample_mz_peaks()

"""

