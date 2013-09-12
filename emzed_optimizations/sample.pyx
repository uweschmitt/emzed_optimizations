cimport cython
cimport numpy as np
import numpy as np

from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
def chromatogram(pm, double mzmin, double mzmax, double rtmin, double rtmax, int msl):
    cdef list spectra = pm.spectra
    cdef int i0, i1, i
    cdef double ii_sum
    cdef double mz
    cdef double rt
    cdef int msLevel
    cdef int n

    cdef np.float32_t[:, :] peaks

    n = len(spectra)

    # we allocate more memory thatn we will need in generatl, but this is faster as we
    # avoide extra loops for looking up rt limits with costly attribute lookup:

    cdef np.ndarray chromatogram = np.zeros((n), dtype=np.float64)
    cdef np.float64_t[:] chromatogram_view = chromatogram

    cdef np.ndarray rts = np.zeros((n), dtype=np.float64)
    cdef np.float64_t[:] rts_view = rts

    # find starting index i0
    i0 = 0
    for i0 in range(n):
        s = spectra[i0]
        rt = s.rt
        msLevel = s.msLevel
        if msLevel == msl and rt >= rtmin:
            break

    # i1 is index of current spectrum of matching ms level:
    # i  counts all spctra from i0 on
    i1 = i0
    for i in range(i0, n):
        spec = spectra[i]
        msLevel = spec.msLevel
        rt = spec.rt
        if rt > rtmax:
            break
        if msLevel != msl:
            continue

        rts_view[i1] = rt
        peaks = spec.peaks
        n = peaks.shape[0]
        ii_sum = 0.0
        for j in range(n):
            mz = peaks[j, 0]
            if mzmin <= mz <= mzmax:
                ii_sum += peaks[j, 1]
        chromatogram_view[i1] = ii_sum
        i1 += 1

    # this slicing is not expensive compared to implemenit look which calulates
    # thie right sizes beforehand
    return rts[i0:i1], chromatogram[i0:i1]



@cython.boundscheck(False)
@cython.wraparound(False)
def sample_image(pm, float rtmin, float rtmax, float mzmin, float mzmax, int w, int h):

    cdef np.ndarray img = np.zeros((h, w), dtype=np.float64)
    cdef np.float64_t[:, :] img_view = img
    cdef list spectra = pm.spectra
    cdef int rt_bin
    cdef float rt, mz
    cdef int n, i, j, mz_bin
    cdef np.float32_t[:, :] peaks

    cdef int ns = len(spectra)

    # I tried splitting the loops, so that a first loop runs i up to rtmin, and a second loop
    # which runs to rtmax. I proceeded like this for the inner loop from mzmin to mzmax,
    # but this did not result in a performance gain, but the code got more comples, so I
    # just use for-loops below with full range testing in the loop bodies

    for i in range(ns):
        spec = spectra[i]
        rt = spec.rt
        if rt < rtmin:
            continue
        if rt > rtmax:
            break
        if spec.msLevel != 1:
            continue
        rt_bin = int((rt - rtmin) / (rtmax - rtmin) * (w - 1))
        peaks = spec.peaks
        n = peaks.shape[0]
        for j in range(n):
            mz = peaks[j, 0]
            if mzmin <= mz and mz <= mzmax:
                mz_bin = int((mz - mzmin) / (mzmax - mzmin) * (h - 1))
                img_view[mz_bin, rt_bin] += peaks[j, 1]
    return img
