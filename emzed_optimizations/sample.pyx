cimport cython
cimport numpy as np
import numpy as np

from libc.stdlib cimport malloc, free, calloc


@cython.boundscheck(False)
@cython.wraparound(False)
def chromatogram(pm, double mzmin, double mzmax, double rtmin, double rtmax, int ms_level=1):
    cdef tuple spectra = pm.spectra
    cdef size_t i0, i
    cdef double ii_sum
    cdef double mz
    cdef double rt
    cdef int msLevel
    cdef size_t n, n1

    cdef np.float64_t[:, :] peaks

    n = len(spectra)

    # we allocate more memory thatn we will need in generatl, but this is faster as we
    # avoide extra loops for looking up rt limits with costly attribute lookup:

    cdef np.ndarray chromatogram = np.zeros((n), dtype=np.float64)
    cdef np.float64_t[:] chromatogram_view = chromatogram

    cdef np.ndarray rts = np.zeros((n), dtype=np.float64)
    cdef np.float64_t[:] rts_view = rts

    i0 = 0
    for i in range(n):
        s = spectra[i]
        rt = s.rt             # avoids rich python comparision in if statement
        msLevel = s.msLevel   # avoids rich python comparision in if statement
        if msLevel == ms_level and rtmin <= rt and rt <= rtmax:
            rts_view[i0] = rt
            peaks = s.peaks
            n1 = peaks.shape[0]
            ii_sum = 0.0
            for j in range(n1):
                mz = peaks[j, 0]
                if mzmin <= mz and mz <= mzmax:
                    ii_sum += peaks[j, 1]
            chromatogram_view[i0] = ii_sum
            i0 += 1

    # this slicing is not expensive compared to implemenit look which calulates
    # thie right sizes beforehand
    return rts[:i0], chromatogram[:i0]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sample_peaks(pm, double rtmin, double rtmax, double mzmin, double mzmax, size_t n_bins,
                 int ms_level=1):

    # avoid zero division later
    assert mzmax > mzmin

    cdef double *i_sums, *mz_i_sums, *i_max
    cdef double mz, ii
    cdef int msLevel

    cdef tuple spectra = pm.spectra
    cdef np.float64_t[:, :] peaks

    cdef size_t ns, n, i, j, mz_bin
    cdef double rt

    cdef np.ndarray result

    i_sums = <double * > calloc(sizeof(double), n_bins)
    if i_sums == NULL:
        return None
    mz_i_sums = <double * > calloc(sizeof(double), n_bins)
    if mz_i_sums == NULL:
        free(i_sums)
        return None
    i_max = <double * > calloc(sizeof(double), n_bins)
    if i_max == NULL:
        free(i_sums)
        free(mz_i_sums)
        return None

    ns = len(spectra)

    # I tried splitting the loops, so that a first loop runs i up to rtmin, and a second loop
    # which runs to rtmax. I proceeded like this for the inner loop from mzmin to mzmax,
    # but this did not result in a performance gain, but the code got more complex, so I
    # just use for-loops below with full range testing in the loop bodies

    for i in range(ns):
        spec = spectra[i]
        rt = spec.rt   # avoids rich python comparision in if statement below
        if rt < rtmin:
            continue
        if rt > rtmax:
            break
        msLevel = spec.msLevel   # avoids rich python comparision in if statement below
        if msLevel != ms_level:
            continue
        peaks = spec.peaks
        n = peaks.shape[0]
        for j in range(n):
            mz = peaks[j, 0]
            if mzmin <= mz and mz <= mzmax:
                mz_bin = int((mz - mzmin) / (mzmax - mzmin) * (n_bins - 1))
                ii = peaks[j, 1]
                i_sums[mz_bin] += ii
                mz_i_sums[mz_bin] += mz * ii
                i_max[mz_bin] = max(i_max[mz_bin], ii)

    result = np.zeros((n_bins, 2), dtype=np.float64)
    peaks = result  # create view
    i = 0
    j = 0
    for i in range(n_bins):
        ii = i_sums[i]
        if ii > 0:
            peaks[j, 0] = mz_i_sums[i] / ii
            peaks[j, 1] = i_max[i]
            j += 1

    free(i_max)
    free(mz_i_sums)
    free(i_sums)
    return result[:j, :]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sample_image(pm, double rtmin, double rtmax, double mzmin, double mzmax, size_t w, size_t h,
                 ms_level=1):

    # avoid zero division later
    assert mzmax > mzmin
    assert rtmax > rtmin

    cdef np.ndarray img = np.zeros((h, w), dtype=np.float64)
    cdef np.float64_t[:, :] img_view = img
    cdef tuple spectra = pm.spectra
    cdef size_t rt_bin
    cdef float rt, mz
    cdef int msLevel
    cdef size_t n, i, j, mz_bin
    cdef np.float64_t[:, :] peaks

    cdef size_t ns = len(spectra)

    # I tried splitting the loops, so that a first loop runs i up to rtmin, and a second loop
    # which runs to rtmax. I proceeded like this for the inner loop from mzmin to mzmax,
    # but this did not result in a performance gain, but the code got more complex, so I
    # just use for-loops below with full range testing in the loop bodies

    for i in range(ns):
        spec = spectra[i]
        rt = spec.rt             # avoids rich python comparision in if statement below
        if rt < rtmin:
            continue
        if rt > rtmax:
            break
        msLevel = spec.msLevel   # avoids rich python comparision in if statement below
        if msLevel != ms_level:
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
