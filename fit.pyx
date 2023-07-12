cimport cython
import numpy as np
cimport numpy as np
from numpy cimport int_t, float_t
from cython.parallel import prange, parallel


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# cpdef fit(
#     int_t N_experiments,
#     np.ndarray[float_t, ndim=1] nominal_mc_hist,
#     np.ndarray[float_t, ndim=2] parameter_space_iterator,
#     int_t N_params,
#     np.ndarray[int_t, ndim=2] survived_mask_matr,
#     np.ndarray[float_t, ndim=1] reco_nuErecQE,
#     np.ndarray[float_t, ndim=1] binning,
#     float_t MC2DATA_RATIO,
#     np.ndarray[float_t, ndim=3] norm_lkh_matr,
#     np.ndarray[float_t, ndim=3] norm_var_hist_matr,
#     np.ndarray[float_t, ndim=2] osc_hist_matr
# ):
#   cdef int_t i, j, k, N_evts_after_osc
#   cdef float_t dm2, sin22t, norm_var_lkh_pre, norm_var_lkh
#   cdef np.ndarray[int_t, ndim=1] surv_evts_mask
#   cdef np.ndarray[float_t, ndim=1] reco_nuErecQE_surv, osc_hist, norm_var_hist, norm_rate_to_nominal

#   for kk in range(N_params):
#       i, j, dm2, sin22t = parameter_space_iterator[kk]
#       i = int(i)
#       j = int(j)
#       print(i, j, dm2, sin22t)

#       surv_evts_mask = survived_mask_matr[i, j]
#       reco_nuErecQE_surv = reco_nuErecQE[surv_evts_mask == 1]
#       osc_hist, _ = np.histogram(reco_nuErecQE_surv, bins=binning)
#       N_evts_after_osc = np.sum(osc_hist)
#       osc_hist_matr[i, j] = osc_hist / MC2DATA_RATIO

#       for k in range(N_experiments):
#         norm_var_hist = np.random.normal(loc=osc_hist, scale=np.sqrt(osc_hist))
#         norm_var_hist = norm_var_hist / MC2DATA_RATIO
#         norm_var_hist[norm_var_hist < 0.0] = 0.0
#         norm_var_hist_matr[i, j, k] = norm_var_hist
#         norm_rate_to_nominal = norm_var_hist / nominal_mc_hist
#         norm_rate_to_nominal[norm_rate_to_nominal <= 0.0] = 1.0e-10
#         norm_var_lkh_pre = norm_var_hist - nominal_mc_hist - nominal_mc_hist * np.log(norm_rate_to_nominal)
#         norm_var_lkh = 2.0 * np.sum(norm_var_lkh_pre)
#         norm_lkh_matr[i, j, k] = norm_var_lkh



# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef void fit2(
#     int N_experiments,
#     np.ndarray[double, ndim=1] nominal_mc_hist,
#     np.ndarray[double, ndim=2] parameter_space_iterator,
#     int N_params,
#     np.ndarray[long, ndim=2] survived_mask_matr,
#     np.ndarray[double, ndim=1] reco_nuErecQE,
#     np.ndarray[double, ndim=1] binning,
#     double MC2DATA_RATIO,
#     np.ndarray[double, ndim=3] norm_lkh_matr,
#     np.ndarray[double, ndim=3] norm_var_hist_matr,
#     np.ndarray[double, ndim=2] osc_hist_matr
# ):
#     cdef int i, j, k, N_evts_after_osc
#     cdef double dm2, sin22t, norm_var_lkh, norm_rate_to_nominal
#     cdef np.ndarray[double, ndim=1] osc_hist, reco_nuErecQE_surv, norm_var_hist, norm_var_lkh_pre
#     for kk in range(N_params):
#         i, j, dm2, sin22t = parameter_space_iterator[kk]
#         i = int(i)
#         j = int(j)
#         print(i, j, dm2, sin22t)
#         surv_evts_mask = survived_mask_matr[i, j]
#         reco_nuErecQE_surv = reco_nuErecQE[surv_evts_mask == 1]

#         osc_hist, _ = np.histogram(reco_nuErecQE_surv, bins=binning)
#         N_evts_after_osc = np.sum(osc_hist)
#         osc_hist_matr[i, j] = osc_hist / MC2DATA_RATIO

#         for k in prange(N_experiments, nogil=True):
#             norm_var_hist = np.random.normal(loc=osc_hist, scale=np.sqrt(osc_hist))
#             norm_var_hist = norm_var_hist / MC2DATA_RATIO
#             norm_var_hist[norm_var_hist < 0.0] = 0.0
#             norm_var_hist_matr[i, j, k] = norm_var_hist
#             norm_rate_to_nominal = norm_var_hist / nominal_mc_hist
#             norm_rate_to_nominal[norm_rate_to_nominal <= 0.0] = 1.0e-10
#             norm_var_lkh_pre = norm_var_hist - nominal_mc_hist - nominal_mc_hist * np.log(norm_rate_to_nominal)
#             norm_var_lkh = 2.0 * np.sum(norm_var_lkh_pre)
#             norm_lkh_matr[i, j, k] = norm_var_lkh
