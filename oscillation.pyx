
import cython
from cython.parallel import prange
from libc.stdio cimport printf
from libc.math cimport sin

import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
cimport numpy as cnp


from numpy.random import uniform


# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
np.import_array()


# Fl64 = np.npy_float64

ctypedef np.npy_int64 I64
ctypedef np.npy_float64 Fl64
ctypedef np.ndarray NDArr


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef NDArr cpy_compute_Lnu(
  NDArr evt_true_nu_parent_decay_point_x,
  NDArr evt_true_nu_parent_decay_point_y,
  NDArr evt_true_nu_parent_decay_point_z,
  NDArr evt_true_vtx_pos_x,
  NDArr evt_true_vtx_pos_y,
  NDArr evt_true_vtx_pos_z):

  cdef NDArr x0 = evt_true_nu_parent_decay_point_x * 10.0 # mm
  cdef NDArr y0 = evt_true_nu_parent_decay_point_y * 10.0 # mm
  cdef NDArr z0 = evt_true_nu_parent_decay_point_z * 10.0 # mm

  cdef NDArr x1 = evt_true_vtx_pos_x                 # mm
  cdef NDArr y1 = evt_true_vtx_pos_y                 # mm
  cdef NDArr z1 = evt_true_vtx_pos_z + 280.0 * 1.0e3 # mm

  cdef NDArr L = np.sqrt((x1 - x0)**2.0 + (y1 - y0)**2.0 + (z1 - z0)**2.0) # mm

  return L


cdef double cy_survival_probability_3p1nu(
    double Enu_GeV, double Lnu_km, double sin22theta14, double dm2_41):
  cdef double root = 1.267 * dm2_41 * Lnu_km / Enu_GeV
  cdef double sinroot = np.sin(root)
  cdef double P = 1.0 - sin22theta14 * sinroot * sinroot
  return P


@cython.wraparound(False)
@cython.boundscheck(False)
cdef NDArr cy_survival_probability_3p1nu_A(
    NDArr Enu_GeV, NDArr Lnu_km, double sin22theta14, double dm2_41):
  cdef NDArr root = 1.267 * dm2_41 * Lnu_km / Enu_GeV
  cdef NDArr sinroot = np.sin(root)
  cdef NDArr P = 1.0 - sin22theta14 * sinroot * sinroot
  return P



@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void cpy_oscillate(
    NDArr true_L_nu_km,
    NDArr true_E_nu_GeV,
    NDArr true_pdg_nu,
    double dm2,
    double sin22theta,
    NDArr survival_prob_arr,
    NDArr survived_mask_arr,
    int N
):
    cdef int i
    cdef double true_nu_pdg, e_nu, l_nu, p_surv, rnd

    survival_prob_arr = cy_survival_probability_3p1nu_A(true_E_nu_GeV, true_L_nu_km, sin22theta, dm2)

    rnd_arr = uniform(0.0, 1.0, N)

    cdef double[:] survival_prob_arr_view = survival_prob_arr
    cdef double[:] survived_mask_arr_view = survived_mask_arr
    cdef int[:] true_pdg_nu_view = true_pdg_nu
    cdef double[:] rnd_arr_view = rnd_arr

    for i in prange(N, nogil=True):
      if true_pdg_nu_view[i] != 12:
        survival_prob_arr_view[i] = 1.0
        continue

      rnd = rnd_arr_view[i]
      p_surv = survival_prob_arr_view[i]
      if rnd > p_surv:
        survived_mask_arr_view[i] = 0




'''
cpdef void bulk_oscillate(
  NDArr true_L_nu_km,
  NDArr true_E_nu_GeV,
  NDArr true_pdg_nu,
  # dm2_steps,
  # sin2theta_steps,
  # dm2_arr,
  # sin22theta_arr,
  NDArr parameter_space_iterator,
  int N_params,
  NDArr survival_prob_matr,
  NDArr survived_mask_matr,
  int N, ## length of arrays
):
  cdef int i, j, k, kk
  cdef double dm2, sin22t

  cdef double[:, :] parameter_space_iterator_view = parameter_space_iterator

  cdef double[:] survival_prob_arr# = np.zeros(N)
  cdef double[:] survived_mask_arr# = np.ones(N)

  cdef double[:] survival_prob_arr_view# = survival_prob_arr
  cdef double[:] survived_mask_arr_view# = survived_mask_arr

  cdef double[:] true_L_nu_km_view = true_L_nu_km
  cdef double[:] true_E_nu_GeV_view = true_E_nu_GeV
  cdef double[:] true_pdg_nu_view = true_pdg_nu

  cdef double[:, :, :] survival_prob_matr_view = survival_prob_matr
  cdef double[:, :, :] survived_mask_matr_view = survived_mask_matr

  for kk in prange(N_params, nogil=False):
    i, j, dm2, sin22t = parameter_space_iterator_view[kk]
    i = int(i)
    j = int(j)

    survival_prob_arr_view = np.zeros(N)
    survived_mask_arr_view = np.ones(N)

    cy_oscillate(
      true_L_nu_km_view,
      true_E_nu_GeV_view,
      true_pdg_nu_view,
      dm2,
      sin22t,
      survival_prob_arr_view,
      survived_mask_arr_view,
      N
    )
    survival_prob_matr_view[i, j] = survival_prob_arr_view
    survived_mask_matr_view[i, j] = survived_mask_arr_view

    # survival_prob_arr_view = 0.0
    # survived_mask_arr_view = 1.0
#'''
