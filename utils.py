import numpy as np
from numba import jit, njit, prange

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)




@jit(parallel=True)
def compute_histograms(hist_reco_nuErecQE_survived, reco_nuErecQE_survived, binning_GeV, dm2_steps, sin2theta_steps):
  for i in prange(dm2_steps):
    for j in range(sin2theta_steps):
      hist_reco_nuErecQE_survived[i, j] = np.histogram(reco_nuErecQE_survived[i, j], bins=binning_GeV)[0]



# @jit(parallel=True)
# def compute_histograms_gamma(hist_reco_nuErecQE_survived, reco_nuErecQE_survived, binning_GeV, dm2_steps, sin2theta_steps):
#   for i in prange(dm2_steps):
#     for j in range(sin2theta_steps):
#       hist_reco_nuErecQE_survived[i, j] = np.histogram(reco_nuErecQE_survived[i, j], bins=binning_GeV)[0]



@jit(parallel=True)
def variate_histograms_with_gauss(gauss_variated_hist, hist_reco_nuErecQE_survived, N_experiments):
  for k in prange(N_experiments):
    norm_var_hist = np.random.normal(loc=hist_reco_nuErecQE_survived, scale=np.sqrt(hist_reco_nuErecQE_survived))
    gauss_variated_hist[:, :, :, k] = norm_var_hist



@njit(parallel=True)
def compute_likelihoods(
    dm2_steps,
    sin22t_steps,
    N_experiments,
    likelihood_H0, ## likelihood in null hypothesis, shape = (N_experiments)
    nominal_mc_as_data_hist, ## shape = (N_bins), normilized to data
    gauss_variated_hist, ## shape = (dm2_steps, sin22t_steps, N_bins, N_experiments), normilized to data
    likelihood_H1, ## likelihood in alternative hypothesis, shape = (dm2_steps, sin22t_steps, N_experiments)
    q_test_statistic, ## shape = (dm2_steps, sin22t_steps, N_experiments)
):
  for i in prange(dm2_steps):
    for j in range(sin22t_steps):
      for k in range(N_experiments):
        lkh_H0 = likelihood_H0[k]

        gauss_rate_to_nominal = gauss_variated_hist[i, j, :, k] / nominal_mc_as_data_hist
        gauss_rate_to_nominal[gauss_rate_to_nominal <= 0.0] = 1.0e-10
        pre_lkh_H1 = gauss_variated_hist[i, j, :, k] - nominal_mc_as_data_hist - nominal_mc_as_data_hist * np.log(gauss_rate_to_nominal)
        lkh_H1 = 2.0 * np.sum(pre_lkh_H1)
        likelihood_H1[i, j, k] = lkh_H1

        q = lkh_H0 - lkh_H1
        q_test_statistic[i, j, k] = q


@njit(parallel=True)
def compute_likelihoods_v2(
    dm2_steps,
    sin22t_steps,
    N_experiments,
    nominal_mc_as_data_hist, ## shape = (N_bins), normilized to data
    gauss_variated_hist, ## shape = (dm2_steps, sin22t_steps, N_bins, N_experiments), normilized to data
    likelihood_H1, ## likelihood in alternative hypothesis, shape = (dm2_steps, sin22t_steps, N_experiments)
):
  for i in prange(dm2_steps):
    for j in range(sin22t_steps):
      for k in range(N_experiments):
        gauss_rate_to_nominal = gauss_variated_hist[i, j, :, k] / nominal_mc_as_data_hist
        gauss_rate_to_nominal[gauss_rate_to_nominal <= 0.0] = 1.0e-10
        pre_lkh_H1 = gauss_variated_hist[i, j, :, k] - nominal_mc_as_data_hist - nominal_mc_as_data_hist * np.log(gauss_rate_to_nominal)
        lkh_H1 = 2.0 * np.sum(pre_lkh_H1)
        likelihood_H1[i, j, k] = lkh_H1


@njit(parallel=True)
def compute_likelihoods_with_gamma_v2(
    dm2_steps,
    sin22t_steps,
    N_experiments,
    nominal_mc_as_data_hist, ## shape = (N_bins), normilized to data
    gauss_variated_hist, ## shape = (dm2_steps, sin22t_steps, N_bins, N_experiments), normilized to data
    nominal_mc_as_data_hist_gamma,
    gauss_variated_hist_gamma,
    likelihood_H1, ## likelihood in alternative hypothesis, shape = (dm2_steps, sin22t_steps, N_experiments)
):
  for i in prange(dm2_steps):
    for j in range(sin22t_steps):
      for k in range(N_experiments):
        gauss_rate_to_nominal = gauss_variated_hist[i, j, :, k] / nominal_mc_as_data_hist
        gauss_rate_to_nominal[gauss_rate_to_nominal <= 0.0] = 1.0e-10
        pre_lkh_H1 = gauss_variated_hist[i, j, :, k] - nominal_mc_as_data_hist - nominal_mc_as_data_hist * np.log(gauss_rate_to_nominal)
        lkh_H1 = 2.0 * np.sum(pre_lkh_H1)
        # likelihood_H1[i, j, k] = lkh_H1

        gauss_rate_to_nominal_g = gauss_variated_hist_gamma[i, j, :, k] / nominal_mc_as_data_hist_gamma
        gauss_rate_to_nominal_g[gauss_rate_to_nominal_g <= 0.0] = 1.0e-10
        pre_lkh_H1_gamma = gauss_variated_hist_gamma[i, j, :, k] - nominal_mc_as_data_hist_gamma - nominal_mc_as_data_hist_gamma * np.log(gauss_rate_to_nominal_g)
        lkh_H1_gamma = 2.0 * np.sum(pre_lkh_H1_gamma)

        likelihood_H1[i, j, k] = lkh_H1 + lkh_H1_gamma
