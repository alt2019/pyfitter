### https://chat.openai.com/chat/7b13409f-f3ed-4b88-b920-d8fc1ab4dc21
#
# https://chat.openai.com/chat/77546387-ec3f-49ce-b69b-888f6f4501b8?__cf_chl_tk=7lEo80VWrtv9A4lLcUGk.XfcwA87jlHhP7biOr8P6Os-1678261034-0-gaNycGzNG9A

import logging

# DEBUG_LEVELV_NUM = 11
# # logging.addLevelName(DEBUG_LEVELV_NUM, "DEBUGV")
# def debugv(message, *args, **kws):
#     if logging.getLogger(__name__).isEnabledFor(DEBUG_LEVELV_NUM):
#         logging.log(DEBUG_LEVELV_NUM, message, args, **kws)
# # logging.Logger.debugv = debugv

# logging.addLevelName(15, "DEBUGV")
# logging.debugv = debugv
# logging.Logger.debugv = debugv

# logging.basicConfig(level = logging.DEBUG)
logging.basicConfig(level = logging.INFO)
# logging.basicConfig(level = 15)
logger = logging.getLogger(__name__)
# logger.debugv = debugv




import numpy as np
from numba import jit, njit, prange
# import numba; numba.config.set("warnings", "error", False)
from neutrino_oscillations import cpy_compute_Lnu

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mplpdf
from matplotlib.backends.backend_pdf import PdfPages

import time

from model import OscillationModel3p1, survival_probability_3p1nu

from scipy.stats import chi2
from scipy.linalg import cholesky
from scipy.optimize import minimize

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

from iminuit import Minuit
import iminuit


from functools import partial



N_PLOTS = None # for gradio

MC2DATA_RATIO = 16.4536 # needed to be extracted from root file



'''
class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position
        self.best_score = float('inf')

class ParticleSwarmOptimization:
    def __init__(self,
                 objective_function,
                 n_particles=50,
                 n_dimensions=2,
                 max_iterations=100,
                 w=0.5,
                 c1=1,
                 c2=1,
                 dm2_min=0.0,
                 dm2_max=10.0,
                 sin22t_min=0.0,
                 sin22t_max=1.0):
        self.objective_function = objective_function
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.particles = []
        for i in range(n_particles):
            dm2 = np.random.uniform(dm2_min, dm2_max, 1)
            sin22t = np.random.uniform(sin22t_min, sin22t_max, 1)
            position = np.array((dm2[0], sin22t[0]))
            # position = np.random.uniform(-5, 5, n_dimensions)
            velocity = np.zeros(n_dimensions)
            self.particles.append(Particle(position, velocity))
        self.global_best_position = np.zeros(n_dimensions)
        self.global_best_score = float('inf')

    def optimize(self):
        for i in range(self.max_iterations):
            for particle in self.particles:
                score = self.objective_function(particle.position)
                if score < particle.best_score:
                    particle.best_position = particle.position
                    particle.best_score = score
                if score < self.global_best_score:
                    self.global_best_position = particle.position
                    self.global_best_score = score
                r1, r2 = np.random.uniform(0, 1, 2)
                particle.velocity = self.w * particle.velocity + \
                                    self.c1 * r1 * (particle.best_position - particle.position) + \
                                    self.c2 * r2 * (self.global_best_position - particle.position)
                particle.position = particle.position + particle.velocity

        return self.global_best_position, self.global_best_score

# Example usage
# def objective_function(x):
#     return (x[0] - 3) ** 2 + (x[1] - 1) ** 2

# pso = ParticleSwarmOptimization(objective_function, n_particles=100, n_dimensions=2)
# best_position, best_score = pso.optimize()

# print('Best position:', best_position)
# print('Best score:', best_score)

#'''

class Particle:
    def __init__(self, position, velocity, objective_function, bounds):
        self.position = position
        self.velocity = velocity
        self.best_position = position
        self.best_score = objective_function(position)
        self.bounds = bounds
        self.objective_function = objective_function

    def update(self, global_best_position, w, c1, c2):
        r1, r2 = np.random.uniform(0, 1, 2)
        inertial_velocity = w * self.velocity
        cognitive_velocity = c1 * r1 * (self.best_position - self.position)
        social_velocity = c2 * r2 * (global_best_position - self.position)
        self.velocity = inertial_velocity + cognitive_velocity + social_velocity
        self.position = np.clip(self.position + self.velocity, *self.bounds)
        self.position[0] = np.clip(self.position[0], 0, 40)
        self.position[1] = np.clip(self.position[1], 0, 1)
        score = self.objective_function(self.position)
        if score < self.best_score:
            self.best_position = self.position
            self.best_score = score

class ParticleSwarmOptimization:
    def __init__(self, objective_function, n_particles=50, n_dimensions=2, max_iterations=100,
                 w=0.5, c1=1, c2=1, bounds=((0, 40), (0, 1))):
        self.objective_function = objective_function
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds
        self.particles = []
        for i in range(n_particles):
            position = np.random.uniform(*bounds, n_dimensions)
            velocity = np.zeros(n_dimensions)
            self.particles.append(Particle(position, velocity, objective_function, bounds))
        self.global_best_position = np.zeros(n_dimensions)
        self.global_best_score = float('inf')

    def optimize(self):
        for i in range(self.max_iterations):
            for particle in self.particles:
                particle.update(self.global_best_position, self.w, self.c1, self.c2)
                if particle.best_score < self.global_best_score:
                    self.global_best_position = particle.best_position
                    self.global_best_score = particle.best_score
        return self.global_best_position, self.global_best_score






@jit(parallel=True)
def compute_histograms(hist_reco_nuErecQE_survived, reco_nuErecQE_survived, binning, dm2_steps, sin2theta_steps):
  for i in prange(dm2_steps):
    for j in range(sin2theta_steps):
      hist_reco_nuErecQE_survived[i, j] = np.histogram(reco_nuErecQE_survived[i, j], bins=binning*1.0e-3)[0]


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



@jit(parallel=True)
def variate_systematics_histograms(
      f0_nuisance_parameters_nominal_value, cholesky_matrix_upperdiag, Vinv,
      N_experiments, N_syst, dm2_steps, sin22t_steps,
      delta):
  for i in prange(dm2_steps):
    for j in range(sin22t_steps):
      for k in range(N_experiments):
        random_variables = np.random.normal(loc=0, scale=1, size=(N_syst))
        f_extr = f0_nuisance_parameters_nominal_value + random_variables @ cholesky_matrix_upperdiag
        delta[i, j, k] = (f_extr - f0_nuisance_parameters_nominal_value).T @ Vinv @ (f_extr - f0_nuisance_parameters_nominal_value)


@njit(parallel=True)
def compute_likelihoods_wsyst(
    dm2_steps,
    sin22t_steps,
    N_experiments,
    likelihood_H0, ## likelihood in null hypothesis, shape = (N_experiments)
    nominal_mc_as_data_hist, ## shape = (N_bins), normilized to data
    gauss_variated_hist, ## shape = (dm2_steps, sin22t_steps, N_bins, N_experiments), normilized to data
    likelihood_H1, ## likelihood in alternative hypothesis, shape = (dm2_steps, sin22t_steps, N_experiments)
    q_test_statistic, ## shape = (dm2_steps, sin22t_steps, N_experiments)
    delta_syst, ## shape = (dm2_steps, sin22t_steps, N_experiments)
):
  for i in prange(dm2_steps):
    for j in range(sin22t_steps):
      for k in range(N_experiments):
        # lkh_H0 = likelihood_H0[k]

        gauss_rate_to_nominal = gauss_variated_hist[i, j, :, k] / nominal_mc_as_data_hist
        gauss_rate_to_nominal[gauss_rate_to_nominal <= 0.0] = 1.0e-10
        pre_lkh_H1 = gauss_variated_hist[i, j, :, k] - nominal_mc_as_data_hist - nominal_mc_as_data_hist * np.log(gauss_rate_to_nominal)
        lkh_H1 = 2.0 * np.sum(pre_lkh_H1) + delta_syst[i, j, k]
        likelihood_H1[i, j, k] = lkh_H1

        # q = lkh_H0 - lkh_H1
        # q_test_statistic[i, j, k] = q


@njit(parallel=True)
def compute_cov_matr(nominal_mc_hist, toy_mc_hist):
  N = nominal_mc_hist.shape[0]
  V = np.zeros((N, N))
  for i in range(N):
    for j in range(N):
      V[i, j] = (toy_mc_hist[i] - nominal_mc_hist[i]) * (toy_mc_hist[j] - nominal_mc_hist[j]) / nominal_mc_hist[i] / nominal_mc_hist[j]
  return V


class Timer:
  def __init__(self):
    self.start_time = None
    self.note = None

  def start(self, note):
    self.start_time = time.time()
    self.note = note

  def check(self, var):
    tm = time.time()

    execution_time = tm - self.start_time
    print(f"Process '{self.note}' time check {var}: {execution_time:.5f} seconds")

  def stop(self):
    tm = time.time()

    execution_time = tm - self.start_time
    print(f"Process '{self.note}' executed in {execution_time:.5f} seconds")

    self.start_time = None
    self.note = None



class Latex:
  dm2 = R"\Delta m^2"
  sin22t = R"sin^2 2\theta"

  @staticmethod
  def generate_negloglkh_eqn():
    func_expr = f"({Latex.dm2}, {Latex.sin22t})"
    lnlog_expr = fR"\ln L{func_expr}"
    sum_expr = R"\sum_{i=1}^N"
    n_exp_i_expr = R"n^{exp}_i"f"{func_expr}"
    n_obs_i_expr = R"n^{obs}_i"
    return (
      fR"$-2 {func_expr} = 2 {sum_expr} \left("
      f"{n_exp_i_expr} - {n_obs_i_expr} + {n_obs_i_expr} "R"\ln \frac{"f"{n_obs_i_expr}"R"}{"f"{n_exp_i_expr}"R"}\right),$"
    )


class Visualizer:
  neglkheqn = R"$-2 \ln L(\Delta m^2, sin^2 2\theta) = 2 \sum_{i=1}^N \left(n^{exp}_i(\Delta m^2, sin^2 2\theta) - n^{obs}_i + n^{obs}_i \ln \frac{n^{obs}_i}{n^{ex
  explanation = (
    R"$n^{exp}_i(\Delta m^2, sin^2 2\theta)$ -- expected number of events in bin i"
    "\n"
    R"$n^{obs}_i$ -- observed number of events (nominal MC)"
  )

  def __init__(self,):
    pass

  @staticmethod
  def draw_althyplkh_wrt_eventsdistr(
        _dm2:float, _sin22t:float, dm2_model_arr, sin22theta_model_arr,
        likelihood_Halt, reco_nuErecQE_GeV, reco_nuErecQE_survived_GeV, binning_GeV
  ):
    print(np.where(dm2_model_arr == _dm2))
    print(np.where(sin22theta_model_arr == _sin22t))
    i_dm2 = np.where(dm2_model_arr == _dm2)[0][0]
    j_sin22t = np.where(sin22theta_model_arr == _sin22t)[0][0]

    lkh_H1_spec = likelihood_Halt[i_dm2, j_sin22t]
    lkh_H1_spec_mean = np.mean(lkh_H1_spec)
    se = reco_nuErecQE_survived_GeV[i_dm2, j_sin22t]

    label = fR"$\Delta m^2 = {_dm2} eV^2$, $sin^2 2\theta = {_sin22t}$"

    fig, axs = plt.subplots(1, 2, figsize=(30, 15))

    ## draw neg log likelihood histogram over experiments
    axs[0].hist(lkh_H1_spec, bins=100, histtype="step", color="blue", linewidth=2, density=True, label=label)
    axs[0].axvline(lkh_H1_spec_mean, color="red", label="likelihood mean")
    axs[0].legend(prop={"size": 18})
    axs[0].set_xlabel(f"{Visualizer.neglkheqn}\n{Visualizer.explanation}", fontsize=16)
    axs[0].set_title(f"likelihood for oscillation hypothesis with {label}", fontsize=20)

    ## draw corresponding events histogram for osc and non-osc hypothsis
    axs[1].hist(reco_nuErecQE_GeV, bins=binning_GeV, weights=[1.0 / MC2DATA_RATIO]*reco_nuErecQE_GeV.shape[0],
      histtype="step", color="black", linewidth=2, label="nominal MC events distribution")
    axs[1].hist(se, bins=binning_GeV, weights=[1.0 / MC2DATA_RATIO]*se.shape[0],
      histtype="step", color="red", linewidth=2, label=f"MC events distribution for {label}")
    axs[1].legend(prop={"size": 18})
    axs[1].xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    axs[1].set_xlabel(f"Reconstructed neutrino energy, GeV", fontsize=16)
    axs[1].set_title(f"Survived events distribution", fontsize=20)

    fig.tight_layout()
    return fig

  @staticmethod
  def draw_althyplkh_wrt_eventsdistr_v2(
        _dm2:float, _sin22t:float, dm2_model_arr, sin22theta_model_arr,
        likelihood_Halt,
        reco_nuErecQE_GeV,
        reco_nuErecQE_survived_GeV,
        reco_nuErecQE_survived_GeV_sel, # reco_nuErecQE_survived_GeV, but with fixed dm2, sin22t
        _dm2_fixed,
        _sin22t_fixed,
        binning_GeV
  ):
    print(np.where(dm2_model_arr == _dm2))
    print(np.where(sin22theta_model_arr == _sin22t))
    i_dm2 = np.where(dm2_model_arr == _dm2)[0][0]
    j_sin22t = np.where(sin22theta_model_arr == _sin22t)[0][0]

    lkh_H1_spec = likelihood_Halt[i_dm2, j_sin22t]
    lkh_H1_spec_mean = np.mean(lkh_H1_spec)
    se = reco_nuErecQE_survived_GeV[i_dm2, j_sin22t]

    label = fR"$\Delta m^2 = {_dm2} eV^2$, $sin^2 2\theta = {_sin22t}$"

    if reco_nuErecQE_survived_GeV_sel is not None and _dm2_fixed and _sin22t_fixed:
      label_fixed = fR"$\Delta m^2 = {_dm2_fixed} eV^2$, $sin^2 2\theta = {_sin22t_fixed}$"

    gs = GridSpec(nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[4, 1])

    fig = plt.figure(figsize=(30, 15))
    ax1 = plt.subplot(gs[:, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 1], sharex=ax2)

    ## draw neg log likelihood histogram over experiments
    ax1.hist(lkh_H1_spec, bins=100, histtype="step", color="blue", linewidth=2, density=True, label=label)
    ax1.axvline(lkh_H1_spec_mean, color="red", label="likelihood mean")
    ax1.legend(prop={"size": 20})
    ax1.set_xlabel(f"{Visualizer.neglkheqn}\n{Visualizer.explanation}", fontsize=20)
    ax1.set_title(f"likelihood for oscillation hypothesis with {label}", fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=15)

    ## draw corresponding events histogram for osc and non-osc hypothsis
    # nom_mc_h = ax2.hist(reco_nuErecQE_GeV, bins=binning_GeV, weights=[1.0 / MC2DATA_RATIO]*reco_nuErecQE_GeV.shape[0],
    #   histtype="step", color="black", linewidth=2, label="nominal MC events distribution")
    # osc_h = ax2.hist(se, bins=binning_GeV, weights=[1.0 / MC2DATA_RATIO]*se.shape[0],
    #   histtype="step", color="red", linewidth=2, label=f"MC events distribution for {label}")
    nom_mc_h = ax2.hist(
      reco_nuErecQE_GeV,
      bins=binning_GeV,
      weights=[1.0 / MC2DATA_RATIO]*reco_nuErecQE_GeV.shape[0],
      histtype="step",
      color="black",
      linewidth=2,
      label="nominal MC events distribution"
    )
    if reco_nuErecQE_survived_GeV_sel is not None and _dm2_fixed and _sin22t_fixed:
      nom_mc_h_fix = ax2.hist(
        reco_nuErecQE_survived_GeV_sel,
        bins=binning_GeV,
        weights=[1.0 / MC2DATA_RATIO]*reco_nuErecQE_survived_GeV_sel.shape[0],
        histtype="step",
        color="green",
        linewidth=2,
        label=f"MC events distribution for fixed {label_fixed}"
      )
    osc_h = ax2.hist(
      se,
      bins=binning_GeV,
      weights=[1.0 / MC2DATA_RATIO]*se.shape[0],
      histtype="step",
      color="red",
      linewidth=2,
      label=f"MC events distribution for {label}"
    )
    ax2.legend(prop={"size": 20})
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_xlabel(f"Reconstructed neutrino energy, GeV", fontsize=20)
    ax2.set_ylabel(f"N events per bin", fontsize=20)
    ax2.set_title(f"Survived events distribution", fontsize=24)

    if reco_nuErecQE_survived_GeV_sel is not None and _dm2_fixed and _sin22t_fixed:
      # ax3.scatter((binning_GeV[:-1] + binning_GeV[1:])/2.0, osc_h[0] / nom_mc_h_fix[0])
      ax3.scatter((binning_GeV[:-1] + binning_GeV[1:])/2.0, nom_mc_h_fix[0] / nom_mc_h[0])
    else:
      ax3.scatter((binning_GeV[:-1] + binning_GeV[1:])/2.0, osc_h[0] / nom_mc_h[0])
    # ax3.bar(binning_GeV[:-1], osc_h[0] / nom_mc_h[0], width=1.0)
    ax3.set_xlabel(f"Reconstructed neutrino energy, GeV", fontsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=15)
    ax3.set_ylabel(R"$N_{osc} / N_{nominal}$ per bin", fontsize=20)

    fig.tight_layout()
    return fig

  @staticmethod
  def draw_althyplkh_wrt_eventsdistr_v3(
        _dm2:float, _sin22t:float,
        dm2_model_arr, sin22theta_model_arr,
        survival_probability_arr,
        likelihood_Halt,
        reco_nuErecQE_GeV,
        reco_nuErecQE_survived_GeV,
        reco_nuErecQE_survived_GeV_sel, # reco_nuErecQE_survived_GeV, but with fixed dm2, sin22t
        _dm2_fixed,
        _sin22t_fixed,
        binning_GeV
  ):
    # print(np.where(dm2_model_arr == _dm2))
    # print(np.where(sin22theta_model_arr == _sin22t))
    i_dm2 = np.where(dm2_model_arr == _dm2)[0][0]
    j_sin22t = np.where(sin22theta_model_arr == _sin22t)[0][0]

    lkh_H1_spec = likelihood_Halt[i_dm2, j_sin22t]
    lkh_H1_spec_mean = np.mean(lkh_H1_spec)
    # lkh_H1_spec_mean = np.min(lkh_H1_spec)
    se = reco_nuErecQE_survived_GeV[i_dm2, j_sin22t]

    label = fR"$\Delta m^2 = {_dm2} eV^2$, $sin^2 2\theta = {_sin22t}$"

    if reco_nuErecQE_survived_GeV_sel is not None and _dm2_fixed and _sin22t_fixed:
      label_fixed = fR"$\Delta m^2 = {_dm2_fixed} eV^2$, $sin^2 2\theta = {_sin22t_fixed}$"

    gs = GridSpec(nrows=3, ncols=2, width_ratios=[1, 1], height_ratios=[4, 1, 5])

    fig = plt.figure(figsize=(30, 15))
    ax1 = plt.subplot(gs[0:2, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 1], sharex=ax2)
    ax4 = plt.subplot(gs[2, 1])

    ## draw neg log likelihood histogram over experiments
    ax1.hist(lkh_H1_spec, bins=100, histtype="step", color="blue", linewidth=2, density=True, label=label)
    ax1.axvline(lkh_H1_spec_mean, color="red", label="likelihood mean")
    ax1.legend(prop={"size": 20})
    ax1.set_xlabel(f"{Visualizer.neglkheqn}\n{Visualizer.explanation}", fontsize=20)
    ax1.set_title(f"likelihood for oscillation hypothesis with {label}", fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=15)

    ## draw corresponding events histogram for osc and non-osc hypothsis
    nom_mc_h = ax2.hist(
      reco_nuErecQE_GeV,
      bins=binning_GeV,
      weights=[1.0 / MC2DATA_RATIO]*reco_nuErecQE_GeV.shape[0],
      histtype="step",
      color="black",
      linewidth=2,
      label="nominal MC events distribution"
    )
    if reco_nuErecQE_survived_GeV_sel is not None and _dm2_fixed and _sin22t_fixed:
      nom_mc_h_fix = ax2.hist(
        reco_nuErecQE_survived_GeV_sel,
        bins=binning_GeV,
        weights=[1.0 / MC2DATA_RATIO]*reco_nuErecQE_survived_GeV_sel.shape[0],
        histtype="step",
        color="green",
        linewidth=2,
        label=f"MC events distribution for fixed {label_fixed}"
      )
    osc_h = ax2.hist(
      se,
      bins=binning_GeV,
      weights=[1.0 / MC2DATA_RATIO]*se.shape[0],
      histtype="step",
      color="red",
      linewidth=2,
      label=f"MC events distribution for {label}"
    )
    ax2.legend(prop={"size": 20})
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_xlabel(f"Reconstructed neutrino energy, GeV", fontsize=20)
    ax2.set_ylabel(f"N events per bin", fontsize=20)
    ax2.set_title(f"Survived events distribution", fontsize=24)

    if reco_nuErecQE_survived_GeV_sel is not None and _dm2_fixed and _sin22t_fixed:
      ax3.scatter((binning_GeV[:-1] + binning_GeV[1:])/2.0, osc_h[0] / nom_mc_h_fix[0])
    else:
      ax3.scatter((binning_GeV[:-1] + binning_GeV[1:])/2.0, osc_h[0] / nom_mc_h[0])
    # ax3.bar(binning_GeV[:-1], osc_h[0] / nom_mc_h[0], width=1.0)
    ax3.set_xlabel(f"Reconstructed neutrino energy, GeV", fontsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=15)
    ax3.set_ylabel(R"$N_{osc} / N_{nominal}$ per bin", fontsize=20)

    x = reco_nuErecQE_GeV
    y = survival_probability_arr[i_dm2, j_sin22t]
    x = x[y!= -1]
    y = y[y!= -1]
    # ax4.scatter(x, y)
    # ax4.set_xlim(0.0, 5.0)
    # ax4.set_ylim(0.0, 1.0)
    # ax4.set_xlabel(f"Reconstructed neutrino energy, GeV", fontsize=20)
    # ax4.tick_params(axis='both', which='major', labelsize=15)
    # ax4.set_ylabel(R"Survival probability", fontsize=20)
    # ax4.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

    # # m = ax4.hist2d(x, y, bins=[binning_GeV, np.arange(0.0, 1.0, 10)])
    m = ax4.hist2d(x, y, bins=10, range=[[0.0, 5.0], [0.0, 1.0]], norm=LogNorm())
    ax4.set_xlabel(f"Reconstructed neutrino energy, GeV", fontsize=20)
    ax4.tick_params(axis='both', which='major', labelsize=15)
    ax4.set_ylabel(R"Survival probability", fontsize=20)
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    fig.colorbar(mappable=m[3], ax=ax4)

    fig.tight_layout()
    return fig


  @staticmethod
  def draw_with_differ_params(
        _dm2_lst:list, _sin22t_lst:list, dm2_model_arr, sin22theta_model_arr,
        lkh_null_hypo, lkh_alt_hypo, colors = ["red", "blue", "green"]
  ):
    requested_indices = []
    for _dm2, _sin22t in zip(_dm2_lst, _sin22t_lst):
      i_dm2 = np.where(dm2_model_arr == _dm2)[0][0]
      j_sin22t = np.where(sin22theta_model_arr == _sin22t)[0][0]
      requested_indices.append((i_dm2, j_sin22t))

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    x = np.arange(0, 250, 0.01)

    # ax_lkh_h1.hist(lkh_null_hypo, bins=100, histtype="step", color="black", linewidth=2, density=True)
    # label = fR"$\Delta m^2 = {dm2_model_arr[10]}$, $sin^2 2\theta = {sin22theta_model_arr[10]}$"
    # ax_lkh_h1.hist(lkh_alt_hypo[10, 20], bins=100, histtype="step", color="red", linewidth=2, density=True, label=label)
    # label = fR"$\Delta m^2 = {dm2_model_arr[40]}$, $sin^2 2\theta = {sin22theta_model_arr[50]}$"
    # ax_lkh_h1.hist(lkh_alt_hypo[40, 50], bins=100, histtype="step", color="blue", linewidth=2, density=True, label=label)
    # label = fR"$\Delta m^2 = {dm2_model_arr[40]}$, $sin^2 2\theta = {sin22theta_model_arr[80]}$"
    # ax_lkh_h1.hist(lkh_alt_hypo[40, 80], bins=100, histtype="step", color="green", linewidth=2, density=True, label=label)

    for k, params_idxs in enumerate(requested_indices):
      i_dm2, j_sin22t = params_idxs
      color = colors[k % len(requested_indices)]
      label = fR"$\Delta m^2 = {dm2_model_arr[i_dm2]}$, $sin^2 2\theta = {sin22theta_model_arr[j_sin22t]}$"
      ax.hist(lkh_alt_hypo[i_dm2, j_sin22t], bins=100, histtype="step", color=color, linewidth=2, density=True, label=label)

    ax.plot(x, chi2.pdf(x, df=34), c="cyan")
    ax.legend(prop={"size": 15})
    ax.set_xlabel(f"{Visualizer.neglkheqn}\n{Visualizer.explanation}", fontsize=16)
    ax.set_title("likelihood for alternative hypothesis", fontsize=20)

    return fig


class Fitter:
  def __init__(self,
               path_with_preprocessed_files,
               mode="FHC",
               whichfgd="fgd1",
               use_syst=False,
               use_realdata=False,
               binning=None,
               model=None,
               docversion="v01",
               resolution_multiplier=1):
    self.mode = mode
    self.whichfgd = whichfgd
    self.use_realdata = use_realdata
    self.path = path_with_preprocessed_files

    self.model = model

    self.mc_truth_vars = None
    self.mc_default_vars = None
    self.data_default_vars = None

    self.reco_selelec_mom = None
    self.reco_nuErecQE = None
    self.true_selelec_pdg = None
    self.true_selelec_ppdg = None
    self.true_nu_parent_decay_point_x = None
    self.true_nu_parent_decay_point_y = None
    self.true_nu_parent_decay_point_z = None
    self.true_vtx_pos_x = None
    self.true_vtx_pos_y = None
    self.true_vtx_pos_z = None
    self.true_nu_pdg = None
    self.true_nu_ene = None

    self.binning = binning if binning else self.set_default_binning()
    self.binning_MeV = None
    self.binning_GeV = None

    self.hist_reco_nuErecQE_survived = None
    self.hist_reco_nuErecQE_survived_normed2data = None

    self.load_files()
    self.set_needed_variables()
    # self.set_default_parameter_space()

    # self.pdf = mplpdf.PdfPages(f"fitting-{mode}-{whichfgd}-v01.pdf")
    self.pdf = mplpdf.PdfPages(f"fitting2-{mode}-{whichfgd}-{docversion}.pdf")
    # self.html = None

    self.visualizer = None

  def load_files(self):
    # data-FHC-default_tree-fgd1.npz
    # data-FHC-default_tree-fgd2.npz
    # mc-FHC-allsyst_tree-fgd12.npz
    # mc-FHC-default_tree-fgd1.npz
    # mc-FHC-default_tree-fgd2.npz
    # mc-FHC-truth_tree-fgd1.npz
    # mc-FHC-truth_tree-fgd2.npz

    with np.load(f"{self.path}/mc-{self.mode}-truth_tree-{self.whichfgd}.npz", allow_pickle=True, encoding="latin1") as mc_truth_file:
      self.mc_truth_vars = mc_truth_file["preprocessed_dict"][()]
      # print(self.mc_truth_vars, type(self.mc_truth_vars))
      for k in self.mc_truth_vars:
        print(k)

    with np.load(f"{self.path}/mc-{self.mode}-default_tree-{self.whichfgd}.npz", allow_pickle=True, encoding="latin1") as mc_default_file:
      self.mc_default_vars = mc_default_file["preprocessed_dict"][()]
      # print(self.mc_default_vars, type(self.mc_default_vars))
      for k in self.mc_default_vars:
        print(k)

    if self.use_realdata:
      with np.load(f"{self.path}/data-{self.mode}-default_tree-{self.whichfgd}.npz", allow_pickle=True, encoding="latin1") as data_default_file:
        self.data_default_vars = data_default_file["preprocessed_dict"][()]
        # print(self.data_default_vars, type(self.data_default_vars))
        for k in self.data_default_vars:
          print(k)


  def set_needed_variables(self):
    self.reco_selelec_mom = [ self.mc_default_vars[key] for key in self.mc_default_vars if "selelec_mom" in key ][0]
    self.reco_nuErecQE = [ self.mc_default_vars[key] for key in self.mc_default_vars if "nuErecQE" in key ][0]

    self.true_selelec_pdg = [ self.mc_default_vars[key] for key in self.mc_default_vars if "selelec_true_pdg" in key ][0]
    self.true_selelec_ppdg = [ self.mc_default_vars[key] for key in self.mc_default_vars if "selelec_true_ppdg" in key ][0]

    self.true_nu_parent_decay_point_x = [ self.mc_default_vars[key] for key in self.mc_default_vars if "true_nu_parent_decay_point_x" in key ][0]
    self.true_nu_parent_decay_point_y = [ self.mc_default_vars[key] for key in self.mc_default_vars if "true_nu_parent_decay_point_y" in key ][0]
    self.true_nu_parent_decay_point_z = [ self.mc_default_vars[key] for key in self.mc_default_vars if "true_nu_parent_decay_point_z" in key ][0]
    self.true_vtx_pos_x = [ self.mc_default_vars[key] for key in self.mc_default_vars if "true_vtx_pos_x" in key ][0]
    self.true_vtx_pos_y = [ self.mc_default_vars[key] for key in self.mc_default_vars if "true_vtx_pos_y" in key ][0]
    self.true_vtx_pos_z = [ self.mc_default_vars[key] for key in self.mc_default_vars if "true_vtx_pos_z" in key ][0]
    self.true_nu_pdg = [ self.mc_default_vars[key] for key in self.mc_default_vars if "true_nu_pdg" in key ][0]
    self.true_nu_ene = [ self.mc_default_vars[key] for key in self.mc_default_vars if "true_nu_ene" in key ][0]

    # self.true_nu_ene = [ self.mc_truth_vars[key] for key in self.mc_truth_vars if "nu_trueE" in key ][0]
    # self.true_nu_pdg = [ self.mc_truth_vars[key] for key in self.mc_truth_vars if "nu_pdg" in key ][0]

  def set_default_binning(self):
    ### https://t2k.org/docs/technotes/158/SterileFit_v7.pdf page 16
    # νe = {200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,
    #       1600,1700,1800,1900,2000,2100,2200,2350,2500,2700,3000,3300,3500,4000,
    #       4400,5000,6000,10000}
    # γ = {200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,
    #      1700,1900,2200,2500,2800,4000,10000}

    # binning = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,1200,1300,1400,1500,
    #            1600,1700,1800,1900,2000,2100,2200,2350,2500,2700,3000,3300,3500,4000,
    #            4400,5000,6000,10000,30000]
    # binning = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,1200,1300,1400,1500,
    #            1600,1700,1800,1900,2000,2100,2200,2350,2500,2700,3000,3300,3500,4000,
    #            4400,5000,6000,8000,20000]
    binning = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,1200,1300,1400,1500,
               1600,1700,1800,1900,2000,2100,2200,2350,2500,2700,3000,3300,3500,4000,
               4400,5000,
               # 6000, 7500, 10000, 20000, 30000
              ] ## MeV
    print("N bin edges: ", len(binning))
    print("N bins: ", len(binning)-1)
    binning = np.array(binning)

    #============================
    self.binning_MeV = binning
    self.binning_GeV = self.binning_MeV * 1.0e-3
    #============================

    return binning

  def set_binning(self, fromfile: str):
    pass


  def oscillate_over_parameter_space(self):
    data = {
      "true_nu_parent_decay_point_x": self.true_nu_parent_decay_point_x,
      "true_nu_parent_decay_point_y": self.true_nu_parent_decay_point_y,
      "true_nu_parent_decay_point_z": self.true_nu_parent_decay_point_z,
      "true_vtx_pos_x": self.true_vtx_pos_x,
      "true_vtx_pos_y": self.true_vtx_pos_y,
      "true_vtx_pos_z": self.true_vtx_pos_z,
      "true_nu_ene": self.true_nu_ene,
      "true_nu_pdg": self.true_nu_pdg,
      "reco_nuErecQE": self.reco_nuErecQE,
    }

    reco_nuErecQE_survived, survival_prob = self.model.compute_oscillations(data)

    self.reco_nuErecQE_survived = reco_nuErecQE_survived

    tm = Timer()
    tm.start("histograms computing")
    hist_reco_nuErecQE_survived = np.zeros((reco_nuErecQE_survived.shape[0], reco_nuErecQE_survived.shape[1], len(self.binning)-1))
    compute_histograms(
      hist_reco_nuErecQE_survived,
      reco_nuErecQE_survived,
      self.binning,
      self.model.dm2_steps,
      self.model.sin22t_steps)
    tm.stop()

    self.hist_reco_nuErecQE_survived = hist_reco_nuErecQE_survived
    self.hist_reco_nuErecQE_survived_normed2data = self.hist_reco_nuErecQE_survived / MC2DATA_RATIO


  @staticmethod
  def compute_lkh_verbose(initial_params, fixed_params):
      dm2, sin22t = initial_params
      t_E_nu_GeV, t_L_nu_km, t_PDG_nu, r_E_nu_GeV, obs_evts_hist, binning, scale2data = fixed_params

      root = 1.267 * dm2 * t_L_nu_km / t_E_nu_GeV
      sinroot = np.sin(root)
      P_surv = 1.0 - sin22t * sinroot * sinroot

      P_surv[t_PDG_nu != 12] = 1.0

      rnd = np.random.uniform(0.0, 1.0, P_surv.shape[0])
      surv_evts_mask = np.ones(P_surv.shape[0])

      surv_evts_mask[rnd > P_surv] = 0

      surv_evts = r_E_nu_GeV[surv_evts_mask == 1]

      surv_evts_hist = np.histogram(surv_evts, bins=binning)[0]

      obs_evts_hist = obs_evts_hist / scale2data
      surv_evts_hist = surv_evts_hist / scale2data

      N_obs_tot = np.sum(obs_evts_hist)
      N_exp_tot = np.sum(surv_evts_hist)

      dN = N_exp_tot - N_obs_tot

      ratio = obs_evts_hist / surv_evts_hist

      ln_term = obs_evts_hist * np.log(ratio)
      ln_term[ln_term == np.inf] = 0.0

      neg_log_lkh = 2.0 * (dN + np.sum(ln_term))

      return neg_log_lkh

  @staticmethod
  def compute_lkh_verbose_2(dm2, sin22t, fixed_params):
      # dm2, sin22t = initial_params
      t_E_nu_GeV, t_L_nu_km, t_PDG_nu, r_E_nu_GeV, obs_evts_hist, binning, scale2data = fixed_params

      root = 1.267 * dm2 * t_L_nu_km / t_E_nu_GeV
      sinroot = np.sin(root)
      P_surv = 1.0 - sin22t * sinroot * sinroot

      P_surv[t_PDG_nu != 12] = 1.0

      rnd = np.random.uniform(0.0, 1.0, P_surv.shape[0])
      surv_evts_mask = np.ones(P_surv.shape[0])

      surv_evts_mask[rnd > P_surv] = 0

      surv_evts = r_E_nu_GeV[surv_evts_mask == 1]

      surv_evts_hist = np.histogram(surv_evts, bins=binning)[0]

      obs_evts_hist = obs_evts_hist / scale2data
      surv_evts_hist = surv_evts_hist / scale2data

      N_obs_tot = np.sum(obs_evts_hist)
      N_exp_tot = np.sum(surv_evts_hist)

      dN = N_exp_tot - N_obs_tot

      ratio = obs_evts_hist / surv_evts_hist

      ln_term = obs_evts_hist * np.log(ratio)
      ln_term[ln_term == np.inf] = 0.0

      neg_log_lkh = 2.0 * (dN + np.sum(ln_term))

      return neg_log_lkh

  @staticmethod
  def compute_lkh_verbose_3(initial_params, t_E_nu_GeV, t_L_nu_km, t_PDG_nu, r_E_nu_GeV, obs_evts_hist, binning, scale2data):
      dm2, sin22t = initial_params
      # t_E_nu_GeV, t_L_nu_km, t_PDG_nu, r_E_nu_GeV, obs_evts_hist, binning, scale2data = fixed_params

      root = 1.267 * dm2 * t_L_nu_km / t_E_nu_GeV
      sinroot = np.sin(root)
      P_surv = 1.0 - sin22t * sinroot * sinroot

      P_surv[t_PDG_nu != 12] = 1.0

      rnd = np.random.uniform(0.0, 1.0, P_surv.shape[0])
      surv_evts_mask = np.ones(P_surv.shape[0])

      surv_evts_mask[rnd > P_surv] = 0

      surv_evts = r_E_nu_GeV[surv_evts_mask == 1]

      surv_evts_hist = np.histogram(surv_evts, bins=binning)[0]

      obs_evts_hist = obs_evts_hist / scale2data
      surv_evts_hist = surv_evts_hist / scale2data

      N_obs_tot = np.sum(obs_evts_hist)
      N_exp_tot = np.sum(surv_evts_hist)

      dN = N_exp_tot - N_obs_tot

      ratio = obs_evts_hist / surv_evts_hist

      ln_term = obs_evts_hist * np.log(ratio)
      ln_term[ln_term == np.inf] = 0.0

      neg_log_lkh = 2.0 * (dN + np.sum(ln_term))

      return neg_log_lkh


  def fit_v4(self, N_toy_experiments, outfile=None):
    tm = Timer()
    tm.start("likelihood minimization")

    ###
    true_L_nu_mm = cpy_compute_Lnu(
      self.true_nu_parent_decay_point_x,
      self.true_nu_parent_decay_point_y,
      self.true_nu_parent_decay_point_z,
      self.true_vtx_pos_x,
      self.true_vtx_pos_y,
      self.true_vtx_pos_z,
    )

    true_L_nu_km = true_L_nu_mm * 1.0e-6

    true_E_nu_GeV = self.true_nu_ene * 1.0e-3
    true_pdg_nu = self.true_nu_pdg

    obs_evts_hist = np.histogram(self.reco_nuErecQE, bins=self.binning*1.0e-3)[0]

    # fixed = (true_E_nu_GeV, true_L_nu_km, true_pdg_nu, self.reco_nuErecQE, obs_evts_hist, self.binning*1.0e-3, MC2DATA_RATIO)

    t_E_nu_GeV, t_L_nu_km, t_PDG_nu, r_E_nu_GeV, obs_evts_hist, binning, scale2data = (
      true_E_nu_GeV, true_L_nu_km, true_pdg_nu, self.reco_nuErecQE, obs_evts_hist, self.binning*1.0e-3, MC2DATA_RATIO
    )

    fixed_params = (t_E_nu_GeV, t_L_nu_km, t_PDG_nu, r_E_nu_GeV, obs_evts_hist, binning, scale2data)

    # for i in range(self.model.dm2_steps):
    #   for j in range(self.model.sin22theta_steps):
    #     dm2 = self.model.dm2[i]
    #     sin22t = self.model.sin22theta[j]
    #     print(i, j, dm2, sin22t)

        # initial_params = (dm2, sin22t)

    # objective_function = partial(self.compute_lkh_verbose, fixed_params=fixed_params)

    # pso = ParticleSwarmOptimization(
    #   objective_function, n_particles=200, n_dimensions=2,
    #   # dm2_min=self.model.dm2_min,
    #   # dm2_max=self.model.dm2_max,
    #   # sin22t_min=self.model.sin22t_min,
    #   # sin22t_max=self.model.sin22t_max
    # )
    # best_position, best_score = pso.optimize()

    # print('Best position:', best_position)
    # print('Best score:', best_score)

    # objective_function = partial(self.compute_lkh_verbose_2, fixed_params=fixed_params)

    # minuit = Minuit(self.compute_lkh_verbose_2, dm2=0, sin22t=0, fixed_params=fixed_params)

    # minuit.limits["dm2"] = (self.model.dm2_min, self.model.dm2_max)
    # minuit.limits["sin22t"] = (self.model.sin22t_min, self.model.sin22t_max)
    # minuit.fix("fixed_params")

    # minuit.migrad()


    # ret = iminuit.minimize(
    #   self.compute_lkh_verbose_3,
    #   np.array((0.0, 0.5)),
    #   args=(fixed_params), method='migrad',
    #   jac=None, hess=None, hessp=None, bounds=[(0, 40), (0, 1)], constraints=None, tol=None, callback=None, options=None)

    # print(ret)

    # x_optimized = minuit.values['dm2']
    # y_optimized = minuit.values['sin22t']

    x0 = 10.0
    y0 = 0.5
    objective_function = self.compute_lkh_verbose_3
    result = minimize(objective_function, [x0, y0],
      args=(t_E_nu_GeV, t_L_nu_km, t_PDG_nu, r_E_nu_GeV, obs_evts_hist, binning, scale2data), method='Nelder-Mead',
      bounds=[(0, 40), (0, 1)],
      tol=1.0e-5)

    print(result)

    tm.stop()

    # print(minuit.values)
    # print(minuit.errors)
    # print(minuit.fval)
    # print(minuit.nfit)
    # print(minuit.parameters)


  def fit_v3(self, N_toy_experiments, outfile=None):

    if outfile:
      outfile_pdf = mplpdf.PdfPages(outfile)
      # outfile_025 = outfile[:-4] + "sin22t_0.25.pdf"
      # outfile_05 = outfile[:-4] + "sin22t_0.5.pdf"
      # outfile_pdf_025 = mplpdf.PdfPages(outfile_025)
      # outfile_pdf_05 = mplpdf.PdfPages(outfile_05)

    logging.info("setting 'gauss_variated_hist' array of shape (N_dm^2_steps, N_sin^2(2t)_steps, N_bins, N_toy_experiments)")
    gauss_variated_hist = np.zeros(
      (
        self.hist_reco_nuErecQE_survived.shape[0],
        self.hist_reco_nuErecQE_survived.shape[1],
        self.hist_reco_nuErecQE_survived.shape[2],
        N_toy_experiments
      )
    )

    logging.info("computing 'gauss_variated_hist' array")
    tm = Timer()
    tm.start("histograms variation")
    variate_histograms_with_gauss(gauss_variated_hist, self.hist_reco_nuErecQE_survived, N_toy_experiments)
    tm.stop()
    gauss_variated_hist[gauss_variated_hist < 0.0] = 0.0
    gauss_variated_hist_norm = gauss_variated_hist / MC2DATA_RATIO

    logging.info("setting nominal MC histogram as data")
    #=====================================================
    # nominal_mc_as_data_hist_normed = np.histogram(self.reco_nuErecQE, bins=self.binning*1.0e-3)[0] / MC2DATA_RATIO
    #=====================================================
    _dm2_fixed = 0.0
    # _dm2_fixed = 10.0
    _sin22t_fixed = 0.0
    # _sin22t_fixed = 0.25
    # _sin22t_fixed = 0.5
    # _sin22t_fixed = 0.625
    # _sin22t_fixed = 0.75
    # _sin22t_fixed = 1.0
    for _dm2_fixed in [0.0, 10.0]:
      for  _sin22t_fixed in [0.0, 0.25, 0.4, 0.5, 0.625, 0.75, 1.0]:
        if _dm2_fixed == 0.0 and _sin22t_fixed > 0.0: continue

        i_dm2 = np.where(self.model.dm2 == _dm2_fixed)[0][0]
        j_sin22t = np.where(self.model.sin22theta == _sin22t_fixed)[0][0]
        nominal_mc_as_data_hist_normed = self.hist_reco_nuErecQE_survived_normed2data[i_dm2, j_sin22t]
        #=====================================================

        logging.info("setting 'likelihood_H1' array of shape (N_dm^2_steps, N_sin^2(2t)_steps, N_toy_experiments)")
        likelihood_H1 = np.zeros(
          (
            self.hist_reco_nuErecQE_survived.shape[0],
            self.hist_reco_nuErecQE_survived.shape[1],
            N_toy_experiments
          )
        )

        logging.info("computing 'likelihood_H1' array")
        tm.start("computation of Lkh in H1")
        compute_likelihoods_v2(
          self.model.dm2_steps,
          self.model.sin22t_steps,
          N_toy_experiments,
          nominal_mc_as_data_hist_normed, ## shape = (N_bins), normilized to data
          gauss_variated_hist_norm, ## shape = (dm2_steps, sin22t_steps, N_bins, N_toy_experiments), normilized to data
          likelihood_H1, ## likelihood in alternative hypothesis, shape = (dm2_steps, sin22t_steps, N_toy_experiments)
        )
        tm.stop()


        logging.info("creating appropriate histograms")
        __sin22t_lst = [0.25, 0.5]
        __dm2_lst = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
        figs_list = []
        # figs_list_025 = []
        # figs_list_05 = []
        for __dm2 in __dm2_lst:
          for __sin22t in __sin22t_lst:
            fig = Visualizer.draw_althyplkh_wrt_eventsdistr_v2(
              _dm2=__dm2,
              _sin22t=__sin22t,
              dm2_model_arr=self.model.dm2,
              sin22theta_model_arr=self.model.sin22theta,
              likelihood_Halt=likelihood_H1,
              reco_nuErecQE_GeV=self.reco_nuErecQE,
              reco_nuErecQE_survived_GeV=self.reco_nuErecQE_survived,
              reco_nuErecQE_survived_GeV_sel=self.reco_nuErecQE_survived[i_dm2, j_sin22t],
              _dm2_fixed=_dm2_fixed,
              _sin22t_fixed=_sin22t_fixed,
              binning_GeV=self.binning*1.0e-3
            )
            # fig = Visualizer.draw_althyplkh_wrt_eventsdistr_v3(
            #   _dm2=__dm2,
            #   _sin22t=__sin22t,
            #   dm2_model_arr=self.model.dm2,
            #   sin22theta_model_arr=self.model.sin22theta,
            #   survival_probability_arr=self.model.survival_probability,
            #   likelihood_Halt=likelihood_H1,
            #   reco_nuErecQE_GeV=self.reco_nuErecQE,
            #   reco_nuErecQE_survived_GeV=self.reco_nuErecQE_survived,
            #   binning_GeV=self.binning*1.0e-3
            # )
            figs_list.append(fig)
            # if __sin22t == 0.25:
            #   figs_list_025.append(fig)
            # if __sin22t == 0.5:
            #   figs_list_05.append(fig)

        title_label = "likelihood mean for alternative hypothesis with "fR"$\Delta m^2 = {_dm2_fixed} eV^2$, $sin^2 2\theta = {_sin22t_fixed}$"
        extent = [self.model.sin22t_min, self.model.sin22t_max, self.model.dm2_min, self.model.dm2_max]
        aspect = (self.model.sin22t_max - self.model.sin22t_min) / (self.model.dm2_max - self.model.dm2_min)
        fig_lkh_alt_mean, ax_lkh_alt_mean = plt.subplots(1, 1, figsize=(20, 15))
        # m = ax_lkh_alt_mean.imshow(np.mean(likelihood_H1, axis=2)[::-1, :], extent=extent, aspect=aspect, cmap="jet")
        # m = ax_lkh_alt_mean.imshow(np.log(np.mean(likelihood_H1, axis=2)[::-1, :]), extent=extent, aspect=aspect, cmap="jet")
        m = ax_lkh_alt_mean.imshow(np.log(np.min(likelihood_H1, axis=2)[::-1, :]), extent=extent, aspect=aspect, cmap="jet")
        ax_lkh_alt_mean.set_xlabel(R"$sin^2 2\theta$", fontsize=20)
        ax_lkh_alt_mean.set_ylabel(R"$\Delta m^2, eV^2$", fontsize=20)
        ax_lkh_alt_mean.set_title(title_label, fontsize=24)
        ax_lkh_alt_mean.tick_params(axis='both', which='major', labelsize=15)
        ax_lkh_alt_mean.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax_lkh_alt_mean.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        cb = fig_lkh_alt_mean.colorbar(m, ax=ax_lkh_alt_mean)
        cb.ax.tick_params(labelsize=15)
        fig_lkh_alt_mean.tight_layout()

        fig_lkh_alt_mean_2, ax_lkh_alt_mean_2 = plt.subplots(1, 1, figsize=(20, 15))
        # m = ax_lkh_alt_mean_2.imshow(np.mean(likelihood_H1, axis=2)[::-1, :], extent=extent, aspect=aspect)
        # m = ax_lkh_alt_mean_2.imshow(np.log(np.mean(likelihood_H1, axis=2)[::-1, :]), extent=extent, aspect=aspect)
        m = ax_lkh_alt_mean_2.imshow(np.log(np.min(likelihood_H1, axis=2)[::-1, :]), extent=extent, aspect=aspect)
        ax_lkh_alt_mean_2.set_xlabel(R"$sin^2 2\theta$", fontsize=20)
        ax_lkh_alt_mean_2.set_ylabel(R"$\Delta m^2, eV^2$", fontsize=20)
        ax_lkh_alt_mean_2.set_title(title_label, fontsize=24)
        ax_lkh_alt_mean_2.tick_params(axis='both', which='major', labelsize=15)
        ax_lkh_alt_mean_2.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax_lkh_alt_mean_2.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        cb = fig_lkh_alt_mean_2.colorbar(m, ax=ax_lkh_alt_mean_2)
        cb.ax.tick_params(labelsize=15)
        fig_lkh_alt_mean_2.tight_layout()

        logging.info("writing histograms to file")
        # if not outfile:
        #   self.pdf.savefig(fig_lkh_alt_mean_2)
        #   self.pdf.savefig(fig_lkh_alt_mean)
        #   for fig in figs_list:
        #     self.pdf.savefig(fig)
        #   self.pdf.close()

        if outfile:
          # for fig in figs_list_025:
          #   outfile_pdf_025.savefig(fig)
          # for fig in figs_list_05:
          #   outfile_pdf_05.savefig(fig)
          outfile_pdf.savefig(fig_lkh_alt_mean)

    if outfile:
      outfile_pdf.close()
      # outfile_pdf_025.close()
      # outfile_pdf_05.close()


  def fit_v2(self):
    N_experiments = 2000
    gauss_variated_hist = np.zeros(
      (
        self.hist_reco_nuErecQE_survived.shape[0],
        self.hist_reco_nuErecQE_survived.shape[1],
        self.hist_reco_nuErecQE_survived.shape[2],
        N_experiments
      )
    )

    tm = Timer()
    tm.start("histograms variation")
    variate_histograms_with_gauss(gauss_variated_hist, self.hist_reco_nuErecQE_survived, N_experiments)
    tm.stop()

    gauss_variated_hist[gauss_variated_hist < 0.0] = 0.0


    ### null hypothesis
    print("self.reco_nuErecQE", self.reco_nuErecQE)
    nominal_mc_as_data = np.histogram(self.reco_nuErecQE, bins=self.binning*1.0e-3)[0] / MC2DATA_RATIO

    #=====================================================
    _dm2 = 10.0
    _sin22t = 0.5
    i_dm2 = np.where(self.model.dm2 == _dm2)[0][0]
    j_sin22t = np.where(self.model.sin22theta == _sin22t)[0][0]
    nominal_mc_as_data = self.hist_reco_nuErecQE_survived_normed2data[i_dm2, j_sin22t]
    #=====================================================

    print("nominal_mc_H0", nominal_mc_as_data)
    gauss_variated_hist_H0 = gauss_variated_hist[0, 0] / MC2DATA_RATIO  # 0, 0 -- dm2 = 0 eV^2, sin22t = 0 -- no oscillations
    print("gauss_variated_hist_H0", gauss_variated_hist_H0)
    lkh_H0 = np.zeros(N_experiments)

    X = np.reshape(nominal_mc_as_data, (nominal_mc_as_data.size, 1))
    nominal_mc_H0_arr = np.repeat(X, N_experiments, axis=1)
    gauss_rate_to_nominal = gauss_variated_hist_H0 / nominal_mc_H0_arr
    gauss_rate_to_nominal[gauss_rate_to_nominal <= 0.0] = 1.0e-10

    tm.start("computation of Lkh in H0")
    for k in range(N_experiments):
      # gauss_rate_to_nominal = gauss_variated_hist_H0[:, k] / nominal_mc_as_data
      # gauss_rate_to_nominal[gauss_rate_to_nominal <= 0.0] = 1.0e-10
      # pre_lkh_H0 = gauss_variated_hist_H0[:, k] - nominal_mc_as_data - nominal_mc_as_data * np.log(gauss_rate_to_nominal)
      pre_lkh_H0 = gauss_variated_hist_H0[:, k] - nominal_mc_as_data - nominal_mc_as_data * np.log(gauss_rate_to_nominal[:, k])
      lkh_H0[k] = 2.0 * np.sum(pre_lkh_H0)
    tm.stop()

    fig_lkh_H0, ax_lkh_H0 = plt.subplots(1, 1, figsize=(15, 10))
    ax_lkh_H0.hist(lkh_H0, bins=100)
    ax_lkh_H0.set_xlabel("likelihood", fontsize=16)
    ax_lkh_H0.set_title("likelihood for null hypothesis", fontsize=20)


    dm2_steps = self.model.dm2_steps
    sin22t_steps = self.model.sin22t_steps

    # likelihood_H1 = np.zeros_like(gauss_variated_hist)
    # q_test_statistic = np.zeros_like(gauss_variated_hist)

    likelihood_H1 = np.zeros(
      (
        self.hist_reco_nuErecQE_survived.shape[0],
        self.hist_reco_nuErecQE_survived.shape[1],
        N_experiments
      )
    )
    q_test_statistic = np.zeros_like(likelihood_H1)

    gauss_variated_hist_norm = gauss_variated_hist / MC2DATA_RATIO

    tm.start("computation of Lkh in H1")
    compute_likelihoods(
      dm2_steps,
      sin22t_steps,
      N_experiments,
      lkh_H0, ## likelihood in null hypothesis, shape = (N_experiments)
      nominal_mc_as_data, ## shape = (N_bins), normilized to data
      gauss_variated_hist_norm, ## shape = (dm2_steps, sin22t_steps, N_bins, N_experiments), normilized to data
      likelihood_H1, ## likelihood in alternative hypothesis, shape = (dm2_steps, sin22t_steps, N_experiments)
      q_test_statistic, ## shape = (dm2_steps, sin22t_steps, N_experiments)
    )
    tm.stop()


    lkh_H1_max = np.unravel_index(likelihood_H1.argmax(), likelihood_H1.shape)
    # lkh_H1_min = unravel_index(likelihood_H1.argmin(), likelihood_H1.shape)
    lkh_H0_max = lkh_H0.argmax()
    print(lkh_H1_max)
    print(likelihood_H1[lkh_H1_max])
    print(self.model.dm2[lkh_H1_max[0]], self.model.sin22theta[lkh_H1_max[1]])
    print(lkh_H0_max)
    print(lkh_H0[lkh_H0_max])
    print(lkh_H0[lkh_H0_max] - likelihood_H1[lkh_H1_max])

    dl_arr = np.zeros(N_experiments)
    dm2_max = np.zeros(N_experiments)
    sin22t_max = np.zeros(N_experiments)
    for k in range(N_experiments):
      lkh_h1 = likelihood_H1[:, :, k]
      lkh_h1_max_idx = np.unravel_index(lkh_h1.argmax(), lkh_h1.shape)
      lkh_h1_max = lkh_h1[lkh_h1_max_idx]

      dl = np.abs(lkh_H0[lkh_H0.argmax()] - lkh_h1_max)
      dl_arr[k] = dl
      dm2_max[k] = self.model.dm2[lkh_h1_max_idx[0]]
      sin22t_max[k] = self.model.sin22theta[lkh_h1_max_idx[1]]

    print(dm2_max, np.min(dm2_max), np.max(dm2_max))
    print(sin22t_max, np.min(sin22t_max), np.max(sin22t_max))


    fig_lkh_diff, ax_lkh_diff = plt.subplots(1, 1, figsize=(15, 10))
    ax_lkh_diff.hist(dl_arr, bins=100, linewidth=2, histtype="step")
    ax_lkh_diff.set_xlabel("likelihood", fontsize=16)
    ax_lkh_diff.set_title("likelihood difference between null and alternative hypothesis", fontsize=20)

    ###============================================
    neglkheqn = R"$-2 \ln L(\Delta m^2, sin^2 2\theta) = 2 \sum_{i=1}^N \left(n^{exp}_i(\Delta m^2, sin^2 2\theta) - n^{obs}_i + n^{obs}_i \ln \frac{n^{obs}_i}{n^{
    explanation = (
      R"$n^{exp}_i(\Delta m^2, sin^2 2\theta)$ -- expected number of events in bin i"
      "\n"
      R"$n^{obs}_i$ -- observed number of events (nominal MC)"
    )

    # fig_lkh_h1_spec = Visualizer.draw_althyplkh_wrt_eventsdistr(
    fig_lkh_h1_spec = Visualizer.draw_althyplkh_wrt_eventsdistr_v2(
      _dm2=5.0,
      _sin22t=1.0,
      dm2_model_arr=self.model.dm2,
      sin22theta_model_arr=self.model.sin22theta,
      likelihood_Halt=likelihood_H1,
      reco_nuErecQE_GeV=self.reco_nuErecQE,
      reco_nuErecQE_survived_GeV=self.reco_nuErecQE_survived,
      binning_GeV=self.binning*1.0e-3
    )

    # fig_lkh_h1_spec2 = Visualizer.draw_althyplkh_wrt_eventsdistr(
    fig_lkh_h1_spec2 = Visualizer.draw_althyplkh_wrt_eventsdistr_v2(
      _dm2=5.0,
      _sin22t=0.5,
      dm2_model_arr=self.model.dm2,
      sin22theta_model_arr=self.model.sin22theta,
      likelihood_Halt=likelihood_H1,
      reco_nuErecQE_GeV=self.reco_nuErecQE,
      reco_nuErecQE_survived_GeV=self.reco_nuErecQE_survived,
      binning_GeV=self.binning*1.0e-3
    )

    # fig_lkh_h1_spec3 = Visualizer.draw_althyplkh_wrt_eventsdistr(
    fig_lkh_h1_spec3 = Visualizer.draw_althyplkh_wrt_eventsdistr_v2(
      _dm2=2.0,
      _sin22t=0.2,
      dm2_model_arr=self.model.dm2,
      sin22theta_model_arr=self.model.sin22theta,
      likelihood_Halt=likelihood_H1,
      reco_nuErecQE_GeV=self.reco_nuErecQE,
      reco_nuErecQE_survived_GeV=self.reco_nuErecQE_survived,
      binning_GeV=self.binning*1.0e-3
    )

    # fig_lkh_h1_spec4 = Visualizer.draw_althyplkh_wrt_eventsdistr(
    fig_lkh_h1_spec4 = Visualizer.draw_althyplkh_wrt_eventsdistr_v2(
      _dm2=7.0,
      _sin22t=0.2,
      dm2_model_arr=self.model.dm2,
      sin22theta_model_arr=self.model.sin22theta,
      likelihood_Halt=likelihood_H1,
      reco_nuErecQE_GeV=self.reco_nuErecQE,
      reco_nuErecQE_survived_GeV=self.reco_nuErecQE_survived,
      binning_GeV=self.binning*1.0e-3
    )

    ###=====================

    # # __sin22t_lst = [0.5, 1.0]
    # __sin22t_lst = [0.5]
    # __dm2_lst = [1.0, 2.0, 3.0, 5.0, 7.0]
    # figs_list = []
    # for __dm2 in __dm2_lst:
    #   for __sin22t in __sin22t_lst:
    #     fig = Visualizer.draw_althyplkh_wrt_eventsdistr_v2(
    #       _dm2=__dm2,
    #       _sin22t=__sin22t,
    #       dm2_model_arr=self.model.dm2,
    #       sin22theta_model_arr=self.model.sin22theta,
    #       likelihood_Halt=likelihood_H1,
    #       reco_nuErecQE_GeV=self.reco_nuErecQE,
    #       reco_nuErecQE_survived_GeV=self.reco_nuErecQE_survived,
    #       binning_GeV=self.binning*1.0e-3
    #     )
    #     figs_list.append(fig)

    __sin22t_lst = [0.5]
    __dm2_lst = [1.0, 2.0, 3.0, 5.0, 7.0]
    figs_list = []
    for __dm2 in __dm2_lst:
      for __sin22t in __sin22t_lst:
        fig = Visualizer.draw_althyplkh_wrt_eventsdistr_v2(
          _dm2=__dm2,
          _sin22t=__sin22t,
          dm2_model_arr=self.model.dm2,
          sin22theta_model_arr=self.model.sin22theta,
          likelihood_Halt=likelihood_H1,
          reco_nuErecQE_GeV=self.reco_nuErecQE,
          reco_nuErecQE_survived_GeV=self.reco_nuErecQE_survived,
          binning_GeV=self.binning*1.0e-3
        )
        # fig = Visualizer.draw_althyplkh_wrt_eventsdistr_v3(
        #   _dm2=__dm2,
        #   _sin22t=__sin22t,
        #   dm2_model_arr=self.model.dm2,
        #   sin22theta_model_arr=self.model.sin22theta,
        #   survival_probability_arr=self.model.survival_probability,
        #   likelihood_Halt=likelihood_H1,
        #   reco_nuErecQE_GeV=self.reco_nuErecQE,
        #   reco_nuErecQE_survived_GeV=self.reco_nuErecQE_survived,
        #   binning_GeV=self.binning*1.0e-3
        # )
        figs_list.append(fig)

    ###============================================


    fig_lkh_h1, ax_lkh_h1 = plt.subplots(1, 1, figsize=(15, 10))
    x = np.arange(0, 250, 0.01)
    ax_lkh_h1.hist(lkh_H0, bins=100, histtype="step", color="black", linewidth=2, density=True)
    label = fR"$\Delta m^2 = {self.model.dm2[10]}$, $sin^2 2\theta = {self.model.sin22theta[10]}$"
    ax_lkh_h1.hist(likelihood_H1[10, 20], bins=100, histtype="step", color="red", linewidth=2, density=True, label=label)
    label = fR"$\Delta m^2 = {self.model.dm2[40]}$, $sin^2 2\theta = {self.model.sin22theta[50]}$"
    ax_lkh_h1.hist(likelihood_H1[40, 50], bins=100, histtype="step", color="blue", linewidth=2, density=True, label=label)
    label = fR"$\Delta m^2 = {self.model.dm2[40]}$, $sin^2 2\theta = {self.model.sin22theta[80]}$"
    ax_lkh_h1.hist(likelihood_H1[40, 80], bins=100, histtype="step", color="green", linewidth=2, density=True, label=label)
    ax_lkh_h1.plot(x, chi2.pdf(x, df=34), c="cyan")
    ax_lkh_h1.legend(prop={"size": 15})
    ax_lkh_h1.set_xlabel(f"{neglkheqn}\n{explanation}", fontsize=16)
    ax_lkh_h1.set_title("likelihood for alternative hypothesis", fontsize=20)

    fig_q_test_stat, ax_q_test_stat = plt.subplots(1, 1, figsize=(15, 10))
    label = fR"$\Delta m^2 = {self.model.dm2[10]}$, $sin^2 2\theta = {self.model.sin22theta[10]}$"
    ax_q_test_stat.hist(q_test_statistic[10, 20], bins=100, histtype="step", color="red", linewidth=2, label=label)
    label = fR"$\Delta m^2 = {self.model.dm2[40]}$, $sin^2 2\theta = {self.model.sin22theta[50]}$"
    ax_q_test_stat.hist(q_test_statistic[40, 50], bins=100, histtype="step", color="blue", linewidth=2, label=label)
    label = fR"$\Delta m^2 = {self.model.dm2[40]}$, $sin^2 2\theta = {self.model.sin22theta[80]}$"
    ax_q_test_stat.hist(q_test_statistic[40, 80], bins=100, histtype="step", color="green", linewidth=2, label=label)
    ax_q_test_stat.legend(prop={"size": 15})
    ax_q_test_stat.set_xlabel("q test statistic", fontsize=16)
    ax_q_test_stat.set_title("q test statistic", fontsize=20)


    extent = [self.model.sin22t_min, self.model.sin22t_max, self.model.dm2_min, self.model.dm2_max]
    aspect = (self.model.sin22t_max - self.model.sin22t_min) / (self.model.dm2_max - self.model.dm2_min)
    fig_lkh_alt_mean, ax_lkh_alt_mean = plt.subplots(1, 1, figsize=(15, 15))
    m = ax_lkh_alt_mean.imshow(np.mean(likelihood_H1, axis=2)[::-1, :], extent=extent, aspect=aspect, cmap="jet")
    ax_lkh_alt_mean.set_xlabel(R"$sin^2 2\theta$", fontsize=16)
    ax_lkh_alt_mean.set_ylabel(R"$\Delta m^2, eV^2$", fontsize=16)
    ax_lkh_alt_mean.set_title("likelihood mean for alternative hypothesis", fontsize=20)
    fig_lkh_alt_mean.colorbar(m, ax=ax_lkh_alt_mean)
    fig_lkh_alt_mean.tight_layout()

    fig_lkh_alt_mean_2, ax_lkh_alt_mean_2 = plt.subplots(1, 1, figsize=(15, 15))
    m = ax_lkh_alt_mean_2.imshow(np.mean(likelihood_H1, axis=2)[::-1, :], extent=extent, aspect=aspect)
    ax_lkh_alt_mean_2.set_xlabel(R"$sin^2 2\theta$", fontsize=16)
    ax_lkh_alt_mean_2.set_ylabel(R"$\Delta m^2, eV^2$", fontsize=16)
    ax_lkh_alt_mean_2.set_title("likelihood mean for alternative hypothesis", fontsize=20)
    fig_lkh_alt_mean_2.colorbar(m, ax=ax_lkh_alt_mean_2)
    fig_lkh_alt_mean_2.tight_layout()


    fig_qteststat_alt_mean, ax_qteststat_alt_mean = plt.subplots(1, 1, figsize=(15, 15))
    m = ax_qteststat_alt_mean.imshow(np.mean(q_test_statistic, axis=2)[::-1, :], extent=extent, aspect=aspect)
    # ax_qteststat_alt_mean.set_xlabel("q test statistic mean", fontsize=16)
    ax_qteststat_alt_mean.set_xlabel(R"$sin^2 2\theta$", fontsize=16)
    ax_qteststat_alt_mean.set_ylabel(R"$\Delta m^2, eV^2$", fontsize=16)
    ax_qteststat_alt_mean.set_title("q test statistic mean for alternative hypothesis", fontsize=20)
    fig_qteststat_alt_mean.colorbar(m, ax=ax_qteststat_alt_mean)
    fig_qteststat_alt_mean.tight_layout()

    fig_lkh_alt_stddev, ax_lkh_alt_stddev = plt.subplots(1, 1, figsize=(15, 15))
    m = ax_lkh_alt_stddev.imshow(np.std(likelihood_H1, axis=2)[::-1, :], extent=extent, aspect=aspect)
    # ax_lkh_alt_stddev.set_xlabel("likelihood standard deviation", fontsize=16)
    ax_lkh_alt_stddev.set_xlabel(R"$sin^2 2\theta$", fontsize=16)
    ax_lkh_alt_stddev.set_ylabel(R"$\Delta m^2, eV^2$", fontsize=16)
    ax_lkh_alt_stddev.set_title("likelihood standard deviation for alternative hypothesis", fontsize=20)
    fig_lkh_alt_stddev.colorbar(m, ax=ax_lkh_alt_stddev)
    fig_lkh_alt_stddev.tight_layout()

    fig_qteststat_alt_stddev, ax_qteststat_alt_stddev = plt.subplots(1, 1, figsize=(15, 15))
    m = ax_qteststat_alt_stddev.imshow(np.std(q_test_statistic, axis=2)[::-1, :], extent=extent, aspect=aspect)
    # ax_qteststat_alt_stddev.set_xlabel("q test statistic standard deviation", fontsize=16)
    ax_qteststat_alt_stddev.set_xlabel(R"$sin^2 2\theta$", fontsize=16)
    ax_qteststat_alt_stddev.set_ylabel(R"$\Delta m^2, eV^2$", fontsize=16)
    ax_qteststat_alt_stddev.set_title("q test statistic standard deviation for alternative hypothesis", fontsize=20)
    fig_qteststat_alt_stddev.colorbar(m, ax=ax_qteststat_alt_stddev)
    fig_qteststat_alt_stddev.tight_layout()


    #***************************************************************
    # explanation_fig, explanation_ax = plt.subplot(1, 1, figsize=(15, 10))
    # with PdfPages('likelihoods_at_some_prameters.pdf') as pdf:
    #   pdf.savefig(fig_lkh_alt_mean)
    #   pdf.savefig(fig_lkh_h1_spec)
    #   pdf.savefig(fig_lkh_h1_spec2)
    #   pdf.savefig(fig_lkh_h1_spec3)
    # plt.close(explanation_fig)

    self.pdf.savefig(fig_lkh_alt_mean_2)
    self.pdf.savefig(fig_lkh_alt_mean)
    # self.pdf.savefig(fig_lkh_h1_spec)
    # self.pdf.savefig(fig_lkh_h1_spec2)
    # self.pdf.savefig(fig_lkh_h1_spec3)
    # self.pdf.savefig(fig_lkh_h1_spec4)
    for fig in figs_list:
      self.pdf.savefig(fig)
    self.pdf.close()

    #***************************************************************


    tm.start("likelihood minimization")

    ###
    true_L_nu_mm = cpy_compute_Lnu(
      self.true_nu_parent_decay_point_x,
      self.true_nu_parent_decay_point_y,
      self.true_nu_parent_decay_point_z,
      self.true_vtx_pos_x,
      self.true_vtx_pos_y,
      self.true_vtx_pos_z,
    )

    true_L_nu_km = true_L_nu_mm * 1.0e-6

    true_E_nu_GeV = self.true_nu_ene * 1.0e-3
    true_pdg_nu = self.true_nu_pdg

    obs_evts_hist = np.histogram(self.reco_nuErecQE, bins=self.binning*1.0e-3)[0]

    # fixed = (true_E_nu_GeV, true_L_nu_km, true_pdg_nu, self.reco_nuErecQE, obs_evts_hist, self.binning*1.0e-3, MC2DATA_RATIO)

    t_E_nu_GeV, t_L_nu_km, t_PDG_nu, r_E_nu_GeV, obs_evts_hist, binning, scale2data = (
      true_E_nu_GeV, true_L_nu_km, true_pdg_nu, self.reco_nuErecQE, obs_evts_hist, self.binning*1.0e-3, MC2DATA_RATIO
    )

    # @jit
    def process_params_inner(dm2, sin22t):
      # t_E_nu_GeV, t_L_nu_km, t_PDG_nu, r_E_nu_GeV, obs_evts_hist, binning, scale2data = fixed_params
      nonlocal t_E_nu_GeV
      nonlocal t_L_nu_km
      nonlocal t_PDG_nu
      nonlocal r_E_nu_GeV
      nonlocal obs_evts_hist
      nonlocal binning
      nonlocal scale2data

      root = 1.267 * dm2 * t_L_nu_km / t_E_nu_GeV
      sinroot = np.sin(root)
      P_surv = 1.0 - sin22t * sinroot * sinroot

      P_surv[t_PDG_nu != 12] = 1.0

      rnd = np.random.uniform(0.0, 1.0, P_surv.shape[0])
      surv_evts_mask = np.ones(P_surv.shape[0])

      surv_evts_mask[rnd > P_surv] = 0

      surv_evts = r_E_nu_GeV[surv_evts_mask == 1]

      surv_evts_hist = np.histogram(surv_evts, bins=binning)[0]

      obs_evts_hist = obs_evts_hist / scale2data
      surv_evts_hist = surv_evts_hist / scale2data

      N_obs_tot = np.sum(obs_evts_hist)
      N_exp_tot = np.sum(surv_evts_hist)

      dN = N_exp_tot - N_obs_tot

      ratio = obs_evts_hist / surv_evts_hist

      ln_term = obs_evts_hist * np.log(ratio)
      ln_term[ln_term == np.inf] = 0.0

      neg_log_lkh = 2.0 * (dN + np.sum(ln_term))

      return neg_log_lkh

    # process_params(dm2, sin22t, t_E_nu_GeV, t_L_nu_km, t_PDG_nu, r_E_nu_GeV, obs_evts_hist, binning, scale2data)
    minuit = Minuit(
      # process_params,
      process_params_inner,
      # dm2=9.0,
      # sin22t=0.7,
      dm2=4.0,
      sin22t=0.7,
      # fixed_params=fixed,
      # t_E_nu_GeV=true_E_nu_GeV,
      # t_L_nu_km=true_L_nu_km,
      # t_PDG_nu=true_pdg_nu,
      # # r_E_nu_GeV=self.reco_nuErecQE*1.0e-3,
      # r_E_nu_GeV=self.reco_nuErecQE,
      # obs_evts_hist=obs_evts_hist,
      # binning=self.binning*1.0e-3,
      # scale2data=MC2DATA_RATIO,
      # limit_dm2=(0.0, 100.0),
      # limit_sin22t=(0.0, 1.0),
      # fix_t_E_nu_GeV=True,
      # fix_t_L_nu_km=True,
      # fix_t_PDG_nu=True,
      # fix_r_E_nu_GeV=True,
      # fix_obs_evts_hist=True,
      # fix_binning=True,
      # fix_scale2data=True,
    )
    minuit.limits["dm2"] = (0.0, 100.0)
    minuit.limits["sin22t"] = (0.0, 1.0)
    # minuit.fix("fixed_params")
    # minuit.fix("t_E_nu_GeV")
    # minuit.fix("t_L_nu_km")
    # minuit.fix("t_PDG_nu")
    # minuit.fix("r_E_nu_GeV")
    # minuit.fix("obs_evts_hist")
    # minuit.fix("binning")
    # minuit.fix("scale2data")

    # minuit.migrad()
    minuit.simplex().migrad()
    minuit.hesse()

    # minuit.scan(ncall=50)

    tm.stop()

    print(minuit.values)
    print(minuit.errors)
    print(minuit.fval)
    print(minuit.nfit)
    print(minuit.parameters)

    # fig, ax = plt.subplots(1, 1)
    # minuit.draw_profile("dm2")
    # plt.show()


    # plt.title("likelihood H1")
    # plt.imshow(likelihood_H1[:, :, 500][::-1, :])
    # plt.colorbar()
    # plt.show()

    # plt.title("q test statistic")
    # plt.imshow(q_test_statistic[:, :, 500][::-1, :])
    # plt.colorbar()
    # plt.show()

    # plt.title("mean likelihood H1")
    # plt.imshow(np.mean(likelihood_H1, axis=2)[::-1, :])
    # plt.colorbar()
    # plt.show()

    # plt.title("mean q test statistic")
    # plt.imshow(np.mean(q_test_statistic, axis=2)[::-1, :])
    # plt.colorbar()
    # plt.show()

    # plt.title("std dev likelihood H1")
    # plt.imshow(np.std(likelihood_H1, axis=2)[::-1, :])
    # plt.colorbar()
    # plt.show()

    # plt.title("std dev q test statistic")
    # plt.imshow(np.std(q_test_statistic, axis=2)[::-1, :])
    # plt.colorbar()
    # plt.show()

    N_PLOTS = 8+4

    return [fig_lkh_H0, fig_lkh_diff, fig_lkh_h1, fig_q_test_stat,
            fig_lkh_alt_mean, fig_qteststat_alt_mean, fig_lkh_alt_stddev, fig_qteststat_alt_stddev,
            fig_lkh_h1_spec, fig_lkh_h1_spec2, fig_lkh_h1_spec3, fig_lkh_h1_spec4]


  def process_systematics(self):
    print("="*100)
    print("=== Processing systematics")
    print("="*100)
    with np.load(f"{self.path}/mc-{self.mode}-allsyst_tree-fgd12.npz", allow_pickle=True, encoding="latin1") as mc_all_syst_file:
      self.mc_all_syst_vars = mc_all_syst_file["preprocessed_dict"][()]
      # print(self.mc_all_syst_vars, type(self.mc_all_syst_vars))
      for k in self.mc_all_syst_vars:
        print(k, self.mc_all_syst_vars[k].shape)

    print("-"*100)
    all_syst_accum_level = self.mc_all_syst_vars["fhc_mc_all_syst__accum_level"]
    all_syst__selelec_nuErecQE = self.mc_all_syst_vars["fhc_mc_all_syst__selelec_nuErecQE"]
    all_syst__selelec_mom = self.mc_all_syst_vars["fhc_mc_all_syst__selelec_mom"]

    syst_binning = [200, 600, 2000, 2500, 10000]

    # self.reco_nuErecQE = all_syst__selelec_nuErecQE
    print("self.reco_nuErecQE", self.reco_nuErecQE)
    nominal_mc_as_data = np.histogram(self.reco_nuErecQE, bins=np.array(syst_binning)*1.0e-3)[0] / MC2DATA_RATIO
    print("nominal_mc_as_data", nominal_mc_as_data)

    V = np.zeros((4, 4))
    N_toys = all_syst_accum_level.shape[1]

    N_evts_toys = np.zeros(N_toys)

    # f0_nuisance_parameters_nominal_value = np.zeros(4)

    for i in range(N_toys):
      print(f"\n\n{i}")
      print(all_syst__selelec_nuErecQE[:, i], all_syst__selelec_nuErecQE[:, i].shape)

      ### extract fgd index for applying cuts
      ### ===================================
      # print(all_syst_accum_level[:, i, :])
      al_fgd1 = all_syst_accum_level[:, i, 0]
      al_fgd2 = all_syst_accum_level[:, i, 1]
      print(al_fgd1, al_fgd1.shape)
      print(al_fgd2, al_fgd2.shape)

      if self.whichfgd == "fgd1":
        al_fgd = al_fgd1
      elif self.whichfgd == "fgd2":
        al_fgd = al_fgd2
      print(al_fgd, al_fgd.shape)

      ### apply fgd 17 cuts
      ### =================
      which_selection_condition = np.where(al_fgd > 17)
      wsc = which_selection_condition
      print(wsc)

      ### apply momentum cut
      ### ==================
      emom = all_syst__selelec_mom[wsc, i][0]
      nuErecQE = all_syst__selelec_nuErecQE[wsc, i][0]
      print(emom, emom.shape)
      print(nuErecQE, nuErecQE.shape)

      emom_cond = np.where(emom >= 200.0)
      nuErecQE = nuErecQE[emom_cond]
      print(nuErecQE, nuErecQE.shape)

      ### create histogram of reconstructed events
      ### ========================================
      nuErecQE_hist = np.histogram(nuErecQE, bins=syst_binning)[0]
      nuErecQE_hist_normed = nuErecQE_hist / MC2DATA_RATIO
      print(nuErecQE_hist, np.sum(nuErecQE_hist))
      print(nuErecQE_hist_normed, np.sum(nuErecQE_hist_normed))

      ### compute covariance matrix
      ### =========================
      pre_Vi = (nuErecQE_hist_normed - nominal_mc_as_data) / nominal_mc_as_data
      print(pre_Vi)
      Vi =  np.outer(pre_Vi, pre_Vi)
      print(Vi)

      # V += Vi

      # Vcov = np.cov(pre_Vi)
      # print(Vcov)
      # Vcorr = np.corrcoef(pre_Vi)
      # print(Vcorr)
      # print(np.stack((Vcov, Vcov), axis=0))

      # cov_mat = np.cov(Vcov, rowvar=False)
      # print("cov_mat", cov_mat)
      # corr_mat = np.corrcoef(Vcov, rowvar=False)
      # print("corr_mat", corr_mat)

      Vcov2 = compute_cov_matr(nominal_mc_as_data, nuErecQE_hist_normed)
      print(Vcov2)
      V += Vcov2

      N_evts_toys[i] = np.sum(nuErecQE_hist_normed)

      # f0_nuisance_parameters_nominal_value += nuErecQE_hist_normed
      # f0_nuisance_parameters_nominal_value += (nuErecQE_hist_normed - nominal_mc_as_data) / nominal_mc_as_data

    V /= N_toys
    print(np.sqrt(np.abs(V)))
    print(V.diagonal())
    print("V", V)

    f0_nuisance_parameters_nominal_value = np.sqrt(V.diagonal())
    print("f0_nuisance_parameters_nominal_value", f0_nuisance_parameters_nominal_value)

    # f0_nuisance_parameters_nominal_value /= N_toys
    # print("f0_nuisance_parameters_nominal_value", f0_nuisance_parameters_nominal_value)

    VV = np.sqrt(np.abs(V))

    plt.imshow(V[::-1,:], extent = [0,4,0,4])
    # plt.imshow(VV[::-1,:], extent = [0,4,0,4])
    # # plt.imshow((VV/np.max(VV))[::-1,:])
    plt.colorbar()
    plt.show()

    # plt.hist(N_evts_toys, bins=100, histtype="step", color="red", linewidth=2)
    # plt.axvline(np.sum(nominal_mc_as_data), color="black")
    # plt.show()

    plt.bar([0,1,2,3], np.sqrt(V.diagonal()))
    plt.show()

    ### compute cholesky matrix
    ### =======================
    cholesky_matrix_upperdiag = cholesky(V, lower=True).T
    # print(cholesky_matrix)
    # print(cholesky_matrix.T)
    # print(cholesky_matrix @ cholesky_matrix.T)

    # random_variables = np.random.normal(loc=0, scale=1, size=(4, len(V)))
    # print(random_variables)

    random_variables = np.random.normal(loc=0, scale=1, size=(4))
    print(random_variables)

    print(random_variables @ cholesky_matrix_upperdiag)

    f_extr = f0_nuisance_parameters_nominal_value + random_variables @ cholesky_matrix_upperdiag
    print("f_extr", f_extr)

    Vinv = np.linalg.inv(V)
    print("Vinv", Vinv)


    N_toy_exps = 1000
    f_extr_arr = np.zeros((N_toy_exps, 4))
    pull = np.zeros((N_toy_exps, 4))
    delta = np.zeros(N_toy_exps)
    for k in range(N_toy_exps):
      random_variables = np.random.normal(loc=0, scale=1, size=(4))
      f_extr = f0_nuisance_parameters_nominal_value + random_variables @ cholesky_matrix_upperdiag
      # if np.any(f_extr < 0.0): continue
      # if np.all(f_extr > 0.0):
      f_extr_arr[k] = f_extr
      pull[k] = (f_extr - f0_nuisance_parameters_nominal_value) / np.sqrt(f0_nuisance_parameters_nominal_value)

      delta[k] = (f_extr - f0_nuisance_parameters_nominal_value).T @ Vinv @ (f_extr - f0_nuisance_parameters_nominal_value)
      print(delta[k], (f_extr - f0_nuisance_parameters_nominal_value),
            (f_extr - f0_nuisance_parameters_nominal_value).T @ Vinv,
            (f_extr - f0_nuisance_parameters_nominal_value).T @ Vinv @ (f_extr - f0_nuisance_parameters_nominal_value))

    # plt.hist(f_extr_arr[:, 0], bins=100)
    # plt.axvline(f0_nuisance_parameters_nominal_value[0], color="black")
    # plt.show()
    # plt.hist(f_extr_arr[:, 1], bins=100)
    # plt.axvline(f0_nuisance_parameters_nominal_value[1], color="black")
    # plt.show()
    # plt.hist(f_extr_arr[:, 2], bins=100)
    # plt.axvline(f0_nuisance_parameters_nominal_value[2], color="black")
    # plt.show()
    # plt.hist(f_extr_arr[:, 3], bins=100)
    # plt.axvline(f0_nuisance_parameters_nominal_value[3], color="black")
    # plt.show()

    # plt.hist(pull[:, 0], bins=100)
    # plt.show()
    # plt.hist(pull[:, 1], bins=100)
    # plt.show()
    # plt.hist(pull[:, 2], bins=100)
    # plt.show()
    # plt.hist(pull[:, 3], bins=100)
    # plt.show()

    plt.hist(delta, bins=100)
    plt.title("$(f-f_0)^T V^{-1} (f-f_0)$", fontsize=18)
    plt.xlabel("$(f-f_0)^T V^{-1} (f-f_0)$", fontsize=16)
    plt.show()


    N_syst = 4
    delta = np.zeros(
      (
        self.model.dm2.shape[0],
        self.model.sin22theta.shape[0],
        N_toy_exps
      )
    )

    gauss_variated_hist = np.zeros(
      (
        self.hist_reco_nuErecQE_survived.shape[0],
        self.hist_reco_nuErecQE_survived.shape[1],
        self.hist_reco_nuErecQE_survived.shape[2],
        N_toy_exps
      )
    )

    likelihood_H1 = np.zeros(
      (
        self.hist_reco_nuErecQE_survived.shape[0],
        self.hist_reco_nuErecQE_survived.shape[1],
        N_toy_exps
      )
    )

    tm = Timer()
    tm.start("histograms variation")
    variate_histograms_with_gauss(gauss_variated_hist, self.hist_reco_nuErecQE_survived, N_toy_exps)
    tm.stop()

    gauss_variated_hist[gauss_variated_hist < 0.0] = 0.0

    gauss_variated_hist_norm = gauss_variated_hist / MC2DATA_RATIO

    tm = Timer()
    tm.start("systematics histograms variation")
    variate_systematics_histograms(
      f0_nuisance_parameters_nominal_value, cholesky_matrix_upperdiag, Vinv,
      N_toy_exps, N_syst, self.model.dm2_steps, self.model.sin22t_steps,
      delta)
    tm.stop()

    _nominal_mc_as_data = np.histogram(self.reco_nuErecQE, bins=self.binning*1.0e-3)[0] / MC2DATA_RATIO
    tm.start("computation of Lkh in H1")
    compute_likelihoods_wsyst(
      self.model.dm2_steps,
      self.model.sin22t_steps,
      N_toy_exps,
      None, ## likelihood in null hypothesis, shape = (N_experiments)
      _nominal_mc_as_data, ## shape = (N_bins), normilized to data
      gauss_variated_hist_norm, ## shape = (dm2_steps, sin22t_steps, N_bins, N_experiments), normilized to data
      likelihood_H1, ## likelihood in alternative hypothesis, shape = (dm2_steps, sin22t_steps, N_experiments)
      None, ## shape = (dm2_steps, sin22t_steps, N_experiments)
      delta
    )
    tm.stop()

    # fig_lkh_h1_spec4 = Visualizer.draw_althyplkh_wrt_eventsdistr(
    #   _dm2=7.0,
    #   _sin22t=0.2,
    #   dm2_model_arr=self.model.dm2,
    #   sin22theta_model_arr=self.model.sin22theta,
    #   likelihood_Halt=likelihood_H1,
    #   reco_nuErecQE_GeV=self.reco_nuErecQE,
    #   reco_nuErecQE_survived_GeV=self.reco_nuErecQE_survived,
    #   binning_GeV=self.binning*1.0e-3
    # )
    # plt.show()

    # plt.hist(likelihood_H1[10, 10], bins=100)
    # plt.show()

    extent = [self.model.sin22t_min, self.model.sin22t_max, self.model.dm2_min, self.model.dm2_max]
    aspect = (self.model.sin22t_max - self.model.sin22t_min) / (self.model.dm2_max - self.model.dm2_min)
    plt.imshow(np.mean(likelihood_H1, axis=2)[::-1,:], extent=extent, aspect=aspect)
    plt.xlabel(R"$sin^2 2\theta$", fontsize=16)
    plt.ylabel(R"$\Delta m^2, eV^2$", fontsize=16)
    plt.title("likelihood with systematic term mean for alternative hypothesis", fontsize=20)
    plt.tight_layout()
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
  m = OscillationModel3p1(
    dm2_min=0.0,
    # dm2_max=100.0,
    # dm2_max=10.0,
    dm2_max=40.0,
    dm2_steps=10,
    sin22t_min=0.0,
    sin22t_max=1.0,
    sin22t_steps=20,
    resolution_multiplier=4,
  )

  ft = Fitter(
    path_with_preprocessed_files="/t2k/users/shvartsman/HEP-soft/T2KSoft/myscripts/fitting/preprocessed",
    mode="FHC",
    whichfgd="fgd1",
    use_syst=False,
    use_realdata=False,
    binning=None,
    model=m,
    # docversion="v2.0",
    # docversion="v3.0",
    # docversion="v3.0a",
    # docversion="v3.0b",
    # docversion="v3.0c",
    # docversion="v3.0d",
    # docversion="v4.0",
    # docversion="v5.0",
    # docversion="v5.0-fdm2_10-fsin22t_0.5",
    # docversion="v5.0-fdm2_10-fsin22t_1.0",
    # docversion="v5.0-fdm2_0-fsin22t_0.0",
    # docversion="v6.0",
    # docversion="v6.0-fdm2_10-fsin22t_0.5",
    # docversion="v6.0-fdm2_10-fsin22t_1.0",
    # docversion="v6.0-fdm2_0-fsin22t_0.0",
    # docversion="v6.0-fdm2_10-fsin22t_0.75",
    # docversion="v6.0-fdm2_10-fsin22t_0.25",
    # docversion="v6.0-fdm2_10-fsin22t_0.625",
    docversion="v7.0",
    # docversion="v7.0",
    resolution_multiplier=1
  )

  ft.oscillate_over_parameter_space()
  # ft.fit_v2()
  # ft.fit_v3(N_toy_experiments=1000, outfile="diplots.pdf")
  # ft.fit_v3(N_toy_experiments=1000, outfile="diplots-2.pdf")
  # ft.fit_v3(N_toy_experiments=1000, outfile="diplots-3.pdf")
  # ft.fit_v3(N_toy_experiments=3000, outfile="diplots-4.pdf")
  # ft.fit_v3(N_toy_experiments=3000, outfile="diplots-5.pdf")
  # ft.fit_v3(N_toy_experiments=1000, outfile="distributions.pdf")

  ft.fit_v4(N_toy_experiments=3000, outfile="diplots-5.pdf")

  # ft.process_systematics()
