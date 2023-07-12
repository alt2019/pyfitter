### https://chat.openai.com/chat/7b13409f-f3ed-4b88-b920-d8fc1ab4dc21
#
# https://chat.openai.com/chat/77546387-ec3f-49ce-b69b-888f6f4501b8?__cf_chl_tk=7lEo80VWrtv9A4lLcUGk.XfcwA87jlHhP7biOr8P6Os-1678261034-0-gaNycGzNG9A

import logging

# logging.basicConfig(level = logging.INFO)
# logger = logging.getLogger(__name__)

from custom_logger import CustomLogger
# logging.setLoggerClass(CustomLogger)
# logging.basicConfig(level = logging.INFO)
# logger = logging.getLogger("pyfitter-v1.py")
# logger.propagate = False
# logger.setLevel(logging.INFO)


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


from utils import compute_histograms, variate_histograms_with_gauss, compute_likelihoods_v2, compute_likelihoods_with_gamma_v2
from timer import Timer
from visualizer import Visualizer

logging.setLoggerClass(CustomLogger)
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger("pyfitter-v1.py")
logger.propagate = False
logger.setLevel(logging.INFO)
logger.propagate = False


N_PLOTS = None # for gradio

MC2DATA_RATIO = 16.4536 # needed to be extracted from root file




def compute_CL(cl=0.95, ndof=2):
  from scipy.stats import chi2

  # n = 2 # number of degrees of freedom
  # CL = 0.95 #confidence level

  crit_val = chi2.ppf(cl, ndof)

  # print(crit_val)
  return crit_val





class Fitter:
  def __init__(self,
               path_with_preprocessed_files,
               gamma_path=None,
               mode="FHC",
               whichfgd="fgd1",
               use_syst=False,
               use_realdata=False,
               use_gamma=False,
               binning=None,
               gamma_binning=None,
               model=None,
               docversion="v01"):
    logger.info('initializing Fitter...')
    self.mode = mode
    self.whichfgd = whichfgd
    self.use_realdata = use_realdata
    self.use_gamma = use_gamma
    self.path = path_with_preprocessed_files
    self.gamma_path = gamma_path

    self.model = model

    self.mc_truth_vars = None
    self.mc_default_vars = None
    self.data_default_vars = None
    self.gamma_mc_default_vars = None

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

    self.hist_reco_nuErecQE_survived = None
    self.hist_reco_nuErecQE_survived_normed2data = None

    self.reco_nuErecQE_survived_gamma = None
    self.hist_reco_nuErecQE_survived_gamma = None
    self.hist_reco_nuErecQE_survived_normed2data_gamma = None

    self.binning_MeV = None
    self.binning_GeV = None
    self.gamma_binning_MeV = None
    self.gamma_binning_GeV = None

    if binning:
      self.binning_MeV = binning
      self.binning_GeV = binning * 1.0e-3
      self.gamma_binning_MeV = gamma_binning
      self.gamma_binning_GeV = gamma_binning * 1.0e-3
    else:
      self.set_default_binning()

    self.load_files()
    self.set_needed_variables()

    self.pdf = mplpdf.PdfPages(f"fitting-v1-{mode}-{whichfgd}-{docversion}.pdf")

    self.visualizer = None


  def load_files(self):
    logger.info('Loading files...')
    # data-FHC-default_tree-fgd1.npz
    # data-FHC-default_tree-fgd2.npz
    # mc-FHC-allsyst_tree-fgd12.npz
    # mc-FHC-default_tree-fgd1.npz
    # mc-FHC-default_tree-fgd2.npz
    # mc-FHC-truth_tree-fgd1.npz
    # mc-FHC-truth_tree-fgd2.npz

    # with np.load(f"{self.path}/mc-{self.mode}-truth_tree-{self.whichfgd}.npz", allow_pickle=True, encoding="latin1") as mc_truth_file:
    #   self.mc_truth_vars = mc_truth_file["preprocessed_dict"][()]
    #   # print(self.mc_truth_vars, type(self.mc_truth_vars))
    #   for k in self.mc_truth_vars:
    #     print(k)

    with np.load(f"{self.path}/mc-{self.mode}-default_tree-{self.whichfgd}.npz", allow_pickle=True, encoding="latin1") as mc_default_file:
      self.mc_default_vars = mc_default_file["preprocessed_dict"][()]
      # print(self.mc_default_vars, type(self.mc_default_vars))
      for k in self.mc_default_vars:
        print(k)

    if self.use_gamma:
      with np.load(f"{self.gamma_path}/gamma-mc-{self.mode}-default_tree-{self.whichfgd}.npz", allow_pickle=True, encoding="latin1") as gamma_mc_default_file:
        self.gamma_mc_default_vars = gamma_mc_default_file["preprocessed_dict"][()]
        # print(self.mc_default_vars, type(self.mc_default_vars))
        for k in self.gamma_mc_default_vars:
          print(k)

    if self.use_realdata:
      with np.load(f"{self.path}/data-{self.mode}-default_tree-{self.whichfgd}.npz", allow_pickle=True, encoding="latin1") as data_default_file:
        self.data_default_vars = data_default_file["preprocessed_dict"][()]
        # print(self.data_default_vars, type(self.data_default_vars))
        for k in self.data_default_vars:
          print(k)

  def set_needed_variables(self):
    logger.info('Setting needed variables...')
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

    self.reco_selelec_mom_gamma = [ self.gamma_mc_default_vars[key] for key in self.gamma_mc_default_vars if "selelec_mom" in key ][0]
    self.reco_nuErecQE_gamma = [ self.gamma_mc_default_vars[key] for key in self.gamma_mc_default_vars if "nuErecQE" in key ][0]

    self.true_selelec_pdg_gamma = [ self.gamma_mc_default_vars[key] for key in self.gamma_mc_default_vars if "selelec_true_pdg" in key ][0]
    self.true_selelec_ppdg_gamma = [ self.gamma_mc_default_vars[key] for key in self.gamma_mc_default_vars if "selelec_true_ppdg" in key ][0]

    self.true_nu_parent_decay_point_x_gamma = [ self.gamma_mc_default_vars[key] for key in self.gamma_mc_default_vars if "true_nu_parent_decay_point_x" in key ][0]
    self.true_nu_parent_decay_point_y_gamma = [ self.gamma_mc_default_vars[key] for key in self.gamma_mc_default_vars if "true_nu_parent_decay_point_y" in key ][0]
    self.true_nu_parent_decay_point_z_gamma = [ self.gamma_mc_default_vars[key] for key in self.gamma_mc_default_vars if "true_nu_parent_decay_point_z" in key ][0]
    self.true_vtx_pos_x_gamma = [ self.gamma_mc_default_vars[key] for key in self.gamma_mc_default_vars if "true_vtx_pos_x" in key ][0]
    self.true_vtx_pos_y_gamma = [ self.gamma_mc_default_vars[key] for key in self.gamma_mc_default_vars if "true_vtx_pos_y" in key ][0]
    self.true_vtx_pos_z_gamma = [ self.gamma_mc_default_vars[key] for key in self.gamma_mc_default_vars if "true_vtx_pos_z" in key ][0]
    self.true_nu_pdg_gamma = [ self.gamma_mc_default_vars[key] for key in self.gamma_mc_default_vars if "true_nu_pdg" in key ][0]
    self.true_nu_ene_gamma = [ self.gamma_mc_default_vars[key] for key in self.gamma_mc_default_vars if "true_nu_ene" in key ][0]


  def set_default_binning(self):
    logger.info('Setting binning...')
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

    # // γ
    # const int nBins = 22;
    # // double pbins[nBins] = {0,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1700,1900,2200,2500,2800,4000,10000};
    # double pbins[nBins] = {0,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1700,1900,2200,2500,2800,4000,4200};

    gamma_binning = [200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1700,1900,2200,2500,2800,4000]
    gamma_binning = np.array(gamma_binning)

    #============================
    self.binning_MeV = binning
    self.binning_GeV = self.binning_MeV * 1.0e-3
    #============================
    self.gamma_binning_MeV = gamma_binning
    self.gamma_binning_GeV = self.gamma_binning_MeV * 1.0e-3


  def oscillate_over_parameter_space(self):
    logger.info('Computing histograms of survived events...')
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
    hist_reco_nuErecQE_survived = np.zeros((reco_nuErecQE_survived.shape[0], reco_nuErecQE_survived.shape[1], len(self.binning_GeV)-1))
    compute_histograms(
      hist_reco_nuErecQE_survived,
      reco_nuErecQE_survived,
      self.binning_GeV,
      self.model.dm2_steps,
      self.model.sin22t_steps)
    tm.stop()

    self.hist_reco_nuErecQE_survived = hist_reco_nuErecQE_survived
    self.hist_reco_nuErecQE_survived_normed2data = self.hist_reco_nuErecQE_survived / MC2DATA_RATIO

  def oscillate_over_parameter_space_gamma(self):
    logger.info('Computing histograms of survived events (gamma)...')
    data = {
      "true_nu_parent_decay_point_x": self.true_nu_parent_decay_point_x_gamma,
      "true_nu_parent_decay_point_y": self.true_nu_parent_decay_point_y_gamma,
      "true_nu_parent_decay_point_z": self.true_nu_parent_decay_point_z_gamma,
      "true_vtx_pos_x": self.true_vtx_pos_x_gamma,
      "true_vtx_pos_y": self.true_vtx_pos_y_gamma,
      "true_vtx_pos_z": self.true_vtx_pos_z_gamma,
      "true_nu_ene": self.true_nu_ene_gamma,
      "true_nu_pdg": self.true_nu_pdg_gamma,
      "reco_nuErecQE": self.reco_nuErecQE_gamma,
    }

    reco_nuErecQE_survived, survival_prob = self.model.compute_oscillations(data, isgamma=True)

    self.reco_nuErecQE_survived_gamma = reco_nuErecQE_survived

    tm = Timer()
    tm.start("histograms computing")
    hist_reco_nuErecQE_survived = np.zeros((reco_nuErecQE_survived.shape[0], reco_nuErecQE_survived.shape[1], len(self.gamma_binning_GeV)-1))
    compute_histograms(
      hist_reco_nuErecQE_survived,
      reco_nuErecQE_survived,
      self.gamma_binning_GeV,
      self.model.dm2_steps,
      self.model.sin22t_steps)
    tm.stop()

    self.hist_reco_nuErecQE_survived_gamma = hist_reco_nuErecQE_survived
    self.hist_reco_nuErecQE_survived_normed2data_gamma = self.hist_reco_nuErecQE_survived_gamma / MC2DATA_RATIO


  @staticmethod
  def compute_lkh_verbose_3(initial_params, t_E_nu_GeV, t_L_nu_km, t_PDG_nu, r_E_nu_GeV, obs_evts_hist, binning_GeV, scale2data):
      dm2, sin22t = initial_params
      # t_E_nu_GeV, t_L_nu_km, t_PDG_nu, r_E_nu_GeV, obs_evts_hist, binning_GeV, scale2data = fixed_params

      root = 1.267 * dm2 * t_L_nu_km / t_E_nu_GeV
      sinroot = np.sin(root)
      P_surv = 1.0 - sin22t * sinroot * sinroot

      P_surv[t_PDG_nu != 12] = 1.0

      rnd = np.random.uniform(0.0, 1.0, P_surv.shape[0])
      surv_evts_mask = np.ones(P_surv.shape[0])

      surv_evts_mask[rnd > P_surv] = 0

      surv_evts = r_E_nu_GeV[surv_evts_mask == 1]

      surv_evts_hist = np.histogram(surv_evts, bins=binning_GeV)[0]

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


  def fit(self, N_toy_experiments, outfile=None):
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

    obs_evts_hist = np.histogram(self.reco_nuErecQE, bins=self.binning_GeV)[0]


    # t_E_nu_GeV, t_L_nu_km, t_PDG_nu, r_E_nu_GeV, obs_evts_hist, binning_GeV, scale2data = (
    #   true_E_nu_GeV, true_L_nu_km, true_pdg_nu, self.reco_nuErecQE, obs_evts_hist, self.binning_GeV, MC2DATA_RATIO
    # )

    # fixed_params = (t_E_nu_GeV, t_L_nu_km, t_PDG_nu, r_E_nu_GeV, obs_evts_hist, binning, scale2data)


    _dm2 = 7.0
    _sin22t = 0.5
    # i_dm2 = np.where(self.model.dm2 == _dm2)[0]#[0]
    # j_sin22t = np.where(self.model.sin22theta == _sin22t)#[0]#[0]
    i_dm2 = np.nonzero(np.isclose([_dm2], self.model.dm2))[0][0]
    j_sin22t = np.nonzero(np.isclose([_sin22t], self.model.sin22theta))[0][0]
    # print(self.model.dm2)
    # print(self.model.sin22theta)
    # print(np.isclose([_sin22t], self.model.sin22theta))
    # print(np.nonzero(np.isclose([_sin22t], self.model.sin22theta))[0][0])
    print(i_dm2, j_sin22t)
    obs_evts_hist = self.hist_reco_nuErecQE_survived_normed2data[i_dm2, j_sin22t]
    print(obs_evts_hist)


    lkh = self.compute_lkh_verbose_3(
      (10.0, 0.7),
      true_E_nu_GeV, true_L_nu_km, true_pdg_nu,
      self.reco_nuErecQE, obs_evts_hist,
      self.binning_GeV, MC2DATA_RATIO
    )
    print(lkh)


    x0 = 10.0
    y0 = 0.7
    objective_function = self.compute_lkh_verbose_3
    result = minimize(objective_function, [x0, y0],
      args=(true_E_nu_GeV, true_L_nu_km, true_pdg_nu,
            self.reco_nuErecQE, obs_evts_hist,
            self.binning_GeV, MC2DATA_RATIO),
      # method='Nelder-Mead',
      method="L-BFGS-B",
      bounds=[(0, 40), (0, 1)],
      tol=1.0e-5,
      options={"maxiter": 500, "disp": True})

    # vals = list(result.values())
    # print(result.x)
    # print(result)

    print(result)
    print(lkh, [x0, y0])
    print(result.fun, result.x)
    # print([vals[0], vals[1]])
    # print(result.values())
    # print(result.errors)
    # print(result.fval)
    # print(result.nfit)
    # print(result.parameters)

    tm.stop()


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


    logging.info("GAMMA: setting 'gauss_variated_hist_gamma' array of shape (N_dm^2_steps, N_sin^2(2t)_steps, N_bins, N_toy_experiments)")
    gauss_variated_hist_gamma = np.zeros(
      (
        self.hist_reco_nuErecQE_survived_gamma.shape[0],
        self.hist_reco_nuErecQE_survived_gamma.shape[1],
        self.hist_reco_nuErecQE_survived_gamma.shape[2],
        N_toy_experiments
      )
    )

    logging.info("GAMMA: computing 'gauss_variated_hist_gamma' array")
    tm = Timer()
    tm.start("GAMMA: histograms variation")
    variate_histograms_with_gauss(gauss_variated_hist_gamma, self.hist_reco_nuErecQE_survived_gamma, N_toy_experiments)
    tm.stop()
    gauss_variated_hist_gamma[gauss_variated_hist_gamma < 0.0] = 0.0
    gauss_variated_hist_norm_gamma = gauss_variated_hist_gamma / MC2DATA_RATIO


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

        nominal_mc_as_data_hist_normed_gamma = self.hist_reco_nuErecQE_survived_normed2data_gamma[i_dm2, j_sin22t]
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
        # compute_likelihoods_v2(
        #   self.model.dm2_steps,
        #   self.model.sin22t_steps,
        #   N_toy_experiments,
        #   nominal_mc_as_data_hist_normed, ## shape = (N_bins), normilized to data
        #   gauss_variated_hist_norm, ## shape = (dm2_steps, sin22t_steps, N_bins, N_toy_experiments), normilized to data
        #   likelihood_H1, ## likelihood in alternative hypothesis, shape = (dm2_steps, sin22t_steps, N_toy_experiments)
        # )
        compute_likelihoods_with_gamma_v2(
          self.model.dm2_steps,
          self.model.sin22t_steps,
          N_toy_experiments,
          nominal_mc_as_data_hist_normed, ## shape = (N_bins), normilized to data
          gauss_variated_hist_norm, ## shape = (dm2_steps, sin22t_steps, N_bins, N_toy_experiments), normilized to data
          nominal_mc_as_data_hist_normed_gamma,
          gauss_variated_hist_norm_gamma,
          likelihood_H1, ## likelihood in alternative hypothesis, shape = (dm2_steps, sin22t_steps, N_toy_experiments)
        )
        tm.stop()


        '''
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
              binning_GeV=self.binning_GeV
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
        #'''


        lkh_mean = np.mean(likelihood_H1, axis=2)
        # print(np.min(lkh_mean))
        # cl95 = compute_CL(0.95, 2)
        # # lkh_mean.isclose()
        # idxs = np.where(np.abs(lkh_mean - cl95) < 0.5)
        # print(idxs)



        title_label = "likelihood mean for alternative hypothesis with "fR"$\Delta m^2 = {_dm2_fixed} eV^2$, $sin^2 2\theta = {_sin22t_fixed}$"
        extent = [self.model.sin22t_min, self.model.sin22t_max, self.model.dm2_min, self.model.dm2_max]
        aspect = (self.model.sin22t_max - self.model.sin22t_min) / (self.model.dm2_max - self.model.dm2_min)
        fig_lkh_alt_mean, ax_lkh_alt_mean = plt.subplots(1, 1, figsize=(20, 15))
        m = ax_lkh_alt_mean.imshow(np.mean(likelihood_H1, axis=2)[::-1, :], extent=extent, aspect=aspect, cmap="jet")
        # m = ax_lkh_alt_mean.imshow(np.mean(likelihood_H1, axis=2)[::-1, :], cmap="jet")

        #ax_lkh_alt_mean.contour(self.model.sin22theta, self.model.dm2, lkh_mean, [compute_CL(0.95, 2)], colors='red', linewidths=2);
        ax_lkh_alt_mean.contour(self.model.sin22theta, self.model.dm2, lkh_mean, [compute_CL(0.95, 2)], colors='white', linewidths=2);
        #i_dm2 = np.where(self.model.dm2 == 7.3)[0][0]
        #j_sin22t = np.where(self.model.sin22theta == 0.38)[0][0]
        #ax_lkh_alt_mean.scatter(i_dm2, j_sin22t, s=20, marker='*', color='gold', zorder=3)
        ax_lkh_alt_mean.scatter(0.38, 7.3, s=120, marker='*', color='gold', zorder=3)

        # ax_lkh_alt_mean.contour(self.model.sin22theta, self.model.dm2, lkh_mean, [compute_CL(0.9, 2)], colors='green', linewidths=2);
        # ax_lkh_alt_mean.contour(self.model.sin22theta, self.model.dm2, lkh_mean, [compute_CL(0.99976, 2)], colors='yellow', linewidths=2);
        # m = ax_lkh_alt_mean.imshow(np.log(np.mean(likelihood_H1, axis=2)[::-1, :]), extent=extent, aspect=aspect, cmap="jet")
        # m = ax_lkh_alt_mean.imshow(np.log(np.min(likelihood_H1, axis=2)[::-1, :]), extent=extent, aspect=aspect, cmap="jet")
        # ax_lkh_alt_mean.scatter(idxs[1],idxs[0][::-1],color='r')
        ax_lkh_alt_mean.set_xlabel(R"$sin^2 2\theta$", fontsize=20)
        ax_lkh_alt_mean.set_ylabel(R"$\Delta m^2, eV^2$", fontsize=20)
        ax_lkh_alt_mean.set_title(title_label, fontsize=24)
        ax_lkh_alt_mean.tick_params(axis='both', which='major', labelsize=15)
        ax_lkh_alt_mean.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax_lkh_alt_mean.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        cb = fig_lkh_alt_mean.colorbar(m, ax=ax_lkh_alt_mean)
        cb.ax.tick_params(labelsize=15)
        fig_lkh_alt_mean.tight_layout()

        # fig_lkh_alt_mean_2, ax_lkh_alt_mean_2 = plt.subplots(1, 1, figsize=(20, 15))
        # m = ax_lkh_alt_mean_2.imshow(np.mean(likelihood_H1, axis=2)[::-1, :], extent=extent, aspect=aspect)
        # # m = ax_lkh_alt_mean_2.imshow(np.log(np.mean(likelihood_H1, axis=2)[::-1, :]), extent=extent, aspect=aspect)
        # # m = ax_lkh_alt_mean_2.imshow(np.log(np.min(likelihood_H1, axis=2)[::-1, :]), extent=extent, aspect=aspect)
        # ax_lkh_alt_mean_2.set_xlabel(R"$sin^2 2\theta$", fontsize=20)
        # ax_lkh_alt_mean_2.set_ylabel(R"$\Delta m^2, eV^2$", fontsize=20)
        # ax_lkh_alt_mean_2.set_title(title_label, fontsize=24)
        # ax_lkh_alt_mean_2.tick_params(axis='both', which='major', labelsize=15)
        # ax_lkh_alt_mean_2.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        # ax_lkh_alt_mean_2.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        # cb = fig_lkh_alt_mean_2.colorbar(m, ax=ax_lkh_alt_mean_2)
        # cb.ax.tick_params(labelsize=15)
        # fig_lkh_alt_mean_2.tight_layout()

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




if __name__ == '__main__':
  m = OscillationModel3p1(
    dm2_min=0.0,
    # dm2_max=100.0,
    dm2_max=10.0,
    # dm2_max=40.0,
    dm2_steps=10,
    sin22t_min=0.0,
    sin22t_max=1.0,
    sin22t_steps=20,
    resolution_multiplier=4,
  )

  ft = Fitter(
    path_with_preprocessed_files="/t2k/users/shvartsman/HEP-soft/T2KSoft/myscripts/fitting/preprocessed",
    gamma_path="/t2k/users/shvartsman/HEP-soft/T2KSoft/myscripts/fitting/preprocessed-gamma",
    mode="FHC",
    whichfgd="fgd1",
    use_syst=False,
    use_realdata=False,
    use_gamma=True,
    binning=None,
    model=m,
    docversion="v0.0",
  )

  ft.oscillate_over_parameter_space()
  ft.oscillate_over_parameter_space_gamma()

  # ft.fit(N_toy_experiments=1000, outfile="diplots-fit_v1-0.pdf")
  # ft.fit_v3(N_toy_experiments=1000, outfile="diplots-fit_v1-0.pdf")
  # ft.fit_v3(N_toy_experiments=1000, outfile="diplots-fit_v1-1.pdf")
  # ft.fit_v3(N_toy_experiments=1000, outfile="diplots-fit_v1-1a.pdf")
  # ft.fit_v3(N_toy_experiments=1000, outfile="diplots-fit_v1-2.pdf")
  # ft.fit_v3(N_toy_experiments=1000, outfile="diplots-fit_v1-3.pdf")
  # ft.fit_v3(N_toy_experiments=1000, outfile="diplots-fit_v1-4.pdf")
  # ft.fit_v3(N_toy_experiments=1000, outfile="diplots-fit_v1-5.pdf")
  # ft.fit_v3(N_toy_experiments=1000, outfile="diplots-fit_v1-6.pdf")

  #ft.fit_v3(N_toy_experiments=1000, outfile="diplots-fit_v1-7.pdf")
  #ft.fit_v3(N_toy_experiments=1000, outfile="diplots-fit_v1-7a.pdf")

  #ft.fit_v3(N_toy_experiments=1000, outfile="diplots-fit_v1-8.pdf")
  #ft.fit_v3(N_toy_experiments=1000, outfile="diplots-fit_v1-8a.pdf")
  ft.fit_v3(N_toy_experiments=1000, outfile="diplots-fit_v1-8b.pdf")
