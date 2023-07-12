import logging

import numpy as np
from numba import jit, njit, prange
from neutrino_oscillations import cpy_compute_Lnu

# logging.basicConfig(level = logging.INFO)
# logger = logging.getLogger(__name__)


from custom_logger import CustomLogger

logging.setLoggerClass(CustomLogger)
logger = logging.getLogger("model.py")
logger.propagate = False
logger.setLevel(logging.DEBUG)



@njit(parallel=True)
def survival_probability_3p1nu(Enu_GeV, Lnu_km, sin22t, dm2):
  """
    Function Name: survival_probability_3p1nu

    Description:
      This function takes in four numpy arrays as input and returns a numpy array as output. The function calculates the survival probability of a neutrino given i

    Parameters:
      Enu_GeV (numpy array): A 3D array containing neutrino energies in GeV.
      Lnu_km (numpy array): A 3D array containing neutrino path lengths in km.
      sin22t (numpy array): A 3D array containing sin^2(2theta) values.
      dm2 (numpy array): A 3D array containing delta m^2 values.

    Returns:
      P (numpy array): A 3D numpy array containing the survival probability of the neutrino.

    Libraries Used:
      numpy
      numba

    Assumptions:
      The input arrays are 3D numpy arrays of the same shape.
      The values in sin22t and dm2 are valid and can be used to calculate the survival probability.
  """
  root = 1.267 * dm2 * Lnu_km / Enu_GeV
  sinroot = np.sin(root)
  P = 1.0 - sin22t * sinroot * sinroot
  return P

@njit(parallel=True)
def oscillate(
    true_L_nu_km,
    true_E_nu_GeV,
    true_pdg_nu,
    dm2,
    sin22theta,
    survival_prob_arr,
    survived_mask_arr,
    N ## length of arrays
  ):
  survival_prob_arr = survival_probability_3p1nu(true_E_nu_GeV, true_L_nu_km, sin22theta, dm2)
  survival_prob_arr[true_pdg_nu != 12] = 1.0
  rnd = np.random.uniform(0.0, 1.0, N)
  survived_mask_arr[rnd > survival_prob_arr] = 0

@njit(parallel=True)
def bulk_oscillate(
    true_L_nu_km,
    true_E_nu_GeV,
    true_pdg_nu,
    dm2_steps,
    sin2theta_steps,
    dm2_arr,
    sin22theta_arr,
    survival_prob_matr,
    survived_mask_matr,
    N, ## length of arrays
  ):
  # for i in range(dm2_steps):
  for i in prange(dm2_steps):
    dm2 = dm2_arr[i]
    for j in range(sin2theta_steps):
      sin22t = sin22theta_arr[j]
      survival_prob_arr = np.zeros(N)
      survived_mask_arr = np.ones(N)
      oscillate(
        true_L_nu_km=true_L_nu_km,
        true_E_nu_GeV=true_E_nu_GeV,
        true_pdg_nu=true_pdg_nu,
        dm2=dm2,
        sin22theta=sin22t,
        survival_prob_arr=survival_prob_arr,
        survived_mask_arr=survived_mask_arr,
        N=N
      )
      survival_prob_matr[i, j] = survival_prob_arr
      survived_mask_matr[i, j] = survived_mask_arr


@jit(parallel=True)
def compute_oscillations(
    true_E_nu_GeV_msh,
    true_L_nu_km_msh,
    sin22t_msh,
    dm2_msh,
    true_pdg_nu,
    rnd,
    reco_nuErecQE_msh
  ):
  """ (sphinx style; ChatGPT)
    This function uses Numba's @jit decorator to optimize the performance of the function. It takes in 7 parameters and returns a tuple of two 3D numpy arrays.

    Parameters:
      true_E_nu_GeV_msh (numpy.ndarray): A 3D array containing true neutrino energies in GeV.
      true_L_nu_km_msh (numpy.ndarray): A 3D array containing true neutrino path lengths in km.
      sin22t_msh (numpy.ndarray): A 3D array containing sin^2(2theta) values.
      dm2_msh (numpy.ndarray): A 3D array containing delta m^2 values.
      true_pdg_nu (float): A float variable containing the true neutrino flavor in PDG code.
      rnd (numpy.ndarray): A 3D array containing random numbers between 0 and 1.
      reco_nuErecQE_msh (numpy.ndarray): A 3D array containing reconstructed neutrino energies in GeV.
    Returns:
      A tuple of two numpy arrays:
      reco_nuErecQE_msh (numpy.ndarray): A modified version of the input reco_nuErecQE_msh array with values set to -1.0 for the elements where the corresponding e
      survival_prob_arr (numpy.ndarray): A modified version of the survival probability array with values set to -1.0 for the elements where the corresponding elem
  """
  """ (doxygen style; ChatGPT)
    @brief Computes the oscillations and modifies the input arrays based on the survival mask array.
    @param true_E_nu_GeV_msh A 3D numpy array containing true neutrino energies in GeV.
    @param true_L_nu_km_msh A 3D numpy array containing true neutrino path lengths in km.
    @param sin22t_msh A 3D numpy array containing sin^2(2theta) values.
    @param dm2_msh A 3D numpy array containing delta m^2 values.
    @param true_pdg_nu A float variable containing the true neutrino flavor in PDG code.
    @param rnd A 3D numpy array containing random numbers between 0 and 1.
    @param reco_nuErecQE_msh A 3D numpy array containing reconstructed neutrino energies in GeV.
    @return A tuple containing two 3D numpy arrays:
    - reco_nuErecQE_msh: A modified version of the input reco_nuErecQE_msh array with values set to -1.0 for the elements where the corresponding element in the su
    - survival_prob_arr: A modified version of the survival probability array with values set to -1.0 for the elements where the corresponding element in the survi
  """
  survival_prob_arr = survival_probability_3p1nu(true_E_nu_GeV_msh, true_L_nu_km_msh, sin22t_msh, dm2_msh)
  survived_mask_arr = np.ones(survival_prob_arr.shape)
  survival_prob_arr[:, :, true_pdg_nu != 12] = 1.0
  survived_mask_arr[rnd > survival_prob_arr] = 0

  reco_nuErecQE_msh[survived_mask_arr == 0] = -1.0
  survival_prob_arr[survived_mask_arr == 0] = -1.0

  return reco_nuErecQE_msh, survival_prob_arr




class OscillationModel:
  def __init__(self):
    pass
    logger.info('initializing OscillationModel...')


class OscillationModel3p1(OscillationModel):
  def __init__(self,
               dm2_min:float=0.0,
               dm2_max:float=10.0,
               dm2_steps:int=10,
               sin22t_min:float=0.0,
               sin22t_max:float=1.0,
               sin22t_steps:int=20,
               resolution_multiplier:int=10):
    logger.info('initializing OscillationModel3p1...')

    self.dm2_min = dm2_min
    self.dm2_max = dm2_max
    self.dm2_steps = dm2_steps * resolution_multiplier + 1
    self.sin22t_min = sin22t_min
    self.sin22t_max = sin22t_max
    self.sin22t_steps = sin22t_steps * resolution_multiplier + 1

    self.dm2_ranges = None
    self.sin22theta_ranges = None
    self.dm2 = None
    self.d_dm2 = None
    self.sin22theta = None
    self.d_sin22theta = None

    # self.survival_probability = None
    self.survival_probability_gamma = None
    self.survived_events_gamma_arr = None
    self.survival_probability = None
    self.survived_events_arr = None

    self.render_parameter_space()


  def render_parameter_space(self):
    self.dm2_ranges = (self.dm2_min, self.dm2_max, self.dm2_steps)
    self.sin22theta_ranges = (self.sin22t_min, self.sin22t_max, self.sin22t_steps)

    self.dm2, self.d_dm2 = np.linspace(*self.dm2_ranges, retstep=True, endpoint=True)
    self.sin22theta, self.d_sin22theta = np.linspace(*self.sin22theta_ranges, retstep=True, endpoint=True)
    # print(self.dm2, self.sin22theta)
    logger.info(f'Dm^2 space: {self.dm2}')
    logger.info(f'sin^2(2theta) space: {self.sin22theta}')
    # print()

  def get(self, dm2_idx, sin22t_idx):
    dm2 = self.dm2[dm2_idx]
    sin22t = self.sin22theta[sin22t_idx]
    return dm2, sin22t

  def get_parameter_space_iterator(self):
    parameter_space_iterator = []
    for i in range(self.dm2_steps):
      dm2 = self.dm2[i]
      for j in range(self.sin2theta_steps):
        sin22t = self.sin22theta[j]
        parameter_space_iterator.append([i, j, dm2, sin22t])
    parameter_space_iterator = np.array(parameter_space_iterator)
    return parameter_space_iterator

  def compute_oscillations(self, data:dict, isgamma=False):
    logger.info('computing neutrino flight path...')
    true_L_nu_mm = cpy_compute_Lnu(
      data["true_nu_parent_decay_point_x"],
      data["true_nu_parent_decay_point_y"],
      data["true_nu_parent_decay_point_z"],
      data["true_vtx_pos_x"],
      data["true_vtx_pos_y"],
      data["true_vtx_pos_z"]
    )

    true_L_nu_km = true_L_nu_mm * 1.0e-6

    true_E_nu_GeV = data["true_nu_ene"] * 1.0e-3
    true_pdg_nu = data["true_nu_pdg"]

    N = true_E_nu_GeV.shape[0]

    reco_nuErecQE = data["reco_nuErecQE"]
    reco_nuErecQE *= 1.0e-3 # GeV

    dm2_msh, sin22t_msh, true_E_nu_GeV_msh = np.meshgrid(self.dm2, self.sin22theta, true_E_nu_GeV, indexing="ij")
    _, _, true_L_nu_km_msh = np.meshgrid(self.dm2, self.sin22theta, true_L_nu_km, indexing="ij")
    _, _, reco_nuErecQE_msh = np.meshgrid(self.dm2, self.sin22theta, reco_nuErecQE, indexing="ij")
    print(dm2_msh.shape, sin22t_msh.shape, true_E_nu_GeV_msh.shape, true_L_nu_km_msh.shape)
    # print(dm2_msh, sin22t_msh, true_E_nu_GeV_msh, true_L_nu_km_msh)

    logger.info('computing oscillations...')
    rnd = np.random.uniform(0.0, 1.0, true_E_nu_GeV_msh.shape)
    survived_events_arr, survival_prob_arr = compute_oscillations(
      true_E_nu_GeV_msh,
      true_L_nu_km_msh,
      sin22t_msh,
      dm2_msh,
      true_pdg_nu,
      rnd,
      reco_nuErecQE_msh
    )

    # print("survived_events_arr[0, 0]", survived_events_arr[0, 0], survived_events_arr[0, 0][survived_events_arr[0, 0]!=-1].shape)
    # print("survived_events_arr[20, 50]", survived_events_arr[20, 50], survived_events_arr[20, 50][survived_events_arr[20, 50]!=-1].shape)

    if isgamma:
      self.survival_probability_gamma = survival_prob_arr
      self.survived_events_gamma_arr = survived_events_arr
    else:
      self.survival_probability = survival_prob_arr
      self.survived_events_arr = survived_events_arr

    return survived_events_arr, survival_prob_arr

  def save(self, topath:str):
    pass

  def load(self, frompath:str):
    pass
