import logging

import numpy as np
from numba import jit, njit, prange
from neutrino_oscillations import cpy_compute_Lnu

import sympy as sp
from sympy import Symbol

from custom_logger import CustomLogger
logging.setLoggerClass(CustomLogger)
logger = logging.getLogger("model.py")
logger.propagate = False
logger.setLevel(logging.DEBUG)



class OscillationModel:
  def __init__(self):
    logger.info('initializing OscillationModel...')

    # self.sin22theta12 = Symbol(R"sin^2(\theta_{12})")
    # self.sin22theta13 = Symbol(R"sin^2(\theta_{13})")
    # self.sin22theta23 = Symbol(R"sin^2(\theta_{23})")

    # print(self.sin22theta13)
    # print(self.sin22theta12)
    # print(self.sin22theta23)
    # print(self.sin22theta23 * self.sin22theta23)

    self.sin22theta12 = Symbol(R"sin22theta12")
    self.sin22theta13 = Symbol(R"sin22theta13")
    self.sin22theta23 = Symbol(R"sin22theta23")


  def survival_probability(self, nu_a, nu_b):
    pass


if __name__ == "__main__":
  OscillationModel()
