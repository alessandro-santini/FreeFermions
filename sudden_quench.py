from .tools import FermionicHamiltonian
from .tools import state

import numpy as np
from scipy.sparse.linalg import expm_multiply, expm
import copy

class sudden_quench:
    def __init__(self,H0: FermionicHamiltonian,H1: FermionicHamiltonian):
        H0.diagonalize()
        H1.diagonalize()
        psi = state(H0)
        self.H0, self.H1, self.psi = H0, H1, psi
        self.W0 = copy.deepcopy(psi.W)
        self.eigs1,self.W1 = np.linalg.eigh(self.H1.H)
        
    def time_evolve(self,t: float):
        self.psi.W = self.W1@expm(-1j*2*np.diag(self.eigs1)*t)@self.W1.T.conj()@self.W0
        self.psi.setUVfromW()
        