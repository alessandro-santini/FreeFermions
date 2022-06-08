from .tools import FermionicHamiltonian
from .tools import correlation_functions
import numpy as np
import copy


class sudden_quench:
    def __init__(self,H0: FermionicHamiltonian,H1: FermionicHamiltonian):
        H0.diagonalize()
        H1.diagonalize()
        self.H0, self.H1 = H0, H1
        self.W0 = copy.deepcopy(H0.W)
        self.eigs1, self.W1 = H1.eigs_complete, H1.W
        self.corr = correlation_functions()
        
    def time_evolve(self,t: float):
        self.Wt = self.W1 @ np.diag(np.exp(-1j*2*(self.eigs1)*t)) @ self.W1.T.conj() @ self.W0
        self.corr.set_W(self.Wt)
        
    def energy(self, H1=None):
        self.corr.setUVfromW()
        if H1 is None:
            return self.corr.energy(self.H1)
        
    