from .tools import FermionicHamiltonian
from .tools import correlation_functions
import numpy as np
from scipy.sparse.linalg import expm


class sudden_quench:
    def __init__(self,H0: FermionicHamiltonian,H1: FermionicHamiltonian):
        H0.diagonalize()
        H1.diagonalize()
        self.H0, self.H1 = H0, H1
        self.eigs1, self.W1 = H1.eigs_complete, H1.W
        self.corr = correlation_functions(H0)
        self.w0 = self.H0.w.copy()
        self.time_evolve(0.)
        self.L = self.H0.L
        
    def time_evolve(self,t: float):
        self.wt = self.W1 @ expm(-1j*2.*np.diag(self.eigs1)*t) @ self.W1.T.conj() @ self.w0
        self.corr.set_W(self.wt)
    
    def energy(self, FH=None):
        self.corr.setUVfromW()
        if FH is None:
            return self.corr.energy(self.H1)
        else:
            self.corr_temp = correlation_functions()
            _, Wnew = np.linalg.eigh(FH.H)
            self.corr_temp.set_W(Wnew@Wnew.T@self.wt)
            self.corr_temp.L = self.L
            self.corr_temp.setUVfromW()
            self.corr_temp.set_correlation_functions()
            return self.corr_temp.energy(FH)
        
    def set_correlation_functions(self):
        self.corr.setUVfromW()
        self.corr.set_correlation_functions()