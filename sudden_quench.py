from .tools import FermionicHamiltonian
from .tools import state

class sudden_quench:
    def __init__(self,H0: FermionicHamiltonian,H1: FermionicHamiltonian):
        H0.diagonalize()
        H1.diagonalize()
        psi = state(H0)
        self.H0, self.H1, self.psi = H0, H1, psi
        
    def time_evolve(self,t):
        W = self.H1.W.T.conj()@np.exp(-1j*)
        