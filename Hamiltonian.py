import numpy as np

class FermionicHamiltonian:
    def __init__(self, A: np.ndarray, B: np.ndarray):
        """
        Initialize the general fermionic quadratic hamiltonian defined as
        
        H = \sum_{ij} 2A_{ij} c^\dagger_i c_j + \sum_{ij} B_{ij} c^\dagger_i c^\dagger_j + B^*_{ij} c_i c_j
        
        Parameters
        ----------
        A : np.ndarray
            Coupling Matrix, hopping term.
        B : np.ndarray
            Coupling Matrix, pair production and pair distruction.
            
        """
        assert np.allclose(A,A.conj().T), "A has to be an Hermitean matrix"
        assert np.allclose(B,-B.T), "B has to be an anti-symmetric matrix"
        assert A.shape == B.shape, "A and B have different shapes"
        
        L = A.shape[0]
        self.L = L
        self.H = np.zeros((L, L))
        self.H[:L, :L] =  A.copy()
        self.H[:L, L:] =  B.copy()
        self.H[L:, :L] = -B.conj().copy()
        self.H[L:, L:] = -A.conj().copy()
        
        self.Swap = np.zeros((L,L))
        self.Swap[L:,:L] = np.eye(L)
        self.Swap[:L,L:] = np.eye(L)
        
        
        
        