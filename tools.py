import numpy as np
from scipy.sparse.linalg import expm_multiply

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
        self.H = np.zeros((2*L, 2*L))
        
        self.H[:L, :L] =  A.copy()
        self.H[:L, L:] =  B.copy()
        self.H[L:, :L] = -B.conj().copy()
        self.H[L:, L:] = -A.conj().copy()
        
        self.A = A.copy()
        self.B = B.copy()
        
        self.Swap = np.zeros((2*L,2*L))
        self.Swap[L:,:L] = np.eye(L)
        self.Swap[:L,L:] = np.eye(L)
        
        self.diagonalize()
        
    def diagonalize(self):
      eig, W = np.linalg.eigh(self.H)
      L = self.L
          
      eig = np.roll(eig, L)
      W   = np.roll(W,L, axis=1)
    
      if(np.isclose(eig[0],eig[-1])):
       state1 = W[:,0]
       state2 = W[:,-1]
       
       a = np.dot(self.Swap, state1+state2) +  state1+state2
       b = np.dot(self.Swap, state1-state2) - (state1-state2)
    
       a = (a/np.linalg.norm(a))[:L]
       b = (b/np.linalg.norm(b))[:L]
    
       w = np.concatenate([a+b,a-b])/np.sqrt(2)
       W[:,0]  = w
       W[:,-1] = self.Swap@w
       res_1   = np.linalg.norm(self.H@w - eig[0]*w)
       res_2   = np.linalg.norm(self.H@self.Swap@w - eig[-1]*self.Swap@w)
       if (res_1 > 10**-11 or res_2 > 10**-11):
          print("WARNING: there are problems in the redefinition of the zero energy eigenstates")
      self.eigs, self.U, self.V = eig[:L],  W[:L,:L], W[L:,:L]

class state:
    def __init__(self, H: FermionicHamiltonian = None):
        if H is not None:
            self.initialize_from_hamiltonian(H)
            
    def initialize_from_hamiltonian(self, FH: FermionicHamiltonian):
        FH.diagonalize()
        self.U = FH.U.copy()
        self.V = FH.V.copy()
        self.L = FH.L
        self.W = np.zeros((2*self.L,self.L),complex)
        self.setWfromUV()
        
    def setWfromUV(self):
        self.W[:self.L,:] = self.U
        self.W[self.L:,:] = self.V
    def setUVfromW(self):
        self.U = self.W[:self.L,:]
        self.V = self.W[self.L:,:]
        
    def time_step(self, dt, FH):
        self.W = expm_multiply(-1j*2*FH.H*dt, self.W)
        
    def correlation_function(self):
        self.setUVfromW()
        G = self.U@self.U.conj().T
        F = self.U@self.V.conj().T
        M = np.eye(self.L)-2*(G+F)
        
        
        