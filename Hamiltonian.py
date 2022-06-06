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

def IsingHamiltonian(L,Jx,Jy,hz, p=0, pbc=True):
    """
    H = - J_x XX -J_y YY - h_z Z

    Parameters
    ----------
    L : int
        length of the chain.
    Jx : float, int or np.ndarray
        Jx couplings.
    Jy : float, int or np.ndarray
        Jy couplings.
    hz : float, int or np.ndarray
        on site interaction.
    p  : int
        fermion parity
    pbc: bool
        sets the boundary conditions
    Returns
    -------
    Ising  Hamiltonian.

    """
    if isinstance(Jx, (float,int)): Jx = np.ones(L)*Jx
    if isinstance(Jy, (float,int)): Jy = np.ones(L)*Jy
    if isinstance(hz, (float,int)): hz = np.ones(L)*hz
    assert Jx.size == L; assert Jy.size == L; assert hz.size == L
    if not pbc: Jx[-1],Jy[-1] = 0.,0.
        
    Jp, Jm = Jx+Jy, Jx-Jy
    
    A = np.diag(hz, k=0) - .5 * np.diag(Jp[:-1], k=1)  - .5 * np.diag(Jp[:-1], k=-1) 
    A[0, -1], A[-1, 0] = 2*[.5*(-1)**p*Jp[-1]]
    
    B = - .5 * np.diag(Jm[:-1], k=1) + .5 * np.diag(Jm[:-1], k=-1)
    B[-1, 0], B[0,-1] = .5*(-1)**p*Jm[-1], -.5*(-1)**p*Jm[-1]

    return FermionicHamiltonian(A, B)