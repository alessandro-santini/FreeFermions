import numpy as np
from .tools import FermionicHamiltonian

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