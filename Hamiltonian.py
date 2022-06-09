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

def KitaevHamiltonian(L,h,alpha,J=1):
    if isinstance(J, (float,int)): J = np.ones(L)*J
    if isinstance(h, (float,int)): h = np.ones(L)*h
    V = np.zeros((L,L))
    V[0,1:] = np.minimum(np.arange(L)[1:],L-np.arange(L)[1:])**(-float(alpha))
    for i in range(1,L):
        V[i,:] = np.roll(V[(i-1),:],shift=1)
    Nalpha = np.sum(V[0,:(L//2)])
    V = J*V/Nalpha
    A = np.diag(h, k=0)-V/2.
    B = -np.triu(V/2)+np.triu(V/2).T
    return FermionicHamiltonian(A, B)

def QuantumDysonHamiltonian(N,sigma,h,J=1):
    L = 2**N
    if isinstance(h, (float,int)): h = np.ones(L)*h
    # Build coupling matrix
    real_states = [np.vectorize(np.binary_repr)(np.arange(2**N),N)][0]
    t = np.array([2.**( - (1 + sigma)*k ) for k in np.arange(0, N) ])
    V = np.zeros((L,L))
    for i, state_a in enumerate(real_states):
        for j, state_b in enumerate(real_states):
            if i != j :
                k = N
                while( state_a[:k] != state_b[:k] or k < 0 ):
                    k = k-1
                else:
                    V[i,j] = t[(N-k-1)]
    
    V = -J*V
    A = np.diag(h, k=0) - V/2
    B = -np.triu(V/2)+np.triu(V/2).T
    return FermionicHamiltonian(A, B)

    