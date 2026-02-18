"""
Core module for free fermion Hamiltonians.

This module provides the FermionicHamiltonian class for quadratic fermionic
Hamiltonians of the form:

H = sum_{ij} A_{ij} (c†_i c_j - c_i c†_j) + sum_{ij} B_{ij} c†_i c†_j + B*_{ij} c_i c_j

The Bogoliubov-de Gennes formalism is used for diagonalization.
"""

import numpy as np
from typing import Optional, Tuple


class FermionicHamiltonian:
    """
    General quadratic fermionic Hamiltonian.

    The Hamiltonian is parametrized by two matrices:
    - A: Hermitian matrix (hopping terms)
    - B: Anti-symmetric matrix (pairing terms)

    Attributes
    ----------
    L : int
        System size (number of sites).
    A : np.ndarray
        Hopping matrix (Hermitian).
    B : np.ndarray
        Pairing matrix (anti-symmetric).
    H : np.ndarray
        Full BdG Hamiltonian matrix (2L x 2L).
    eigs : np.ndarray
        Single-particle energies (L positive eigenvalues).
    U, V : np.ndarray
        Bogoliubov transformation matrices.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, auto_diagonalize: bool = True):
        """
        Initialize the fermionic Hamiltonian.

        Parameters
        ----------
        A : np.ndarray
            Hopping matrix. Must be Hermitian.
        B : np.ndarray
            Pairing matrix. Must be anti-symmetric.
        auto_diagonalize : bool
            If True, diagonalize immediately after construction.
        """
        self._validate_matrices(A, B)

        L = A.shape[0]
        self.L = L
        self.A = A.copy()
        self.B = B.copy()

        # Build the BdG Hamiltonian
        self.H = np.zeros((2*L, 2*L), dtype=complex)
        self.H[:L, :L] = A
        self.H[:L, L:] = B
        self.H[L:, :L] = -B.conj()
        self.H[L:, L:] = -A.conj()

        # Swap matrix for particle-hole symmetry operations
        self._Swap = np.zeros((2*L, 2*L))
        self._Swap[L:, :L] = np.eye(L)
        self._Swap[:L, L:] = np.eye(L)

        # Initialize storage for diagonalization results
        self.eigs = None
        self.eigs_complete = None
        self.U = None
        self.V = None
        self.W = None

        if auto_diagonalize:
            self.diagonalize()

    @staticmethod
    def _validate_matrices(A: np.ndarray, B: np.ndarray):
        """Validate that A is Hermitian and B is anti-symmetric."""
        if not np.allclose(A, A.conj().T):
            raise ValueError("A must be Hermitian (A = A†)")
        if not np.allclose(B, -B.T):
            raise ValueError("B must be anti-symmetric (B = -B^T)")
        if A.shape != B.shape:
            raise ValueError(f"A and B must have same shape, got {A.shape} and {B.shape}")

    def diagonalize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Diagonalize the BdG Hamiltonian.

        The BdG Hamiltonian has particle-hole symmetry, so eigenvalues come
        in pairs ±ε. We extract the positive eigenvalues and the corresponding
        Bogoliubov transformation matrices U and V.

        Returns
        -------
        eigs : np.ndarray
            Single-particle energies (positive eigenvalues).
        U : np.ndarray
            Particle component of Bogoliubov transformation.
        V : np.ndarray
            Hole component of Bogoliubov transformation.
        """
        L = self.L

        # Diagonalize the full BdG Hamiltonian
        eig, W = np.linalg.eigh(self.H)
        self.eigs_complete = eig.copy()
        self.W_complete = W.copy()

        # Roll to get positive eigenvalues in first L positions
        # (eigh returns sorted eigenvalues, so negative ones come first)
        eig = np.roll(eig, L)
        W = np.roll(W, L, axis=1)

        # Handle degenerate zero-energy states (critical point physics)
        if np.isclose(eig[0], eig[-1], atol=1e-10):
            W = self._handle_zero_modes(eig, W)

        # Extract Bogoliubov matrices
        self.eigs = eig[:L]
        self.U = W[:L, :L].copy()
        self.V = W[L:, :L].copy()

        # Store combined transformation matrix
        self.W = np.zeros((2*L, L), dtype=complex)
        self.W[:L, :] = self.U
        self.W[L:, :] = self.V

        return self.eigs, self.U, self.V

    def _handle_zero_modes(self, eig: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Handle degenerate zero-energy eigenstates.

        At critical points (e.g., h=J in Ising), there can be zero-energy modes
        that need special treatment to ensure proper fermionic anticommutation.

        Parameters
        ----------
        eig : np.ndarray
            Eigenvalues (already rolled).
        W : np.ndarray
            Eigenvectors (already rolled).

        Returns
        -------
        W : np.ndarray
            Modified eigenvector matrix.
        """
        L = self.L
        state1 = W[:, 0]
        state2 = W[:, -1]

        # Construct proper linear combinations satisfying particle-hole
        a = np.dot(self._Swap, state1 + state2) + (state1 + state2)
        b = np.dot(self._Swap, state1 - state2) - (state1 - state2)

        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)

        if a_norm < 1e-10 or b_norm < 1e-10:
            return W  # Cannot construct proper eigenstates

        a = (a / a_norm)[:L]
        b = (b / b_norm)[:L]

        w = np.concatenate([a + b, a - b]) / np.sqrt(2)
        W[:, 0] = w
        W[:, -1] = self._Swap @ w

        # Verify the construction
        res_1 = np.linalg.norm(self.H @ w - eig[0] * w)
        res_2 = np.linalg.norm(self.H @ (self._Swap @ w) - eig[-1] * (self._Swap @ w))

        if res_1 > 1e-10 or res_2 > 1e-10:
            import warnings
            warnings.warn(
                f"Zero-mode redefinition may be inaccurate. "
                f"Residuals: {res_1:.2e}, {res_2:.2e}"
            )

        return W

    def ground_state_energy(self) -> float:
        """
        Compute the ground state energy.

        For the BdG Hamiltonian, the ground state energy is:
        E_0 = -sum_k |epsilon_k|

        where epsilon_k are the positive single-particle energies.

        Returns
        -------
        E0 : float
            Ground state energy.
        """
        if self.eigs is None:
            self.diagonalize()
        return -np.sum(np.abs(self.eigs))

    def copy(self) -> 'FermionicHamiltonian':
        """Create a deep copy of the Hamiltonian."""
        return FermionicHamiltonian(self.A.copy(), self.B.copy())
