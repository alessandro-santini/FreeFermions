"""
Correlation functions for free fermion systems.

This module computes various correlation functions from the Bogoliubov
transformation matrices, including:
- Fermionic correlators: G = <c†c>, F = <cc>
- Spin correlators: <σ^x σ^x>, <σ^z σ^z>
- Entanglement entropy

The XX correlator is computed using the Pfaffian method, which properly
accounts for the Jordan-Wigner string.
"""

import numpy as np
from scipy.special import xlogy
from pfapack.pfaffian import pfaffian
from typing import Optional, Tuple, Union
from .core import FermionicHamiltonian


class CorrelationFunctions:
    """
    Compute and store correlation functions for free fermion ground states.

    The correlation functions are parametrized by two matrices:
    - G_{ij} = <c†_i c_j> : Normal Green's function
    - F_{ij} = <c_i c_j>  : Anomalous (pairing) correlator

    For the ground state of a quadratic Hamiltonian:
    G = V* V^T
    F = V* U^T

    Attributes
    ----------
    L : int
        System size.
    G : np.ndarray
        Normal correlation matrix <c†_i c_j>.
    F : np.ndarray
        Anomalous correlation matrix <c_i c_j>.
    """

    def __init__(self, hamiltonian: Optional[FermionicHamiltonian] = None):
        """
        Initialize correlation functions.

        Parameters
        ----------
        hamiltonian : FermionicHamiltonian, optional
            If provided, compute correlations from ground state.
        """
        self.L = None
        self.U = None
        self.V = None
        self.G = None
        self.F = None
        self._Gamma = None  # Covariance matrix for Pfaffian calculations

        if hamiltonian is not None:
            self.from_hamiltonian(hamiltonian)

    def from_hamiltonian(self, H: FermionicHamiltonian):
        """
        Initialize from a diagonalized Hamiltonian (ground state).

        Parameters
        ----------
        H : FermionicHamiltonian
            Diagonalized Hamiltonian.
        """
        if H.U is None:
            H.diagonalize()

        self.L = H.L
        self.U = H.U.copy()
        self.V = H.V.copy()
        self._compute_fermionic_correlators()

    def from_UV(self, U: np.ndarray, V: np.ndarray):
        """
        Initialize from Bogoliubov transformation matrices.

        Parameters
        ----------
        U : np.ndarray
            Particle component (L x L).
        V : np.ndarray
            Hole component (L x L).
        """
        self.L = U.shape[0]
        self.U = U.copy()
        self.V = V.copy()
        self._compute_fermionic_correlators()

    def _compute_fermionic_correlators(self):
        """
        Compute G and F from U and V matrices.

        For the ground state (vacuum of Bogoliubov quasiparticles):
        G_{ij} = <c†_i c_j> = (V* V^T)_{ij}
        F_{ij} = <c_i c_j> = (V* U^T)_{ij}
        """
        self.G = self.V.conj() @ self.V.T
        self.F = self.V.conj() @ self.U.T
        self._Gamma = None  # Reset covariance matrix

    def _build_majorana_correlator_matrix(self):
        """
        Build the full Majorana two-point correlator matrix.

        For Majorana fermions a_j = c_j + c†_j, b_j = i(c†_j - c_j), we compute
        the matrix M where M[m,n] = <γ_m γ_n> with interleaved ordering:
        γ_{2j} = a_j, γ_{2j+1} = b_j

        This matrix is used for computing XX correlators via Pfaffian.
        The off-diagonal part is antisymmetric (M[m,n] = -M[n,m] for m≠n)
        while M[m,m] = 1 (Majorana normalization).

        From the fermion correlators G = <c†c> and F = <cc>:
        <a_i a_j> = G_ij - G_ji + 2i*Im(F_ij) + δ_ij
        <b_i b_j> = G_ij - G_ji - 2i*Im(F_ij) + δ_ij
        <a_i b_j> = i[δ_ij - G_ij - G_ji - 2*Re(F_ij)]
        <b_i a_j> = -<a_i b_j>
        """
        if self._Gamma is not None:
            return

        L = self.L
        # Matrix is 2L × 2L with ordering (a_0, b_0, a_1, b_1, ..., a_{L-1}, b_{L-1})
        self._Gamma = np.zeros((2*L, 2*L), dtype=complex)

        for ii in range(L):
            for jj in range(L):
                # Compute the four correlators from G = <c†c> and F = <cc>
                # Note: The free fermion code uses F = V* U^T which has opposite sign
                # from the physical <cc> correlator due to BdG convention differences.
                # We negate F here to match the physical convention.
                Gij = self.G[ii, jj]
                Gji = self.G[jj, ii]
                Fij = -self.F[ii, jj]  # Sign flip to match physical convention

                # <a_i a_j> = <(c_i + c†_i)(c_j + c†_j)>
                #   = F_ij + (δ_ij - G_ji) + G_ij + (-F*_ij)
                #   = G_ij - G_ji + (F_ij - F*_ij) + δ_ij
                #   = G_ij - G_ji + 2i*Im(F_ij) + δ_ij
                aa = Gij - Gji + 2j * np.imag(Fij)
                if ii == jj:
                    aa += 1

                # <b_i b_j> = G_ij - G_ji - 2i*Im(F_ij) + δ_ij
                bb = Gij - Gji - 2j * np.imag(Fij)
                if ii == jj:
                    bb += 1

                # <a_i b_j> = i[δ_ij - G_ij - G_ji - 2*Re(F_ij)]
                ab = 1j * (-Gij - Gji - 2 * np.real(Fij))
                if ii == jj:
                    ab += 1j

                # <b_i a_j> = i[G_ij + G_ji - 2*Re(F_ij) - δ_ij]
                ba = 1j * (Gij + Gji - 2 * np.real(Fij))
                if ii == jj:
                    ba -= 1j

                # Store with interleaved indexing: γ_{2k} = a_k, γ_{2k+1} = b_k
                self._Gamma[2*ii, 2*jj] = aa      # <a_i a_j>
                self._Gamma[2*ii+1, 2*jj+1] = bb  # <b_i b_j>
                self._Gamma[2*ii, 2*jj+1] = ab    # <a_i b_j>
                self._Gamma[2*ii+1, 2*jj] = ba    # <b_i a_j>

    # ---------- ZZ Correlator ----------
    def zz_correlator(self, i: int, j: int) -> complex:
        """
        Compute <sigma^z_i sigma^z_j>.

        Using sigma^z = 1 - 2n = 1 - 2c†c and Wick's theorem:
        <sigma^z_i sigma^z_j> = (1 - 2<n_i>)(1 - 2<n_j>) + 4*(<c†_i c_j><c_i c†_j> - <c†_i c†_j><c_i c_j>)
                              = (1 - 2G_ii)(1 - 2G_jj) + 4*(|G_ij|² - |F_ij|²)   for i != j

        Parameters
        ----------
        i, j : int
            Site indices.

        Returns
        -------
        corr : complex
            The ZZ correlation function.
        """
        if i == j:
            return 1.0 + 0j

        Gii, Gjj = self.G[i, i], self.G[j, j]
        Gij = self.G[i, j]
        Fij = self.F[i, j]

        mz_i = 1 - 2*Gii  # <sigma^z_i>
        mz_j = 1 - 2*Gjj  # <sigma^z_j>

        # Connected correlator from Wick's theorem:
        # <n_i n_j> = G_ii G_jj - |G_ij|² + |F_ij|²
        # So: <σ^z_i σ^z_j> = <σ^z_i><σ^z_j> + 4(|F_ij|² - |G_ij|²)
        connected = 4*(np.abs(Fij)**2 - np.abs(Gij)**2)

        return mz_i * mz_j + connected

    def zz_correlator_matrix(self) -> np.ndarray:
        """
        Compute the full ZZ correlation matrix.

        Returns
        -------
        ZZ : np.ndarray
            Matrix where ZZ[i,j] = <sigma^z_i sigma^z_j>.
        """
        L = self.L
        ZZ = np.zeros((L, L), dtype=complex)

        mz = np.diag(1 - 2*self.G)  # <sigma^z>
        mz_outer = np.outer(mz, mz)  # <sigma^z_i><sigma^z_j>

        # Connected part: 4*(|F_ij|² - |G_ij|²)
        connected = 4*(np.abs(self.F)**2 - np.abs(self.G)**2)

        ZZ = mz_outer + connected
        np.fill_diagonal(ZZ, 1.0)  # <sigma^z_i sigma^z_i> = 1

        return np.real_if_close(ZZ)

    # ---------- XX Correlator ----------
    def xx_correlator(self, i: int, j: int) -> complex:
        """
        Compute <sigma^x_i sigma^x_j> using the Pfaffian of Majorana correlators.

        For free fermions after Jordan-Wigner transformation:
        σ^x_i σ^x_j = a_i (∏_{k=i}^{j-1} σ^z_k) a_j

        where a_k = c_k + c†_k are Majorana fermions. Using σ^z_k = -i a_k b_k,
        this becomes:
        <σ^x_i σ^x_j> = (-i)^n <b_i a_{i+1} b_{i+1} ... a_{j-1} b_{j-1} a_j>

        where n = j - i. By Wick's theorem for Majorana fermions, this is:
        <σ^x_i σ^x_j> = (-i)^n × Pf(M_sub)

        where M_sub is the 2n × 2n antisymmetric matrix of Majorana correlators
        for the operators (b_i, a_{i+1}, b_{i+1}, ..., a_{j-1}, b_{j-1}, a_j).

        Parameters
        ----------
        i, j : int
            Site indices.

        Returns
        -------
        corr : complex
            The XX correlation function.
        """
        if i > j:
            i, j = j, i

        if i == j:
            return 1.0 + 0j

        n = j - i  # Distance

        # Build the Majorana correlator matrix if not already done
        self._build_majorana_correlator_matrix()

        # Extract indices for the Majorana operators:
        # (b_i, a_{i+1}, b_{i+1}, a_{i+2}, ..., b_{j-1}, a_j)
        # With interleaved indexing γ_{2k} = a_k, γ_{2k+1} = b_k:
        # b_i = 2i+1, a_{i+1} = 2(i+1), b_{i+1} = 2(i+1)+1, ..., a_j = 2j
        indices = []
        indices.append(2*i + 1)  # b_i
        for k in range(i + 1, j):
            indices.append(2*k)      # a_k
            indices.append(2*k + 1)  # b_k
        indices.append(2*j)  # a_j

        # Extract the 2n × 2n submatrix
        M_sub = self._Gamma[np.ix_(indices, indices)]

        # Make it antisymmetric (the off-diagonal parts already are)
        # The diagonal of <γ_m γ_m> = 1, but for Pfaffian we need antisymmetric matrix
        # So we use (M - M.T) / 2 for the antisymmetric part
        M_antisym = (M_sub - M_sub.T) / 2

        # Compute Pfaffian
        pf = pfaffian(M_antisym)

        # Multiply by (-i)^n
        result = ((-1j) ** n) * pf

        return result

    def xx_correlator_matrix(self) -> np.ndarray:
        """
        Compute the full XX correlation matrix.

        This computes <sigma^x_i sigma^x_j> for all pairs i, j.
        Note: This is O(L³) due to Pfaffian computations.

        Returns
        -------
        XX : np.ndarray
            Matrix where XX[i,j] = <sigma^x_i sigma^x_j>.
        """
        L = self.L
        XX = np.zeros((L, L), dtype=complex)

        for i in range(L):
            XX[i, i] = 1.0
            for j in range(i+1, L):
                XX[i, j] = self.xx_correlator(i, j)
                XX[j, i] = XX[i, j]  # Symmetric

        return np.real_if_close(XX)

    # ---------- Local magnetizations ----------
    def magnetization_z(self) -> np.ndarray:
        """
        Compute local z-magnetization <sigma^z_i> = 1 - 2<n_i>.

        Returns
        -------
        mz : np.ndarray
            Array of local z-magnetizations.
        """
        return np.real_if_close(1 - 2*np.diag(self.G))

    def magnetization_x(self) -> np.ndarray:
        """
        Compute local x-magnetization <sigma^x_i>.

        For translationally invariant ground states, this is typically zero
        unless there is symmetry breaking.

        Returns
        -------
        mx : np.ndarray
            Array of local x-magnetizations.
        """
        # <sigma^x_i> = <c_i + c†_i> = 0 for number-conserving states
        # For BCS-type states with pairing, this can be non-zero
        # It's related to the anomalous correlator
        L = self.L
        mx = np.zeros(L, dtype=complex)

        for i in range(L):
            # <sigma^x_i> involves a Pfaffian of a 2x2 matrix at site i
            # which simplifies to a determinant
            # Actually for a single site: <sigma^x_i> = <c_i + c†_i>
            # This is typically 0 by fermion number parity unless in a coherent state
            mx[i] = 0  # Zero for standard ground states

        return np.real_if_close(mx)

    # ---------- Entanglement Entropy ----------
    def entanglement_entropy(self, subsystem_size: int) -> float:
        """
        Compute entanglement entropy for a contiguous subsystem.

        Uses the correlation matrix method: eigenvalues of the reduced
        correlation matrix give the entanglement spectrum.

        Parameters
        ----------
        subsystem_size : int
            Size of the subsystem (sites 0 to subsystem_size-1).

        Returns
        -------
        S : float
            von Neumann entanglement entropy.
        """
        l = subsystem_size
        if l <= 0 or l >= self.L:
            return 0.0

        # Build the 2l x 2l correlation matrix for the subsystem
        A = np.zeros((2*l, 2*l), dtype=complex)

        G_sub = self.G[:l, :l]
        F_sub = self.F[:l, :l]

        A[:l, :l] = -1j*(G_sub - G_sub.T + F_sub - F_sub.conj())
        A[:l, l:] = -np.eye(l) + G_sub + G_sub.T - F_sub - F_sub.conj()
        A[l:, :l] = np.eye(l) - G_sub - G_sub.T - F_sub - F_sub.conj()
        A[l:, l:] = -1j*(G_sub - G_sub.T - F_sub + F_sub.conj())

        # Eigenvalues of i*A give the entanglement spectrum
        Lambda = np.linalg.eigvalsh(1j * A)[l:]

        # Entanglement entropy
        Pq = (1 + Lambda) / 2
        # Avoid log(0) issues
        Pq = np.clip(Pq, 1e-15, 1 - 1e-15)

        return float(-np.sum(xlogy(Pq, Pq) + xlogy(1 - Pq, 1 - Pq)))

    def entanglement_entropy_profile(self) -> np.ndarray:
        """
        Compute entanglement entropy for all bipartitions.

        Returns
        -------
        S : np.ndarray
            Array of entanglement entropies for subsystem sizes 1 to L-1.
        """
        return np.array([self.entanglement_entropy(l) for l in range(1, self.L)])

    # ---------- Energy ----------
    def energy(self, H: FermionicHamiltonian) -> float:
        """
        Compute expectation value of energy for the current state.

        Parameters
        ----------
        H : FermionicHamiltonian
            Hamiltonian to compute energy with.

        Returns
        -------
        E : float
            Energy expectation value.
        """
        # E = Tr(A * (G - (I - G*))) + Tr(B * (F† - F))
        #   = Tr(A * (2G - I)) + Tr(B * (F† - F))
        A_term = np.einsum('ij,ij->', -H.A, self.G - (np.eye(self.L) - self.G.conj()))
        B_term = np.einsum('ij,ij->', H.B, self.F.conj().T - self.F)

        return float(np.real_if_close(A_term + B_term))
