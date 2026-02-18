"""
Time evolution for free fermion systems.

This module provides tools for quantum quench dynamics in free fermion systems.
The time evolution of correlation functions can be computed exactly using the
Bogoliubov transformation.

For a sudden quench from H0 to H1:
|psi(t)> = exp(-i H1 t) |psi_0>

where |psi_0> is the ground state of H0.
"""

import numpy as np
from typing import Optional, Union, List
from tqdm import tqdm
from .core import FermionicHamiltonian
from .correlators import CorrelationFunctions


class SuddenQuench:
    """
    Handle sudden quench dynamics from H0 to H1.

    The system starts in the ground state of H0 and evolves under H1.
    The time-evolved correlation functions are computed exactly.

    Attributes
    ----------
    H0 : FermionicHamiltonian
        Initial Hamiltonian.
    H1 : FermionicHamiltonian
        Final (post-quench) Hamiltonian.
    L : int
        System size.
    correlations : CorrelationFunctions
        Current correlation functions.
    current_time : float
        Current evolution time.
    """

    def __init__(self, H0: FermionicHamiltonian, H1: FermionicHamiltonian):
        """
        Initialize the quench dynamics.

        Parameters
        ----------
        H0 : FermionicHamiltonian
            Initial Hamiltonian (defines initial state).
        H1 : FermionicHamiltonian
            Final Hamiltonian (governs time evolution).
        """
        # Ensure both Hamiltonians are diagonalized
        if H0.U is None:
            H0.diagonalize()
        if H1.U is None:
            H1.diagonalize()

        assert H0.L == H1.L, "Hamiltonians must have the same system size"

        self.H0 = H0
        self.H1 = H1
        self.L = H0.L

        # Store the complete eigendata from H1 for time evolution
        self._eigs1 = H1.eigs_complete  # Full spectrum (2L eigenvalues)
        self._W1 = H1.W_complete  # Full eigenvector matrix

        # Initial Bogoliubov amplitudes (ground state of H0)
        self._w0 = np.zeros((2*self.L, self.L), dtype=complex)
        self._w0[:self.L, :] = H0.U
        self._w0[self.L:, :] = H0.V

        # Initialize correlation functions
        self.correlations = CorrelationFunctions()
        self.correlations.L = self.L

        # Set to t=0 state
        self.current_time = 0.0
        self._evolve_to(0.0)

    def set_post_quench_hamiltonian(self, H1: FermionicHamiltonian):
        """
        Change the post-quench Hamiltonian.

        Useful for studying response to different final Hamiltonians
        starting from the same initial state.

        Parameters
        ----------
        H1 : FermionicHamiltonian
            New final Hamiltonian.
        """
        if H1.U is None:
            H1.diagonalize()

        assert H1.L == self.L, "New Hamiltonian must have same system size"

        self.H1 = H1
        self._eigs1 = H1.eigs_complete
        self._W1 = H1.W_complete

    def evolve(self, t: float) -> CorrelationFunctions:
        """
        Evolve to time t and return correlation functions.

        Parameters
        ----------
        t : float
            Evolution time.

        Returns
        -------
        correlations : CorrelationFunctions
            Correlation functions at time t.
        """
        self._evolve_to(t)
        return self.correlations

    def _evolve_to(self, t: float):
        """
        Internal method to perform time evolution.

        The time evolution of the Bogoliubov amplitudes is:
        w(t) = W1 @ exp(-2i * diag(E1) * t) @ W1† @ w(0)

        where W1 are the eigenvectors of H1 and E1 are the eigenvalues.
        """
        self.current_time = t

        # Time evolution: |w(t)> = U(t) |w(0)> where U = W @ e^{-iEt} @ W†
        # The factor of 2 comes from the BdG doubling
        phases = np.exp(-1j * 2.0 * self._eigs1 * t)
        wt = self._W1 @ (phases[:, None] * (self._W1.conj().T @ self._w0))

        # Extract U(t) and V(t)
        Ut = wt[:self.L, :]
        Vt = wt[self.L:, :]

        # Update correlation functions
        self.correlations.U = Ut
        self.correlations.V = Vt
        self.correlations._compute_fermionic_correlators()

    def energy(self, H: Optional[FermionicHamiltonian] = None) -> float:
        """
        Compute energy expectation value at current time.

        Parameters
        ----------
        H : FermionicHamiltonian, optional
            Hamiltonian to compute energy with. Default is H1.

        Returns
        -------
        E : float
            Energy expectation value.
        """
        if H is None:
            H = self.H1
        return self.correlations.energy(H)

    def time_series(
        self,
        times: np.ndarray,
        observables: Optional[List[str]] = None,
        progress: bool = True
    ) -> dict:
        """
        Compute time series of observables.

        Parameters
        ----------
        times : np.ndarray
            Array of times to compute observables at.
        observables : list of str, optional
            List of observables to compute. Options:
            - 'energy': Energy <H1>
            - 'occupation': Site occupation <n_i>
            - 'entropy': Entanglement entropy profile
            - 'mz': Magnetization <sigma^z_i>
            - 'xx': Longitudinal correlation matrix <sigma^x_i sigma^x_j> (shape: n_times x L x L)
            Default is ['energy', 'occupation'].
        progress : bool, optional
            Whether to show a progress bar. Default is True.

        Returns
        -------
        results : dict
            Dictionary with observable names as keys and time series as values.
            Each value has shape (len(times), ...) where ... depends on the observable.
        """
        if observables is None:
            observables = ['energy', 'occupation']

        results = {obs: [] for obs in observables}
        results['times'] = times

        for t in tqdm(times, desc="time evolution", disable=not progress):
            self._evolve_to(t)

            if 'energy' in observables:
                results['energy'].append(self.energy())

            if 'occupation' in observables:
                results['occupation'].append(np.diag(self.correlations.G).real.copy())

            if 'entropy' in observables:
                results['entropy'].append(self.correlations.entanglement_entropy_profile())

            if 'mz' in observables:
                results['mz'].append(self.correlations.magnetization_z())

            if 'xx' in observables:
                results['xx'].append(self.correlations.xx_correlator_matrix())

        # Convert lists to arrays
        for key in results:
            if key != 'times':
                results[key] = np.array(results[key])

        return results


class AdiabaticEvolution:
    """
    Adiabatic (slow) quench dynamics.

    The Hamiltonian changes slowly from H(0) to H(T) over time T.
    This can be used to prepare ground states or study adiabatic protocols.
    """

    def __init__(
        self,
        H_initial: FermionicHamiltonian,
        H_final: FermionicHamiltonian,
        total_time: float,
        schedule: str = 'linear'
    ):
        """
        Initialize adiabatic evolution.

        Parameters
        ----------
        H_initial : FermionicHamiltonian
            Initial Hamiltonian at t=0.
        H_final : FermionicHamiltonian
            Final Hamiltonian at t=T.
        total_time : float
            Total evolution time T.
        schedule : str
            Interpolation schedule: 'linear' or 'cosine'.
        """
        assert H_initial.L == H_final.L

        self.H_initial = H_initial
        self.H_final = H_final
        self.total_time = total_time
        self.schedule = schedule
        self.L = H_initial.L

        # Initialize from ground state of H_initial
        if H_initial.U is None:
            H_initial.diagonalize()

        self._w = np.zeros((2*self.L, self.L), dtype=complex)
        self._w[:self.L, :] = H_initial.U
        self._w[self.L:, :] = H_initial.V

        self.current_time = 0.0
        self.correlations = CorrelationFunctions()
        self.correlations.L = self.L
        self._update_correlations()

    def _interpolation_param(self, t: float) -> float:
        """Get interpolation parameter s(t) in [0, 1]."""
        s = t / self.total_time
        s = np.clip(s, 0, 1)

        if self.schedule == 'linear':
            return s
        elif self.schedule == 'cosine':
            return (1 - np.cos(np.pi * s)) / 2
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def _hamiltonian_at(self, t: float) -> FermionicHamiltonian:
        """Get interpolated Hamiltonian at time t."""
        s = self._interpolation_param(t)
        A = (1 - s) * self.H_initial.A + s * self.H_final.A
        B = (1 - s) * self.H_initial.B + s * self.H_final.B
        return FermionicHamiltonian(A, B)

    def _update_correlations(self):
        """Update correlation functions from current state."""
        self.correlations.U = self._w[:self.L, :]
        self.correlations.V = self._w[self.L:, :]
        self.correlations._compute_fermionic_correlators()

    def evolve(self, dt: float, num_steps: int = 1) -> CorrelationFunctions:
        """
        Evolve the system by time dt using Trotter decomposition.

        Parameters
        ----------
        dt : float
            Time step.
        num_steps : int
            Number of steps to take.

        Returns
        -------
        correlations : CorrelationFunctions
            Correlation functions after evolution.
        """
        for _ in range(num_steps):
            # Get Hamiltonian at midpoint
            t_mid = self.current_time + dt / 2
            H_mid = self._hamiltonian_at(t_mid)

            # Evolve by dt under H(t_mid)
            phases = np.exp(-1j * 2.0 * H_mid.eigs_complete * dt)
            self._w = H_mid.W_complete @ (
                phases[:, None] * (H_mid.W_complete.conj().T @ self._w)
            )

            self.current_time += dt

        self._update_correlations()
        return self.correlations
