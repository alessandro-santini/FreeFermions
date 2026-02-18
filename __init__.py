"""
FreeFermions - A library for free fermion quantum systems.

This package provides tools for studying quadratic fermionic Hamiltonians,
including ground state properties, correlation functions, entanglement,
and time evolution after quantum quenches.

Main modules:
- core: FermionicHamiltonian class for general quadratic Hamiltonians
- models: Factory functions for Ising and Kitaev long-range models
- correlators: Correlation functions (XX, ZZ, entanglement entropy)
- dynamics: Time evolution (sudden quench, adiabatic evolution)
- verification: QuTiP-based verification for small systems

Example usage:
    from FreeFermions import IsingHamiltonian, CorrelationFunctions

    # Create Ising Hamiltonian
    H = IsingHamiltonian(L=100, Jx=1.0, Jy=0.0, hz=0.5)

    # Compute correlation functions
    corr = CorrelationFunctions(H)
    xx = corr.xx_correlator_matrix()
    zz = corr.zz_correlator_matrix()
    entropy = corr.entanglement_entropy_profile()
"""

__version__ = "0.2.0"

# Core classes
from .core import FermionicHamiltonian

# Correlation functions
from .correlators import CorrelationFunctions

# Model Hamiltonians
from .models import (
    IsingHamiltonian,
    KitaevLongRange,
    both_parity_sectors,
    ground_state_energy_both_sectors,
)

# Dynamics
from .dynamics import SuddenQuench, AdiabaticEvolution

# Verification (optional, requires qutip)
from .verification import (
    verify_ising_model,
    verify_kitaev_long_range,
    run_all_verifications,
)

# Legacy imports for backwards compatibility
from .core import FermionicHamiltonian as FermionicHamiltonian
from .correlators import CorrelationFunctions as correlation_functions

__all__ = [
    # Core
    "FermionicHamiltonian",
    # Correlators
    "CorrelationFunctions",
    "correlation_functions",  # Legacy alias
    # Models
    "IsingHamiltonian",
    "KitaevLongRange",
    "both_parity_sectors",
    "ground_state_energy_both_sectors",
    # Dynamics
    "SuddenQuench",
    "AdiabaticEvolution",
    # Verification
    "verify_ising_model",
    "verify_kitaev_long_range",
    "run_all_verifications",
]
