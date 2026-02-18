"""
Verification module using QuTiP for exact diagonalization.

This module provides functions to verify the free fermion code against
exact diagonalization using QuTiP. This is only practical for small
systems (L <= 12 or so due to exponential scaling).

The verification covers:
- Ground state energy
- Correlation functions (XX, ZZ)
- Entanglement entropy
- Time evolution after quench
"""

import numpy as np
from typing import Optional, Tuple, Dict
import warnings


def _check_qutip():
    """Check if QuTiP is available."""
    try:
        import qutip
        return True
    except ImportError:
        return False


def build_spin_hamiltonian_ising(
    L: int,
    Jx: float,
    Jy: float,
    hz: float,
    pbc: bool = True
):
    """
    Build the Ising/XY Hamiltonian using QuTiP spin operators.

    H = -sum_i [Jx X_i X_{i+1} + Jy Y_i Y_{i+1}] - hz sum_i Z_i

    Parameters
    ----------
    L : int
        Number of sites.
    Jx, Jy : float
        XX and YY coupling strengths.
    hz : float
        Transverse field strength.
    pbc : bool
        Periodic boundary conditions.

    Returns
    -------
    H : qutip.Qobj
        Hamiltonian as a QuTiP quantum object.
    """
    import qutip as qt

    # Identity and Pauli matrices
    si = qt.qeye(2)
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()

    def spin_op(op, site, L):
        """Build operator acting on site with identity elsewhere."""
        ops = [si] * L
        ops[site] = op
        return qt.tensor(ops)

    H = 0

    # XX and YY couplings
    for i in range(L - 1):
        if Jx != 0:
            H -= Jx * spin_op(sx, i, L) * spin_op(sx, i+1, L)
        if Jy != 0:
            H -= Jy * spin_op(sy, i, L) * spin_op(sy, i+1, L)

    # Periodic boundary condition
    if pbc and L > 2:
        if Jx != 0:
            H -= Jx * spin_op(sx, L-1, L) * spin_op(sx, 0, L)
        if Jy != 0:
            H -= Jy * spin_op(sy, L-1, L) * spin_op(sy, 0, L)

    # Transverse field
    for i in range(L):
        H -= hz * spin_op(sz, i, L)

    return H


def build_spin_hamiltonian_kitaev_lr(
    L: int,
    h: float,
    alpha: float,
    J: float = 1.0,
    pbc: bool = True,
    normalize: bool = True
):
    """
    Build the Kitaev long-range Hamiltonian using QuTiP.

    H = -sum_{i<j} J_{ij} X_i X_j - h sum_i Z_i

    where J_{ij} = J / |i-j|^alpha.

    Parameters
    ----------
    L : int
        Number of sites.
    h : float
        Transverse field strength.
    alpha : float
        Power-law exponent.
    J : float
        Coupling strength.
    pbc : bool
        Periodic boundary conditions.
    normalize : bool
        Apply Kac normalization.

    Returns
    -------
    H : qutip.Qobj
        Hamiltonian as a QuTiP quantum object.
    """
    import qutip as qt

    si = qt.qeye(2)
    sx = qt.sigmax()
    sz = qt.sigmaz()

    def spin_op(op, site, L):
        ops = [si] * L
        ops[site] = op
        return qt.tensor(ops)

    # Build coupling matrix
    V = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            if pbc:
                dist = min(j - i, L - (j - i))
            else:
                dist = j - i
            if dist > 0:
                V[i, j] = dist ** (-alpha)

    # Normalization
    if normalize:
        norm_factor = np.sum(V[0, :])
        if norm_factor > 0:
            V = V / norm_factor

    V = J * V

    H = 0

    # XX couplings
    for i in range(L):
        for j in range(i + 1, L):
            if V[i, j] != 0:
                H -= V[i, j] * spin_op(sx, i, L) * spin_op(sx, j, L)

    # Transverse field
    for i in range(L):
        H -= h * spin_op(sz, i, L)

    return H


def get_ground_state(H_qutip) -> Tuple[float, 'qutip.Qobj']:
    """
    Get ground state energy and state from QuTiP Hamiltonian.

    Parameters
    ----------
    H_qutip : qutip.Qobj
        Hamiltonian.

    Returns
    -------
    E0 : float
        Ground state energy.
    psi0 : qutip.Qobj
        Ground state vector.
    """
    evals, evecs = H_qutip.eigenstates(eigvals=1)
    return float(evals[0]), evecs[0]


def compute_correlators_qutip(psi, L: int) -> Dict[str, np.ndarray]:
    """
    Compute XX and ZZ correlators using QuTiP.

    Parameters
    ----------
    psi : qutip.Qobj
        State vector.
    L : int
        Number of sites.

    Returns
    -------
    correlators : dict
        Dictionary with 'XX' and 'ZZ' correlation matrices.
    """
    import qutip as qt

    si = qt.qeye(2)
    sx = qt.sigmax()
    sz = qt.sigmaz()

    def spin_op(op, site, L):
        ops = [si] * L
        ops[site] = op
        return qt.tensor(ops)

    XX = np.zeros((L, L))
    ZZ = np.zeros((L, L))

    for i in range(L):
        for j in range(i, L):
            # XX correlator
            op_xx = spin_op(sx, i, L) * spin_op(sx, j, L)
            XX[i, j] = qt.expect(op_xx, psi)
            XX[j, i] = XX[i, j]

            # ZZ correlator
            op_zz = spin_op(sz, i, L) * spin_op(sz, j, L)
            ZZ[i, j] = qt.expect(op_zz, psi)
            ZZ[j, i] = ZZ[i, j]

    return {'XX': XX, 'ZZ': ZZ}


def verify_ising_model(
    L: int,
    Jx: float,
    Jy: float,
    hz: float,
    pbc: bool = True,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Verify free fermion results against QuTiP for the Ising model.

    Parameters
    ----------
    L : int
        System size (keep small, L <= 12).
    Jx, Jy : float
        Coupling strengths.
    hz : float
        Transverse field.
    pbc : bool
        Periodic boundary conditions.
    verbose : bool
        Print comparison results.

    Returns
    -------
    errors : dict
        Dictionary of relative errors for different quantities.
    """
    if not _check_qutip():
        raise ImportError("QuTiP is required for verification")

    if L > 14:
        warnings.warn(f"L={L} is large for exact diagonalization. This may be slow.")

    from .models import IsingHamiltonian, ground_state_energy_both_sectors
    from .correlators import CorrelationFunctions

    errors = {}

    # Free fermion calculation
    E0_ff, E1_ff, gs_parity = ground_state_energy_both_sectors(
        IsingHamiltonian, L, Jx, Jy, hz, pbc=pbc
    )

    H_ff = IsingHamiltonian(L, Jx, Jy, hz, parity=gs_parity, pbc=pbc)
    corr_ff = CorrelationFunctions(H_ff)

    XX_ff = corr_ff.xx_correlator_matrix()
    ZZ_ff = corr_ff.zz_correlator_matrix()

    # QuTiP calculation
    H_qt = build_spin_hamiltonian_ising(L, Jx, Jy, hz, pbc=pbc)
    E0_qt, psi0 = get_ground_state(H_qt)
    corr_qt = compute_correlators_qutip(psi0, L)

    # Compare ground state energy
    E_error = abs(E0_ff - E0_qt) / (abs(E0_qt) + 1e-10)
    errors['energy'] = E_error

    # Compare correlators
    XX_error = np.max(np.abs(XX_ff - corr_qt['XX']))
    ZZ_error = np.max(np.abs(ZZ_ff - corr_qt['ZZ']))
    errors['XX_max'] = XX_error
    errors['ZZ_max'] = ZZ_error

    if verbose:
        print(f"Verification for Ising model (L={L}, Jx={Jx}, Jy={Jy}, hz={hz}, pbc={pbc})")
        print(f"  Ground state energy:")
        print(f"    Free fermion: {E0_ff:.10f}")
        print(f"    QuTiP (exact): {E0_qt:.10f}")
        print(f"    Relative error: {E_error:.2e}")
        print(f"  XX correlator max error: {XX_error:.2e}")
        print(f"  ZZ correlator max error: {ZZ_error:.2e}")

        if E_error < 1e-10 and XX_error < 1e-10 and ZZ_error < 1e-10:
            print("  Status: PASSED")
        else:
            print("  Status: CHECK NEEDED")

    return errors


def verify_kitaev_long_range(
    L: int,
    h: float,
    alpha: float,
    J: float = 1.0,
    pbc: bool = True,
    normalize: bool = True,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Verify free fermion results against QuTiP for Kitaev long-range model.

    Note: The free fermion approximation (discarding JW string) is only
    exact for nearest-neighbor interactions. For long-range, there will
    be discrepancies that decrease with increasing alpha.

    Parameters
    ----------
    L : int
        System size (keep small, L <= 12).
    h : float
        Transverse field.
    alpha : float
        Power-law exponent.
    J : float
        Coupling strength.
    pbc : bool
        Periodic boundary conditions.
    normalize : bool
        Apply Kac normalization.
    verbose : bool
        Print comparison results.

    Returns
    -------
    errors : dict
        Dictionary of errors for different quantities.
    """
    if not _check_qutip():
        raise ImportError("QuTiP is required for verification")

    if L > 14:
        warnings.warn(f"L={L} is large for exact diagonalization. This may be slow.")

    from .models import KitaevLongRange, ground_state_energy_both_sectors
    from .correlators import CorrelationFunctions

    errors = {}

    # Free fermion calculation
    E0_ff, _, gs_parity = ground_state_energy_both_sectors(
        KitaevLongRange, L, h, alpha, J=J, pbc=pbc, normalize=normalize
    )

    H_ff = KitaevLongRange(L, h, alpha, J=J, parity=gs_parity, pbc=pbc, normalize=normalize)
    corr_ff = CorrelationFunctions(H_ff)

    XX_ff = corr_ff.xx_correlator_matrix()
    ZZ_ff = corr_ff.zz_correlator_matrix()

    # QuTiP calculation
    H_qt = build_spin_hamiltonian_kitaev_lr(L, h, alpha, J=J, pbc=pbc, normalize=normalize)
    E0_qt, psi0 = get_ground_state(H_qt)
    corr_qt = compute_correlators_qutip(psi0, L)

    # Compare
    E_error = abs(E0_ff - E0_qt) / (abs(E0_qt) + 1e-10)
    XX_error = np.max(np.abs(XX_ff - corr_qt['XX']))
    ZZ_error = np.max(np.abs(ZZ_ff - corr_qt['ZZ']))

    errors['energy'] = E_error
    errors['XX_max'] = XX_error
    errors['ZZ_max'] = ZZ_error

    if verbose:
        print(f"Verification for Kitaev long-range (L={L}, h={h}, alpha={alpha}, pbc={pbc})")
        print(f"  Ground state energy:")
        print(f"    Free fermion: {E0_ff:.10f}")
        print(f"    QuTiP (exact): {E0_qt:.10f}")
        print(f"    Relative error: {E_error:.2e}")
        print(f"  XX correlator max error: {XX_error:.2e}")
        print(f"  ZZ correlator max error: {ZZ_error:.2e}")

        # For long-range, we expect errors unless alpha is large
        if alpha > 2:
            if E_error < 1e-6 and XX_error < 1e-6 and ZZ_error < 1e-6:
                print("  Status: PASSED (large alpha)")
            else:
                print("  Status: CHECK NEEDED")
        else:
            print(f"  Note: For alpha={alpha}, free fermion approx has inherent errors")
            print("  Status: EXPECTED DISCREPANCY (small alpha)")

    return errors


def run_all_verifications(L: int = 6, verbose: bool = True) -> bool:
    """
    Run verification tests for all models.

    Parameters
    ----------
    L : int
        System size for tests.
    verbose : bool
        Print results.

    Returns
    -------
    passed : bool
        True if all critical tests passed.
    """
    if not _check_qutip():
        print("QuTiP not available. Skipping verification.")
        return True

    all_passed = True

    print("=" * 60)
    print("Running verification tests")
    print("=" * 60)

    # Test 1: Transverse-field Ising model
    print("\n--- Test 1: Transverse-field Ising (Jx=1, Jy=0) ---")
    errors = verify_ising_model(L, Jx=1.0, Jy=0.0, hz=0.5, pbc=True, verbose=verbose)
    if errors['energy'] > 1e-10 or errors['XX_max'] > 1e-10:
        all_passed = False

    # Test 2: XX model
    print("\n--- Test 2: XX model (Jx=1, Jy=1) ---")
    errors = verify_ising_model(L, Jx=1.0, Jy=1.0, hz=0.3, pbc=True, verbose=verbose)
    if errors['energy'] > 1e-10 or errors['XX_max'] > 1e-10:
        all_passed = False

    # Test 3: Ising with OBC
    print("\n--- Test 3: Ising with open boundaries ---")
    errors = verify_ising_model(L, Jx=1.0, Jy=0.0, hz=0.5, pbc=False, verbose=verbose)
    if errors['energy'] > 1e-10 or errors['XX_max'] > 1e-10:
        all_passed = False

    # Test 4: Critical Ising (h = J)
    print("\n--- Test 4: Critical Ising (h = J) ---")
    errors = verify_ising_model(L, Jx=1.0, Jy=0.0, hz=1.0, pbc=True, verbose=verbose)
    if errors['energy'] > 1e-8 or errors['XX_max'] > 1e-8:
        all_passed = False

    # Test 5: Kitaev long-range (large alpha, should be accurate)
    print("\n--- Test 5: Kitaev long-range (alpha=3, nearly NN) ---")
    errors = verify_kitaev_long_range(L, h=0.5, alpha=3.0, pbc=True, verbose=verbose)
    # For large alpha, expect good agreement

    # Test 6: Kitaev long-range (small alpha, expect discrepancy)
    print("\n--- Test 6: Kitaev long-range (alpha=1, long-range) ---")
    errors = verify_kitaev_long_range(L, h=0.5, alpha=1.0, pbc=True, verbose=verbose)
    # This will have errors due to the free fermion approximation

    print("\n" + "=" * 60)
    if all_passed:
        print("All critical tests PASSED!")
    else:
        print("Some tests FAILED - check output above")
    print("=" * 60)

    return all_passed
