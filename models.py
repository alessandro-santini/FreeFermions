"""
Model Hamiltonians for free fermion systems.

This module provides factory functions for common spin chain models
that can be mapped to free fermions via Jordan-Wigner transformation:

- Ising model: H = -Jx XX - Jy YY - hz Z
- Kitaev long-range: H = -sum_{i<j} J/|i-j|^α X_i X_j - h Z

For periodic boundary conditions, the Jordan-Wigner transformation
introduces a parity-dependent sign in the boundary term. The system
splits into two parity sectors that should be handled separately.
"""

import numpy as np
from typing import Union, Tuple
from .core import FermionicHamiltonian


def IsingHamiltonian(
    L: int,
    Jx: Union[float, np.ndarray],
    Jy: Union[float, np.ndarray],
    hz: Union[float, np.ndarray],
    parity: int = 0,
    pbc: bool = True
) -> FermionicHamiltonian:
    """
    Create the XY model Hamiltonian (generalized Ising).

    H = -sum_i [Jx_i X_i X_{i+1} + Jy_i Y_i Y_{i+1}] - sum_i hz_i Z_i

    After Jordan-Wigner transformation:
    X_i X_{i+1} -> (c†_i + c_i)(c†_{i+1} + c_{i+1})
    Y_i Y_{i+1} -> (c†_i - c_i)(c†_{i+1} - c_{i+1})
    Z_i -> 1 - 2c†_i c_i

    Parameters
    ----------
    L : int
        Number of sites.
    Jx : float or np.ndarray
        XX coupling strength(s). If array, Jx[i] couples sites i and i+1.
    Jy : float or np.ndarray
        YY coupling strength(s). If array, Jy[i] couples sites i and i+1.
    hz : float or np.ndarray
        Transverse field strength(s).
    parity : int
        Fermion parity sector (0 = even, 1 = odd).
        Only affects PBC: determines sign of boundary term.
    pbc : bool
        If True, use periodic boundary conditions.

    Returns
    -------
    H : FermionicHamiltonian
        The fermionic Hamiltonian.

    Notes
    -----
    For PBC, the boundary term X_L X_1 (or Y_L Y_1) involves a Jordan-Wigner
    string across the whole chain, giving a sign (-1)^N_f where N_f is the
    total fermion number. The parity parameter selects which sector to use.

    Special cases:
    - Jx = J, Jy = 0: Transverse-field Ising model
    - Jx = Jy = J: XX model (free fermion hopping)
    - Jx = -Jy = J: XY model in rotating frame
    """
    # Convert scalar parameters to arrays
    if isinstance(Jx, (float, int)):
        Jx = np.ones(L) * Jx
    if isinstance(Jy, (float, int)):
        Jy = np.ones(L) * Jy
    if isinstance(hz, (float, int)):
        hz = np.ones(L) * hz

    Jx = np.asarray(Jx, dtype=float)
    Jy = np.asarray(Jy, dtype=float)
    hz = np.asarray(hz, dtype=float)

    assert Jx.size == L, f"Jx must have length {L}"
    assert Jy.size == L, f"Jy must have length {L}"
    assert hz.size == L, f"hz must have length {L}"

    # For OBC, set boundary couplings to zero
    if not pbc:
        Jx = Jx.copy()
        Jy = Jy.copy()
        Jx[-1] = 0.0
        Jy[-1] = 0.0

    # Jp = Jx + Jy (symmetric combination)
    # Jm = Jx - Jy (antisymmetric combination)
    Jp = Jx + Jy
    Jm = Jx - Jy

    # Build the A matrix (hopping)
    # H_hop = -J_+ (c†_i c_{i+1} + h.c.) corresponds to A_{i,i+1} = -J_+/2
    A = np.diag(hz, k=0).astype(complex)
    A -= 0.5 * np.diag(Jp[:-1], k=1)
    A -= 0.5 * np.diag(Jp[:-1], k=-1)

    # Build the B matrix (pairing)
    # H_pair = -J_- (c†_i c†_{i+1} - h.c.) corresponds to B_{i,i+1} = -J_-/2
    B = np.zeros((L, L), dtype=complex)
    B -= 0.5 * np.diag(Jm[:-1], k=1)
    B += 0.5 * np.diag(Jm[:-1], k=-1)

    # Periodic boundary conditions with parity
    if pbc:
        parity_sign = (-1) ** parity
        # Boundary hopping
        A[0, -1] = 0.5 * parity_sign * Jp[-1]
        A[-1, 0] = 0.5 * parity_sign * Jp[-1]
        # Boundary pairing
        B[-1, 0] = 0.5 * parity_sign * Jm[-1]
        B[0, -1] = -0.5 * parity_sign * Jm[-1]

    return FermionicHamiltonian(A, B)


def KitaevLongRange(
    L: int,
    h: Union[float, np.ndarray],
    alpha: float,
    J: float = 1.0,
    parity: int = 0,
    pbc: bool = True,
    normalize: bool = True
) -> FermionicHamiltonian:
    """
    Create the Kitaev long-range Hamiltonian.

    H = -sum_{i<j} J_{ij} X_i X_j - h sum_i Z_i

    where J_{ij} = J / |i-j|^α (with optional normalization).

    After Jordan-Wigner transformation and discarding the string operator
    (mean-field / free-fermion approximation):
    X_i X_j -> (c†_i + c_i)(c†_j + c_j)

    Parameters
    ----------
    L : int
        Number of sites.
    h : float or np.ndarray
        Transverse field strength(s).
    alpha : float
        Power-law exponent for the long-range coupling.
        alpha -> infinity recovers nearest-neighbor Ising.
        alpha = 0 is all-to-all coupling.
    J : float
        Overall coupling strength.
    parity : int
        Fermion parity sector (0 = even, 1 = odd).
    pbc : bool
        If True, use periodic boundary conditions.
        For PBC, distance is minimum of |i-j| and L-|i-j|.
    normalize : bool
        If True, normalize couplings by Kac prescription:
        J_{ij} -> J_{ij} / (sum_{j>0} J_{0j})

    Returns
    -------
    H : FermionicHamiltonian
        The fermionic Hamiltonian.

    Notes
    -----
    The "discarding the string" approximation is exact in the thermodynamic
    limit for alpha > 1, where the string contributions vanish. For smaller
    alpha, this becomes a mean-field approximation.

    The parity sectors arise because the Jordan-Wigner string wrapping around
    the periodic boundary gives a sign depending on total fermion number.
    The physical ground state is typically in the even parity sector.
    """
    # Convert scalar h to array
    if isinstance(h, (float, int)):
        h = np.ones(L) * h
    h = np.asarray(h, dtype=float)
    assert h.size == L, f"h must have length {L}"

    # Build the coupling matrix J_{ij}
    V = np.zeros((L, L), dtype=float)

    for i in range(L):
        for j in range(i + 1, L):
            if pbc:
                # Use chord distance for PBC
                dist = min(j - i, L - (j - i))
            else:
                dist = j - i

            if dist > 0:
                V[i, j] = dist ** (-alpha)
                V[j, i] = V[i, j]

    # Normalization (Kac prescription)
    if normalize:
        # Sum over half the chain (avoid double counting)
        if pbc:
            norm_factor = np.sum(V[0, :])
        else:
            norm_factor = np.sum(V[0, 1:])
        if norm_factor > 0:
            V = V / norm_factor

    V = J * V

    # Build A and B matrices
    # X_i X_j = (c†_i + c_i)(c†_j + c_j)
    #         = c†_i c_j + c†_j c_i + c†_i c†_j + c_j c_i
    # This gives hopping A_{ij} = -V_{ij}/2 and pairing B_{ij} = -V_{ij}/2 (antisymmetrized)

    A = np.diag(h).astype(complex)
    A -= V / 2

    # B must be antisymmetric: B_{ij} = -B_{ji}
    # The pairing term c†_i c†_j + c_j c_i with coefficient -V_{ij}
    B = np.zeros((L, L), dtype=complex)
    B -= np.triu(V / 2, k=1)
    B += np.tril(V / 2, k=-1)

    # Handle parity for PBC
    # The parity affects the boundary terms in the long-range coupling
    if pbc and parity == 1:
        # For odd parity, we need to flip the sign of couplings that
        # wrap around the boundary. In the long-range case, this affects
        # all couplings where the "short path" goes through the boundary.
        for i in range(L):
            for j in range(i + 1, L):
                dist_direct = j - i
                dist_wrap = L - dist_direct

                if dist_wrap < dist_direct:
                    # This coupling wraps around
                    A[i, j] *= -1
                    A[j, i] *= -1
                    B[i, j] *= -1
                    B[j, i] *= -1

    return FermionicHamiltonian(A, B)


def both_parity_sectors(
    model_func,
    *args,
    **kwargs
) -> Tuple[FermionicHamiltonian, FermionicHamiltonian]:
    """
    Get Hamiltonians for both parity sectors.

    Parameters
    ----------
    model_func : callable
        Model function (IsingHamiltonian or KitaevLongRange).
    *args, **kwargs
        Arguments to pass to model_func.

    Returns
    -------
    H_even : FermionicHamiltonian
        Hamiltonian in even parity sector.
    H_odd : FermionicHamiltonian
        Hamiltonian in odd parity sector.
    """
    kwargs_even = kwargs.copy()
    kwargs_odd = kwargs.copy()
    kwargs_even['parity'] = 0
    kwargs_odd['parity'] = 1

    H_even = model_func(*args, **kwargs_even)
    H_odd = model_func(*args, **kwargs_odd)

    return H_even, H_odd


def ground_state_energy_both_sectors(
    model_func,
    *args,
    **kwargs
) -> Tuple[float, float, int]:
    """
    Find the true ground state energy by checking both parity sectors.

    Parameters
    ----------
    model_func : callable
        Model function (IsingHamiltonian or KitaevLongRange).
    *args, **kwargs
        Arguments to pass to model_func.

    Returns
    -------
    E0 : float
        Ground state energy (minimum of both sectors).
    E1 : float
        First excited state energy (from the other sector).
    gs_parity : int
        Parity of the ground state (0 or 1).
    """
    H_even, H_odd = both_parity_sectors(model_func, *args, **kwargs)

    E_even = H_even.ground_state_energy()
    E_odd = H_odd.ground_state_energy()

    if E_even <= E_odd:
        return E_even, E_odd, 0
    else:
        return E_odd, E_even, 1
