#!/usr/bin/env python
"""
Example: Small System Analysis with QuTiP Verification

Demonstrates:
1. Comparison with exact diagonalization (QuTiP)
2. Both parity sectors
3. Excited states analysis
4. Correlation functions verification

Run with: uv run FreeFermions/examples/example_small_system.py
"""

import numpy as np
import qutip as qt
from FreeFermions import (
    IsingHamiltonian,
    CorrelationFunctions,
    both_parity_sectors,
    ground_state_energy_both_sectors,
)


def build_qutip_ising(L, Jx, hz, pbc=True):
    """Build Ising Hamiltonian using QuTiP for exact diagonalization."""
    si = qt.qeye(2)
    sx = qt.sigmax()
    sz = qt.sigmaz()

    def spin_op(op, site):
        ops = [si] * L
        ops[site] = op
        return qt.tensor(ops)

    H = 0
    # XX couplings
    for i in range(L - 1):
        H -= Jx * spin_op(sx, i) * spin_op(sx, i + 1)
    if pbc:
        H -= Jx * spin_op(sx, L - 1) * spin_op(sx, 0)
    # Transverse field
    for i in range(L):
        H -= hz * spin_op(sz, i)

    return H


def main():
    print("=" * 60)
    print("Small System Analysis: Free Fermions vs Exact Diagonalization")
    print("=" * 60)

    L = 6  # Small system for exact comparison
    Jx = 1.0
    hz = 0.5

    print(f"\nSystem size: L = {L}")
    print(f"Parameters: Jx = {Jx}, hz = {hz}")

    # ============ QuTiP Exact Diagonalization ============
    print("\n--- QuTiP Exact Diagonalization ---")

    H_qt = build_qutip_ising(L, Jx, hz, pbc=True)
    energies_qt, states_qt = H_qt.eigenstates()

    print(f"Full spectrum (first 10 levels):")
    for i in range(min(10, len(energies_qt))):
        print(f"  E_{i} = {energies_qt[i]:.6f}")

    # Ground state
    E0_qt = energies_qt[0]
    psi0 = states_qt[0]

    # ============ Free Fermion Calculation ============
    print("\n--- Free Fermion Calculation ---")

    # Both parity sectors
    H_even, H_odd = both_parity_sectors(IsingHamiltonian, L, Jx, 0, hz, pbc=True)

    E0_even = H_even.ground_state_energy()
    E0_odd = H_odd.ground_state_energy()

    print(f"Even parity sector ground state: E = {E0_even:.6f}")
    print(f"Odd parity sector ground state: E = {E0_odd:.6f}")

    # Single-particle spectrum
    print(f"\nSingle-particle energies (even sector):")
    for i, eps in enumerate(H_even.eigs[:5]):
        print(f"  ε_{i} = {eps:.6f}")

    # ============ Comparison ============
    print("\n--- Comparison ---")

    E0_ff, E1_ff, gs_parity = ground_state_energy_both_sectors(
        IsingHamiltonian, L, Jx, 0, hz, pbc=True
    )

    print(f"QuTiP ground state energy: {E0_qt:.10f}")
    print(f"Free fermion ground state energy: {E0_ff:.10f}")
    print(f"Difference: {abs(E0_qt - E0_ff):.2e}")

    # Correlation functions
    print("\n--- Correlation Functions ---")

    si = qt.qeye(2)
    sx, sz = qt.sigmax(), qt.sigmaz()

    def spin_op(op, site):
        ops = [si] * L
        ops[site] = op
        return qt.tensor(ops)

    # Free fermion correlations
    H_ff = IsingHamiltonian(L, Jx, 0, hz, parity=gs_parity, pbc=True)
    corr = CorrelationFunctions(H_ff)

    print("\nZZ correlations <σ^z_0 σ^z_j>:")
    print("  j | QuTiP | FreeFermion | Error")
    ZZ = corr.zz_correlator_matrix()
    for j in range(L):
        zz_qt = qt.expect(spin_op(sz, 0) * spin_op(sz, j), psi0)
        zz_ff = ZZ[0, j].real
        print(f"  {j} | {zz_qt:7.4f} | {zz_ff:11.4f} | {abs(zz_qt - zz_ff):.2e}")

    print("\nXX correlations <σ^x_0 σ^x_j> (NN only reliable):")
    print("  j | QuTiP | FreeFermion | Error")
    for j in range(min(3, L)):  # Only show first few
        xx_qt = qt.expect(spin_op(sx, 0) * spin_op(sx, j), psi0)
        xx_ff = corr.xx_correlator(0, j).real
        status = "✓" if abs(xx_qt - xx_ff) < 0.01 else "~"
        print(f"  {j} | {xx_qt:7.4f} | {xx_ff:11.4f} | {abs(xx_qt - xx_ff):.2e} {status}")
    print("  Note: XX correlator exact for NN, approximate for longer distances")

    # ============ Excited States Analysis ============
    print("\n--- Excited States (Parity Sectors) ---")

    print("\nThe many-body spectrum can be built from single-particle energies:")
    print("E_n = E_0 + sum of excitation energies")

    # Build some excited state energies from single-particle spectrum
    eps_even = H_even.eigs
    eps_odd = H_odd.eigs

    print(f"\nEven sector excitations (ε_k):")
    for k in range(min(4, len(eps_even))):
        print(f"  ε_{k} = {eps_even[k]:.6f}")

    print(f"\nFirst few many-body energies from free fermions:")
    # Ground state
    print(f"  GS: E = {E0_ff:.6f} (parity {gs_parity})")
    # First excitation in each sector
    print(f"  Even sector 1st exc: E = {E0_even + 2*eps_even[0]:.6f}")
    print(f"  Odd sector GS: E = {E0_odd:.6f}")

    print("\nQuTiP many-body spectrum (first 6):")
    for i in range(min(6, len(energies_qt))):
        print(f"  E_{i} = {energies_qt[i]:.6f}")

    # ============ Phase Transition ============
    print("\n--- Phase Transition Signature ---")

    h_values = np.linspace(0.2, 2.0, 10)
    gaps = []

    print("\nEnergy gap vs transverse field:")
    print("  h   | Gap")
    for h in h_values:
        E0, E1, _ = ground_state_energy_both_sectors(
            IsingHamiltonian, L, Jx, 0, h, pbc=True
        )
        gap = E1 - E0
        gaps.append(gap)
        if abs(h - 1.0) < 0.1:  # Near critical point
            print(f"  {h:.1f} | {gap:.6f} <- near critical point")
        else:
            print(f"  {h:.1f} | {gap:.6f}")

    print("\nGap closes at h = J = 1 (critical point)")


if __name__ == "__main__":
    main()
