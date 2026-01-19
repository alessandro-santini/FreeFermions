#!/usr/bin/env python
"""
Example: Transverse-Field Ising Model

Demonstrates:
1. Creating Ising Hamiltonian with periodic boundary conditions
2. Computing ground state energy and correlations
3. Entanglement entropy
4. Quench dynamics

Run with: uv run FreeFermions/examples/example_ising.py
"""

import numpy as np
import matplotlib.pyplot as plt
from FreeFermions import (
    IsingHamiltonian,
    CorrelationFunctions,
    SuddenQuench,
    ground_state_energy_both_sectors,
)


def main():
    print("=" * 60)
    print("Transverse-Field Ising Model")
    print("H = -J Σ σ^x_i σ^x_{i+1} - h Σ σ^z_i")
    print("=" * 60)

    # System parameters
    L = 64  # Chain length
    J = 1.0  # XX coupling
    h = 0.5  # Transverse field

    print(f"\nSystem size: L = {L}")
    print(f"Coupling: J = {J}")
    print(f"Field: h = {h}")
    print(f"Phase: {'Ferromagnetic (h < J)' if h < J else 'Paramagnetic (h > J)'}")

    # Get ground state energy from both parity sectors
    E_even, E_odd, gs_parity = ground_state_energy_both_sectors(
        IsingHamiltonian, L, J, 0, h, pbc=True
    )
    print(f"\nGround state energy: E_0 = {E_even:.6f}")
    print(f"Energy per site: E_0/L = {E_even/L:.6f}")
    print(f"Ground state parity sector: {gs_parity}")

    # Create Hamiltonian in ground state sector
    H = IsingHamiltonian(L, Jx=J, Jy=0, hz=h, parity=gs_parity, pbc=True)

    # Compute correlations
    corr = CorrelationFunctions(H)

    # Local magnetization
    mz = corr.magnetization_z()
    print(f"\nAverage magnetization <σ^z> = {np.mean(mz):.6f}")

    # ZZ correlation function
    ZZ = corr.zz_correlator_matrix()
    print(f"ZZ correlation <σ^z_0 σ^z_1> = {ZZ[0,1]:.6f}")

    # XX correlation (nearest-neighbor)
    XX_nn = corr.xx_correlator(0, 1)
    print(f"XX correlation <σ^x_0 σ^x_1> = {XX_nn.real:.6f}")

    # Entanglement entropy
    entropy = corr.entanglement_entropy_profile()
    S_half = corr.entanglement_entropy(L // 2)
    print(f"Half-chain entanglement entropy: S(L/2) = {S_half:.6f}")

    # ============ Quench Dynamics ============
    print("\n" + "=" * 60)
    print("Quench Dynamics: h = 10 → h = 1 (paramagnetic → critical)")
    print("=" * 60)

    # Initial state: ground state of paramagnetic Hamiltonian
    H0 = IsingHamiltonian(L, Jx=J, Jy=0, hz=10.0, parity=0, pbc=True)
    # Final Hamiltonian: critical point
    H1 = IsingHamiltonian(L, Jx=J, Jy=0, hz=1.0, parity=0, pbc=True)

    quench = SuddenQuench(H0, H1)

    # Time evolution
    times = np.linspace(0, 10, 50)
    results = quench.time_series(times, observables=['energy', 'mz'])

    print(f"Initial energy: {results['energy'][0]:.6f}")
    print(f"Final energy (t=10): {results['energy'][-1]:.6f}")
    print(f"Target GS energy: {H1.ground_state_energy():.6f}")

    # ============ Plotting ============
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: ZZ correlation vs distance
    ax = axes[0, 0]
    distances = np.arange(1, L // 2)
    zz_vs_dist = [ZZ[0, d] for d in distances]
    ax.plot(distances, zz_vs_dist, 'b.-')
    ax.set_xlabel('Distance r')
    ax.set_ylabel(r'$\langle\sigma^z_0 \sigma^z_r\rangle$')
    ax.set_title('ZZ Correlation Function')
    ax.grid(True, alpha=0.3)

    # Plot 2: Entanglement entropy profile
    ax = axes[0, 1]
    subsystem_sizes = np.arange(1, L)
    ax.plot(subsystem_sizes, entropy, 'g.-')
    ax.set_xlabel('Subsystem size l')
    ax.set_ylabel('Entanglement entropy S(l)')
    ax.set_title('Entanglement Entropy Profile')
    ax.grid(True, alpha=0.3)

    # Plot 3: Energy after quench
    ax = axes[1, 0]
    ax.plot(times, results['energy'], 'b-', linewidth=2)
    ax.axhline(y=H1.ground_state_energy(), color='r', linestyle='--',
               label='GS energy of H₁')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Energy ⟨H₁⟩')
    ax.set_title('Energy After Quench')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Magnetization after quench
    ax = axes[1, 1]
    avg_mz = np.mean(results['mz'], axis=1)
    ax.plot(times, avg_mz, 'm-', linewidth=2)
    ax.set_xlabel('Time t')
    ax.set_ylabel(r'$\langle\sigma^z\rangle$')
    ax.set_title('Average Magnetization After Quench')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ising_example.png', dpi=150)
    print(f"\nPlot saved to 'ising_example.png'")
    plt.show()


if __name__ == "__main__":
    main()
