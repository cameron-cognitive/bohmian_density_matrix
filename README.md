# Density Matrix Bohmian Relaxation Simulation

This repository contains code for simulating quantum relaxation dynamics in Bohmian mechanics using the density matrix formalism, as described in Eddy Keming Chen's paper "Density Matrix Realism."

## Overview

Bohmian mechanics traditionally uses a universal wave function to guide particles. However, as Chen argues, we can also formulate Bohmian mechanics using a density matrix as the fundamental object, which can be either pure or mixed. This project provides a numerical implementation of this formalism and investigates relaxation to quantum equilibrium under both pure and mixed state guidance.

## Theoretical Background

### Wave Function Realism vs. Density Matrix Realism

- **Wave Function Realism (WFR)**: The quantum state of the universe is objective; it must be pure.
- **Density Matrix Realism (DMR)**: The quantum state of the universe is objective; it can be pure or impure.

### The Density Matrix Guidance Equation

In density matrix Bohmian mechanics, the particle velocity is given by:

```
dQ_i/dt = (ℏ/m_i) Im[∇_i W(q,q',t) / W(q,q',t)]|(q=q'=Q)
```

where W is the density matrix, and the expression is evaluated at the diagonal point where q=q'=Q.

### Quantum Relaxation

A key question in Bohmian mechanics is how an initial non-equilibrium distribution ρ(q,t) ≠ |ψ(q,t)|² evolves toward the equilibrium distribution |ψ(q,t)|². In the density matrix formalism, the equilibrium distribution is given by the diagonal elements W(q,q,t).

We measure relaxation using the H-function:

```
H = ∫ ρ ln(ρ/W(q,q,t)) dq
```

which decreases over time as the system approaches equilibrium.

## Code Structure

The repository contains the following main components:

1. `full_density_matrix.py`: Complete implementation of density matrix evolution and particle dynamics
2. `quantum_metrics.py`: Quantum information metrics for analyzing pure and mixed states
3. `analysis_script.py`: Advanced analysis tools for comparing relaxation behavior
4. `simulation_config.md`: Documentation of the current wave function and initial distribution

## Running the Simulations

### Prerequisites

- Python 3.7+
- NumPy
- SciPy
- Matplotlib

### Installation

1. Clone this repository:
```bash
git clone https://github.com/cameron-cognitive/bohmian_density_matrix.git
cd bohmian_density_matrix
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Simulation

To run the full density matrix simulation with both pure and mixed states:

```bash
python full_density_matrix.py
```

To run the quantum information metrics analysis:

```bash
python quantum_metrics.py
```

To run advanced analysis comparing pure and mixed states:

```bash
python analysis_script.py
```

## Experiments

The implementation allows for several key experiments:

1. **Pure vs. Mixed State Relaxation**: Compare relaxation rates between pure state and mixed state density matrices
2. **Coarse-Graining Effects**: Analyze how different coarse-graining scales affect the H-function
3. **Initial Distribution Dependence**: Test different initial non-equilibrium distributions
4. **Quantum Information Analysis**: Investigate correlations between quantum information metrics and relaxation

## Visualizations

The simulation produces visualizations of:

1. **Density Matrix**: Diagonal and off-diagonal elements over time
2. **Particle Distribution**: Evolution of particle positions
3. **Velocity Field**: Bohmian velocity field derived from the density matrix
4. **H-function**: Relaxation curve showing approach to equilibrium
5. **Quantum Metrics**: Entropy, purity, and participation ratio over time

## Current Test Configuration

We are currently using:

- **Wave Function**: Superposition of 16 energy eigenstates (m,n=1,2,3,4) with equal amplitudes but random phases
- **Initial Distribution**: Ground state distribution |φ₁₁|² as used in Valentini's paper
- **Box Parameters**: L = π, with infinite potential walls

See `simulation_config.md` for detailed information.

## Results

Our simulations show:

1. Both pure and mixed state density matrices lead to relaxation toward quantum equilibrium
2. Off-diagonal elements of the density matrix play a crucial role in the guidance equation
3. Mixed states show distinct relaxation behavior compared to pure states

## References

1. Chen, E. K. (2024). "Density Matrix Realism." In The Open Systems View: Physics, Metaphysics and Methodology. Oxford University Press.
2. Valentini, A., & Westman, H. (2005). "Dynamical Origin of Quantum Probabilities." Proceedings of the Royal Society A, 461, 253-272.
3. Dürr, D., Goldstein, S., Tumulka, R., & Zanghì, N. (2005). "On the Role of Density Matrices in Bohmian Mechanics." Foundations of Physics, 35(3), 449-467.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
