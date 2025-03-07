# Wave Function and Initial Distribution Documentation

This document explains the current wave function we are using and the initial particle distribution in our density matrix Bohmian mechanics simulation.

## Wave Function

We are using a superposition of the first 16 energy eigenstates of a 2D box, with modes m,n = 1,2,3,4. The wave function at time t=0 is:

$$\psi(x,y,0) = \frac{1}{4}\sum_{m=1}^{4}\sum_{n=1}^{4} \phi_{mn}(x,y)e^{i\theta_{mn}}$$

Where:
- $\phi_{mn}(x,y) = \frac{2}{\sqrt{L^2}} \sin(m\pi x/L) \sin(n\pi y/L)$ are the energy eigenfunctions
- $\theta_{mn}$ are random phases, unique for each mode
- $L = \pi$ is the box side length

This is the same wave function used in Valentini's paper, which allows for a direct comparison with his results.

### Properties of the Wave Function:

1. **Energy Eigenvalues**: $E_{mn} = \frac{1}{2}(m^2 + n^2)(\pi/L)^2$
2. **Time Evolution**: $\psi(x,y,t) = \frac{1}{4}\sum_{m=1}^{4}\sum_{n=1}^{4} \phi_{mn}(x,y)e^{i(\theta_{mn} - E_{mn}t)}$
3. **Recurrence Time**: The wave function recurs to its initial value every $T = 4\pi$ time units
4. **Amplitude Distribution**: Equal amplitudes (1/4) for all modes, but with random phases
5. **Position Probability Density**: $|\psi(x,y,t)|^2$ evolves complexly over time

## Initial Particle Distribution

For the non-equilibrium particle distribution, we are using the ground state distribution as in Valentini's paper:

$$\rho(x,y,0) = |\phi_{11}(x,y)|^2 = \left(\frac{2}{\pi}\right)^2 \sin^2(x) \sin^2(y)$$

This is different from the equilibrium distribution $|\psi(x,y,0)|^2$, which would be the Born rule distribution.

### Properties of the Initial Distribution:

1. **Single Mode**: Corresponds to only the ground state energy eigenfunction
2. **Simple Structure**: Has a single maximum at the center of the box
3. **Non-Equilibrium**: Not equal to $|\psi(x,y,0)|^2$, providing a clear demonstration of relaxation

## Density Matrix Formulation

### Pure State Case
For the pure state case, we formulate the density matrix as:

$$W(x,y,x',y',0) = \psi(x,y,0)\psi^*(x',y',0)$$

### Mixed State Case
For the mixed state case, we use an equal mixture of all 16 energy eigenstates:

$$W(x,y,x',y',0) = \frac{1}{16}\sum_{m=1}^{4}\sum_{n=1}^{4} \phi_{mn}(x,y)\phi_{mn}(x',y')$$

This represents a fundamentally impure quantum state with no coherence between different energy eigenstates.

## Comparison to Valentini's Experiment

Our simulation aims to recreate Valentini's experiment with two important extensions:

1. We implement the full density matrix formalism, allowing for both pure and mixed states
2. We can analyze the role of off-diagonal elements in the relaxation process

Valentini showed that an initial non-equilibrium distribution approaches the quantum equilibrium distribution $|\psi|^2$ over time. Our experiments extend this to investigate whether:

1. The same relaxation occurs with mixed state density matrices
2. The relaxation rate depends on the purity of the quantum state
3. There are fundamental differences in how pure and mixed states guide Bohmian particles

This setup provides a robust framework for investigating quantum relaxation dynamics in the context of Density Matrix Realism.