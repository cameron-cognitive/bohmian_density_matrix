# Debugging Guide for Bohmian Density Matrix Simulation

This guide documents the issues identified and fixed in the Bohmian Density Matrix simulation.

## Key Issues Fixed

### 1. Gradient Calculation Improvements

- **Problem**: Inconsistent interpolation methods for wave function and gradients
- **Solution**: Implemented consistent bilinear interpolation for both wave function and gradient calculations
- **Benefits**: Smoother velocity fields and more accurate particle trajectories

### 2. Numerical Stability Near Nodes

- **Problem**: Division by very small values near wave function nodes
- **Solution**: Added regularization when W_val is close to zero
- **Benefits**: Prevents numerical instabilities and unbounded velocities

### 3. Improved Particle Integration

- **Problem**: Simple Euler integration had large errors
- **Solution**: Implemented RK4 integration with velocity interpolation
- **Benefits**: More accurate particle trajectories

### 4. Better Boundary Handling

- **Problem**: Particles could get stuck at or escape from boundaries
- **Solution**: Improved reflection logic with corner case handling
- **Benefits**: Ensures all particles remain within the domain

### 5. Normalization in H-function

- **Problem**: Inconsistent normalization in H-function calculation
- **Solution**: Proper probability density normalization with bin area
- **Benefits**: More accurate H-function values

### 6. Performance Optimizations

- **Problem**: Redundant computations of density matrix evolution
- **Solution**: Added caching of evolution and velocity fields
- **Benefits**: Significant performance improvement

## How to Monitor Simulation Health

The new simulation includes a `validate_simulation()` method that checks:

1. **Wave function normalization**: Should be close to 1.0
2. **H-function behavior**: Should decrease monotonically
3. **Velocity magnitudes**: Should not exceed reasonable bounds
4. **Particle positions**: Should all remain within the box

Call this method regularly during simulation to ensure everything is working correctly.

## Recommended Simulation Parameters

- **Grid size (N_grid)**: 30-50 for good balance of accuracy and performance
- **Number of modes (N_modes)**: 4 is sufficient for interesting dynamics
- **Time step (dt)**: Keep below 0.1 for stable integration
- **Coarse-graining scale (epsilon)**: π/20 gives good balance for H-function

## Visualization Improvements

1. Added proper contour plots for wave function density
2. Improved velocity field visualization with magnitude coloring
3. Better particle distribution visualization with quantum equilibrium comparison
4. Enhanced H-function plots with exponential fits
