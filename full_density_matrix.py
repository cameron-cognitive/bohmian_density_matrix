import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import time
import os
from scipy.sparse import csr_matrix, lil_matrix
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

class FullDensityMatrixSimulation:
    """
    Simulation of Bohmian mechanics with full density matrix implementation.
    Properly handles off-diagonal elements of the density matrix for both
    pure and mixed states.
    """
    
    def __init__(self, N_grid=30, N_modes=4, L=np.pi, N_particles=1000, use_pure_state=True):
        """
        Initialize the full density matrix simulation.
        
        Parameters:
        -----------
        N_grid : int
            Number of grid points in each dimension
            For full density matrix, this is limited by memory (N_grid^4 elements)
        N_modes : int
            Number of modes in each dimension (m,n from 1 to N_modes)
        L : float
            Box side length
        N_particles : int
            Number of particles for simulation
        use_pure_state : bool
            If True, initialize a pure state density matrix
            If False, initialize a mixed state density matrix
        """
        self.N_grid = N_grid
        self.N_modes = N_modes
        self.L = L
        self.N_particles = N_particles
        self.use_pure_state = use_pure_state
        
        # Physical constants
        self.hbar = 1.0
        self.mass = 1.0
        
        # Create spatial grid
        self.x = np.linspace(0, L, N_grid)
        self.y = np.linspace(0, L, N_grid)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize random phases for the wave function (for pure state)
        np.random.seed(42)  # For reproducibility
        self.theta_mn = np.random.uniform(0, 2*np.pi, (N_modes, N_modes))
        
        # Compute energy eigenvalues
        self.eigenvalues = np.zeros((N_modes, N_modes))
        for m in range(1, N_modes+1):
            for n in range(1, N_modes+1):
                self.eigenvalues[m-1, n-1] = self.energy_mn(m, n)
        
        # Store eigenfunctions for efficiency
        self.eigen_grid = {}
        for m in range(1, N_modes+1):
            for n in range(1, N_modes+1):
                self.eigen_grid[(m,n)] = self.phi_mn(m, n, self.X, self.Y)
        
        # Initialize caches for efficiency
        self._phi_cache = {}
        self.current_evolution_time = 0
        self.last_velocity_time = -1  # Invalid time to force initial calculation
        
        # Initialize density matrix and particles
        self.initialize_density_matrix()
        self.initialize_particles(distribution_type='ground_state')
        
        # Storage for H-function data
        self.times = []
        self.h_values = []
        
        # Setup output directory for results
        self.setup_output_directory()
