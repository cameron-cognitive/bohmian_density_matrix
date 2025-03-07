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
    
    def setup_output_directory(self):
        """Create directory for saving results."""
        state_type = "pure" if self.use_pure_state else "mixed"
        self.output_dir = f"density_matrix_results_{state_type}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def phi_mn(self, m, n, x, y):
        """Eigenfunction for the 2D infinite square well."""
        # Check if scalar or array inputs
        if np.isscalar(x) and np.isscalar(y):
            cache_key = (m, n, x, y)
            if cache_key in self._phi_cache:
                return self._phi_cache[cache_key]
            
            result = (2/np.sqrt(self.L**2)) * np.sin(m*x*np.pi/self.L) * np.sin(n*y*np.pi/self.L)
            self._phi_cache[cache_key] = result
            return result
        else:
            # For array inputs, use vectorized calculation
            return (2/np.sqrt(self.L**2)) * np.sin(m*x*np.pi/self.L) * np.sin(n*y*np.pi/self.L)
    
    def dphi_mn_dx(self, m, n, x, y):
        """Derivative of eigenfunction with respect to x."""
        return (2/np.sqrt(self.L**2)) * (m*np.pi/self.L) * np.cos(m*x*np.pi/self.L) * np.sin(n*y*np.pi/self.L)
    
    def dphi_mn_dy(self, m, n, x, y):
        """Derivative of eigenfunction with respect to y."""
        return (2/np.sqrt(self.L**2)) * (n*np.pi/self.L) * np.sin(m*x*np.pi/self.L) * np.cos(n*y*np.pi/self.L)
    
    def energy_mn(self, m, n):
        """Energy eigenvalue for the 2D infinite square well."""
        return 0.5 * (m**2 + n**2) * (np.pi/self.L)**2
    
    def initialize_density_matrix(self):
        """Initialize the full density matrix."""
        if self.use_pure_state:
            self.initialize_pure_density_matrix()
        else:
            self.initialize_mixed_density_matrix()
    
    def initialize_pure_density_matrix(self):
        """Initialize a pure state density matrix W(x,y,x',y',0)."""
        # For a pure state, we first build the wave function
        psi = np.zeros((self.N_grid, self.N_grid), dtype=complex)
        
        # Build the wave function (similar to Valentini's paper)
        for m in range(1, self.N_modes+1):
            for n in range(1, self.N_modes+1):
                psi += (1/4) * self.eigen_grid[(m,n)] * np.exp(1j * self.theta_mn[m-1, n-1])
        
        # Store the wave function for reference
        self.psi = psi
        
        # Store diagonal elements for convenience
        self.W_diag = np.abs(psi)**2
        
        # Validate wave function normalization
        norm = np.sum(self.W_diag) * self.dx * self.dy
        print(f"Initial wave function norm: {norm:.6f} (should be close to 1.0)")
        
        # Store the coefficients in the energy eigenbasis
        self.coefficients = np.zeros((self.N_modes, self.N_modes), dtype=complex)
        for m in range(1, self.N_modes+1):
            for n in range(1, self.N_modes+1):
                self.coefficients[m-1, n-1] = (1/4) * np.exp(1j * self.theta_mn[m-1, n-1])
        
        # Initialize the full density matrix in energy eigenbasis
        # For pure state: ρ_mn,pq = c_mn * conj(c_pq)
        self.rho_energy = np.zeros((self.N_modes, self.N_modes, self.N_modes, self.N_modes), dtype=complex)
        for m in range(self.N_modes):
            for n in range(self.N_modes):
                for p in range(self.N_modes):
                    for q in range(self.N_modes):
                        self.rho_energy[m, n, p, q] = self.coefficients[m, n] * np.conj(self.coefficients[p, q])
        
        # Calculate a sample of off-diagonal elements for visualization
        # (Full matrix would be too large to store completely)
        self.calculate_offdiagonal_samples()
    
    def initialize_mixed_density_matrix(self):
        """Initialize a mixed state density matrix."""
        # For a mixed state, we use an equal mixture of energy eigenstates
        # This is the Initial Projection Hypothesis approach
        
        # Calculate diagonal elements for visualization
        self.W_diag = np.zeros((self.N_grid, self.N_grid))
        for m in range(1, self.N_modes+1):
            for n in range(1, self.N_modes+1):
                self.W_diag += (1/16) * self.eigen_grid[(m,n)]**2
        
        # Validate density matrix normalization
        norm = np.sum(self.W_diag) * self.dx * self.dy
        print(f"Initial density matrix norm: {norm:.6f} (should be close to 1.0)")
        
        # For a mixed state, we don't have a wave function
        self.psi = None
        
        # Initialize the full density matrix in energy eigenbasis
        # For this equal mixture: ρ_mn,pq = (1/16) if m=p and n=q, 0 otherwise
        self.rho_energy = np.zeros((self.N_modes, self.N_modes, self.N_modes, self.N_modes), dtype=complex)
        for m in range(self.N_modes):
            for n in range(self.N_modes):
                self.rho_energy[m, n, m, n] = 1/16
        
        # Calculate a sample of off-diagonal elements for visualization
        self.calculate_offdiagonal_samples()
    
    def calculate_offdiagonal_samples(self):
        """
        Calculate and store samples of off-diagonal elements.
        We can't store the full N_grid^4 matrix in memory,
        so we calculate representative samples for visualization.
        """
        # Create a smaller grid for off-diagonal samples
        off_diag_N = min(10, self.N_grid)  # Limit to 10 points for visualization
        off_diag_indices = np.linspace(0, self.N_grid-1, off_diag_N, dtype=int)
        
        # Create storage for samples
        self.W_off_diag_x_slice = np.zeros((self.N_grid, self.N_grid), dtype=complex)
        self.W_off_diag_y_slice = np.zeros((self.N_grid, self.N_grid), dtype=complex)
        
        # Fix a reference point in the middle of the box
        ref_x_idx = self.N_grid // 2
        ref_y_idx = self.N_grid // 2
        ref_x = self.x[ref_x_idx]
        ref_y = self.y[ref_y_idx]
        
        # Calculate W(x,ref_y,ref_x,ref_y) - off-diagonal in x
        for i in range(self.N_grid):
            if self.use_pure_state:
                self.W_off_diag_x_slice[i, ref_y_idx] = self.psi[i, ref_y_idx] * np.conj(self.psi[ref_x_idx, ref_y_idx])
            else:
                # For mixed state, we need to calculate from the energy eigenbasis
                W_val = 0
                for m in range(1, self.N_modes+1):
                    for n in range(1, self.N_modes+1):
                        for p in range(1, self.N_modes+1):
                            for q in range(1, self.N_modes+1):
                                # Only diagonal terms are non-zero for our mixed state
                                if m == p and n == q:
                                    phi_mn_x = self.phi_mn(m, n, self.x[i], ref_y)
                                    phi_pq_x = self.phi_mn(p, q, ref_x, ref_y)
                                    W_val += (1/16) * phi_mn_x * phi_pq_x
                self.W_off_diag_x_slice[i, ref_y_idx] = W_val
        
        # Calculate W(ref_x,y,ref_x,ref_y) - off-diagonal in y
        for j in range(self.N_grid):
            if self.use_pure_state:
                self.W_off_diag_y_slice[ref_x_idx, j] = self.psi[ref_x_idx, j] * np.conj(self.psi[ref_x_idx, ref_y_idx])
            else:
                # For mixed state, we need to calculate from the energy eigenbasis
                W_val = 0
                for m in range(1, self.N_modes+1):
                    for n in range(1, self.N_modes+1):
                        for p in range(1, self.N_modes+1):
                            for q in range(1, self.N_modes+1):
                                # Only diagonal terms are non-zero for our mixed state
                                if m == p and n == q:
                                    phi_mn_y = self.phi_mn(m, n, ref_x, self.y[j])
                                    phi_pq_y = self.phi_mn(p, q, ref_x, ref_y)
                                    W_val += (1/16) * phi_mn_y * phi_pq_y
                self.W_off_diag_y_slice[ref_x_idx, j] = W_val
    
    def evolve_density_matrix(self, t):
        """
        Evolve the full density matrix to time t.
        Uses the energy eigenbasis for efficient time evolution.
        """
        # Check if we already evolved to this time
        if hasattr(self, 'current_evolution_time') and self.current_evolution_time == t:
            return
        
        self.current_evolution_time = t
        
        if self.use_pure_state:
            # Evolve the wave function
            psi_t = np.zeros((self.N_grid, self.N_grid), dtype=complex)
            
            for m in range(1, self.N_modes+1):
                for n in range(1, self.N_modes+1):
                    phase = np.exp(-1j * self.eigenvalues[m-1, n-1] * t)
                    psi_t += self.coefficients[m-1, n-1] * phase * self.eigen_grid[(m,n)]
            
            self.psi = psi_t
            self.W_diag = np.abs(psi_t)**2
            
            # Evolve the density matrix in energy eigenbasis
            # For pure state evolution: ρ_mn,pq(t) = ρ_mn,pq(0) * exp(-i*(E_mn-E_pq)*t/ħ)
            for m in range(self.N_modes):
                for n in range(self.N_modes):
                    for p in range(self.N_modes):
                        for q in range(self.N_modes):
                            energy_diff = self.eigenvalues[m, n] - self.eigenvalues[p, q]
                            phase = np.exp(-1j * energy_diff * t)
                            # Start from initial and apply phase, rather than evolving incrementally
                            initial_coef = self.coefficients[m, n] * np.conj(self.coefficients[p, q])
                            self.rho_energy[m, n, p, q] = initial_coef * phase
            
        else:
            # For a mixed state with this particular initialization (stationary state),
            # the diagonal elements W_diag don't change with time
            # But we still evolve the full density matrix for the guidance equation
            
            # Evolve the density matrix in energy eigenbasis
            # ρ_mn,pq(t) = ρ_mn,pq(0) * exp(-i*(E_mn-E_pq)*t/ħ)
            for m in range(self.N_modes):
                for n in range(self.N_modes):
                    for p in range(self.N_modes):
                        for q in range(self.N_modes):
                            # Skip if initial value is zero (optimization)
                            if abs(self.rho_energy[m, n, p, q]) < 1e-15:
                                continue
                                
                            energy_diff = self.eigenvalues[m, n] - self.eigenvalues[p, q]
                            phase = np.exp(-1j * energy_diff * t)
                            
                            # For an equal mixture of energy eigenstates, only diagonal elements
                            # (m=p and n=q) are non-zero, and their phase is always 1
                            # But we include the calculation for a general mixed state
                            if m == p and n == q:
                                # No need to modify - these stay constant for stationary state
                                pass
                            else:
                                self.rho_energy[m, n, p, q] *= phase
        
        # Update off-diagonal samples for visualization
        self.calculate_offdiagonal_samples()
    
    def compute_W_and_gradient_at_point(self, x, y, t):
        """
        Compute the density matrix value W(x,y,x,y,t) and its gradients
        for the guidance equation at a specific point (x,y).
        
        Using the energy eigenbasis for efficiency:
        W(x,y,x',y',t) = ∑_{m,n,p,q} ρ_{mn,pq}(t) φ_mn(x,y)φ_pq*(x',y')
        
        Parameters:
        -----------
        x, y : float
            Position at which to compute W and its gradient
        t : float
            Time at which to compute W and its gradient
            
        Returns:
        --------
        W_val : float
            Value of W(x,y,x,y,t)
        dW_dx, dW_dy : complex
            Partial derivatives of W(x,y,x',y',t) with respect to x and y,
            evaluated at x'=x, y'=y
        """
        # First update the density matrix to time t (if needed)
        if hasattr(self, 'current_evolution_time') and self.current_evolution_time != t:
            self.evolve_density_matrix(t)
        
        # For pure state, we can use the wave function directly with improved interpolation
        if self.use_pure_state and self.psi is not None:
            # Get grid indices
            i = int((x / self.L) * (self.N_grid - 1))
            j = int((y / self.L) * (self.N_grid - 1))
            i = max(0, min(i, self.N_grid - 2))
            j = max(0, min(j, self.N_grid - 2))
            
            # Bilinear interpolation weights
            dx = (x - self.x[i]) / self.dx
            dy = (y - self.y[j]) / self.dy
            
            # Bilinear interpolation for psi
            psi_val = (1-dx)*(1-dy)*self.psi[i, j] + dx*(1-dy)*self.psi[i+1, j] + \
                     (1-dx)*dy*self.psi[i, j+1] + dx*dy*self.psi[i+1, j+1]
            
            # Calculate gradients using central differences with appropriate handling of boundaries
            if i > 0 and i < self.N_grid-2 and j > 0 and j < self.N_grid-2:
                # Central difference for interior points with interpolation
                # x-gradient at the four corners of the interpolation cell
                dpsi_dx_00 = (self.psi[i+1, j] - self.psi[i-1, j]) / (2*self.dx)
                dpsi_dx_10 = (self.psi[i+2, j] - self.psi[i, j]) / (2*self.dx)
                dpsi_dx_01 = (self.psi[i+1, j+1] - self.psi[i-1, j+1]) / (2*self.dx)
                dpsi_dx_11 = (self.psi[i+2, j+1] - self.psi[i, j+1]) / (2*self.dx)
                
                # Interpolate the gradient
                dpsi_dx = (1-dx)*(1-dy)*dpsi_dx_00 + dx*(1-dy)*dpsi_dx_10 + \
                         (1-dx)*dy*dpsi_dx_01 + dx*dy*dpsi_dx_11
                
                # Similar for y-gradient
                dpsi_dy_00 = (self.psi[i, j+1] - self.psi[i, j-1]) / (2*self.dy)
                dpsi_dy_10 = (self.psi[i+1, j+1] - self.psi[i+1, j-1]) / (2*self.dy)
                dpsi_dy_01 = (self.psi[i, j+2] - self.psi[i, j]) / (2*self.dy)
                dpsi_dy_11 = (self.psi[i+1, j+2] - self.psi[i+1, j]) / (2*self.dy)
                
                dpsi_dy = (1-dx)*(1-dy)*dpsi_dy_00 + dx*(1-dy)*dpsi_dy_10 + \
                         (1-dx)*dy*dpsi_dy_01 + dx*dy*dpsi_dy_11
            else:
                # One-sided differences for boundary points
                if i == 0:
                    dpsi_dx_j = (self.psi[i+1, j] - self.psi[i, j]) / self.dx
                    dpsi_dx_j1 = (self.psi[i+1, j+1] - self.psi[i, j+1]) / self.dx
                elif i == self.N_grid-2:
                    dpsi_dx_j = (self.psi[i+1, j] - self.psi[i, j]) / self.dx
                    dpsi_dx_j1 = (self.psi[i+1, j+1] - self.psi[i, j+1]) / self.dx
                else:
                    dpsi_dx_j = (self.psi[i+1, j] - self.psi[i-1, j]) / (2*self.dx)
                    dpsi_dx_j1 = (self.psi[i+1, j+1] - self.psi[i-1, j+1]) / (2*self.dx)
                
                if j == 0:
                    dpsi_dy_i = (self.psi[i, j+1] - self.psi[i, j]) / self.dy
                    dpsi_dy_i1 = (self.psi[i+1, j+1] - self.psi[i+1, j]) / self.dy
                elif j == self.N_grid-2:
                    dpsi_dy_i = (self.psi[i, j+1] - self.psi[i, j]) / self.dy
                    dpsi_dy_i1 = (self.psi[i+1, j+1] - self.psi[i+1, j]) / self.dy
                else:
                    dpsi_dy_i = (self.psi[i, j+1] - self.psi[i, j-1]) / (2*self.dy)
                    dpsi_dy_i1 = (self.psi[i+1, j+1] - self.psi[i+1, j-1]) / (2*self.dy)
                
                # Interpolate gradients
                dpsi_dx = (1-dy)*dpsi_dx_j + dy*dpsi_dx_j1
                dpsi_dy = (1-dx)*dpsi_dy_i + dx*dpsi_dy_i1
            
            # Calculate W and its gradients
            W_val = np.abs(psi_val)**2
            dW_dx = 2 * np.real(dpsi_dx * np.conj(psi_val))
            dW_dy = 2 * np.real(dpsi_dy * np.conj(psi_val))
            
            # Handle numerical instability for small W_val
            epsilon = 1e-10  # Small regularization constant
            if W_val < epsilon:
                # Use the limit of the gradient/W as W approaches 0
                # This is a mathematical approximation that avoids division by very small numbers
                # For a pure state, this approaches 2*∇|ψ|/|ψ| as |ψ|→0
                if np.abs(psi_val) > epsilon/10:
                    dW_dx_by_W = 2 * np.real(dpsi_dx / psi_val)
                    dW_dy_by_W = 2 * np.real(dpsi_dy / psi_val)
                    return epsilon, dW_dx_by_W, dW_dy_by_W
                else:
                    return epsilon, 0, 0
        else:
            # For mixed state or if we need to calculate directly from energy eigenbasis
            W_val = 0
            dW_dx = 0j
            dW_dy = 0j
            
            # More efficient calculation for our specific mixed state
            if not self.use_pure_state:
                # For our mixed state with only diagonal elements in rho_energy
                for m in range(1, self.N_modes+1):
                    for n in range(1, self.N_modes+1):
                        phi_val = self.phi_mn(m, n, x, y)
                        dphi_dx = self.dphi_mn_dx(m, n, x, y)
                        dphi_dy = self.dphi_mn_dy(m, n, x, y)
                        
                        # Add diagonal contribution (m=p, n=q)
                        coef = 1/16  # Fixed for our equal mixture
                        W_val += coef * phi_val**2
                        dW_dx += coef * 2 * phi_val * dphi_dx
                        dW_dy += coef * 2 * phi_val * dphi_dy
            else:
                # General calculation for any density matrix
                for m in range(1, self.N_modes+1):
                    for n in range(1, self.N_modes+1):
                        phi_mn = self.phi_mn(m, n, x, y)
                        dphi_mn_dx = self.dphi_mn_dx(m, n, x, y)
                        dphi_mn_dy = self.dphi_mn_dy(m, n, x, y)
                        
                        for p in range(1, self.N_modes+1):
                            for q in range(1, self.N_modes+1):
                                phi_pq = self.phi_mn(p, q, x, y)
                                dphi_pq_dx = self.dphi_mn_dx(p, q, x, y)
                                dphi_pq_dy = self.dphi_mn_dy(p, q, x, y)
                                
                                # Get the time-evolved coefficient
                                m_idx, n_idx = m-1, n-1
                                p_idx, q_idx = p-1, q-1
                                
                                coef = self.rho_energy[m_idx, n_idx, p_idx, q_idx]
                                
                                # Add contribution to W and its gradients
                                W_val += np.real(coef * phi_mn * phi_pq)
                                dW_dx += coef * (dphi_mn_dx * phi_pq + phi_mn * dphi_pq_dx)
                                dW_dy += coef * (dphi_mn_dy * phi_pq + phi_mn * dphi_pq_dy)
            
            # Handle numerical instability for small W_val
            epsilon = 1e-10  # Small regularization constant
            if W_val < epsilon:
                return epsilon, dW_dx/epsilon, dW_dy/epsilon
        
        return W_val, dW_dx, dW_dy