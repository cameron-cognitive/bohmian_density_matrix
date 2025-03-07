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
        return (2/np.sqrt(self.L**2)) * np.sin(m*x*np.pi/self.L) * np.sin(n*y*np.pi/self.L)
    
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
                            self.rho_energy[m, n, p, q] *= phase
            
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
                            energy_diff = self.eigenvalues[m, n] - self.eigenvalues[p, q]
                            phase = np.exp(-1j * energy_diff * t)
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
        # First, calculate W(x,y,x,y,t)
        W_val = 0
        
        # For pure state, we can use the wave function directly
        if self.use_pure_state and self.psi is not None:
            # Interpolate psi value at the given point
            i = int((x / self.L) * (self.N_grid - 1))
            j = int((y / self.L) * (self.N_grid - 1))
            i = max(0, min(i, self.N_grid - 2))
            j = max(0, min(j, self.N_grid - 2))
            
            # Simple bilinear interpolation
            dx = (x - self.x[i]) / self.dx
            dy = (y - self.y[j]) / self.dy
            
            psi_val = (1-dx)*(1-dy)*self.psi[i, j] + dx*(1-dy)*self.psi[i+1, j] + \
                     (1-dx)*dy*self.psi[i, j+1] + dx*dy*self.psi[i+1, j+1]
            
            W_val = np.abs(psi_val)**2
        else:
            # For mixed state or if we need to calculate directly from energy eigenbasis
            for m in range(1, self.N_modes+1):
                for n in range(1, self.N_modes+1):
                    for p in range(1, self.N_modes+1):
                        for q in range(1, self.N_modes+1):
                            phi_mn = self.phi_mn(m, n, x, y)
                            phi_pq = self.phi_mn(p, q, x, y)
                            
                            # Get the time-evolved coefficient
                            m_idx, n_idx = m-1, n-1
                            p_idx, q_idx = p-1, q-1
                            
                            # For our mixed state with only diagonal elements
                            if self.use_pure_state or (m == p and n == q):
                                W_val += np.real(self.rho_energy[m_idx, n_idx, p_idx, q_idx] * phi_mn * phi_pq)
        
        # Now calculate gradients of W(x,y,x',y',t) at (x'=x,y'=y)
        dW_dx = 0j
        dW_dy = 0j
        
        # For pure state, we can calculate from the wave function
        if self.use_pure_state and self.psi is not None:
            # Interpolate psi and its gradients
            i = int((x / self.L) * (self.N_grid - 1))
            j = int((y / self.L) * (self.N_grid - 1))
            i = max(0, min(i, self.N_grid - 2))
            j = max(0, min(j, self.N_grid - 2))
            
            # Calculate gradients of psi
            dpsi_dx = (self.psi[i+1, j] - self.psi[i, j]) / self.dx
            dpsi_dy = (self.psi[i, j+1] - self.psi[i, j]) / self.dy
            
            # For pure state W(x,y,x',y',t) = ψ(x,y,t)ψ*(x',y',t)
            # when evaluated at x'=x, y'=y, the gradient is:
            # ∇_x W(x,y,x,y,t) = ∇_x ψ(x,y,t) · ψ*(x,y,t) + ψ(x,y,t) · ∇_x ψ*(x,y,t)
            psi_val = self.psi[i, j]
            dW_dx = dpsi_dx * np.conj(psi_val) + psi_val * np.conj(dpsi_dx)
            dW_dy = dpsi_dy * np.conj(psi_val) + psi_val * np.conj(dpsi_dy)
        else:
            # Calculate from energy eigenbasis
            for m in range(1, self.N_modes+1):
                for n in range(1, self.N_modes+1):
                    for p in range(1, self.N_modes+1):
                        for q in range(1, self.N_modes+1):
                            m_idx, n_idx = m-1, n-1
                            p_idx, q_idx = p-1, q-1
                            
                            # Only include non-zero coefficients
                            if not self.use_pure_state and (m != p or n != q):
                                continue
                                
                            phi_mn = self.phi_mn(m, n, x, y)
                            phi_pq = self.phi_mn(p, q, x, y)
                            
                            # Derivatives of φ_mn(x,y)
                            dphi_mn_dx = (2/np.sqrt(self.L**2)) * m*np.pi/self.L * np.cos(m*x*np.pi/self.L) * np.sin(n*y*np.pi/self.L)
                            dphi_mn_dy = (2/np.sqrt(self.L**2)) * n*np.pi/self.L * np.sin(m*x*np.pi/self.L) * np.cos(n*y*np.pi/self.L)
                            
                            # Derivatives of φ_pq(x,y)
                            dphi_pq_dx = (2/np.sqrt(self.L**2)) * p*np.pi/self.L * np.cos(p*x*np.pi/self.L) * np.sin(q*y*np.pi/self.L)
                            dphi_pq_dy = (2/np.sqrt(self.L**2)) * q*np.pi/self.L * np.sin(p*x*np.pi/self.L) * np.cos(q*y*np.pi/self.L)
                            
                            # Contribution to gradients
                            dW_dx += self.rho_energy[m_idx, n_idx, p_idx, q_idx] * (dphi_mn_dx * phi_pq + phi_mn * dphi_pq_dx)
                            dW_dy += self.rho_energy[m_idx, n_idx, p_idx, q_idx] * (dphi_mn_dy * phi_pq + phi_mn * dphi_pq_dy)
        
        return W_val, dW_dx, dW_dy
    
    def compute_velocity_field(self, t, grid_downsample=1):
        """
        Compute the Bohmian velocity field based on the density matrix.
        
        Parameters:
        -----------
        t : float
            Time at which to compute the velocity field
        grid_downsample : int
            Factor by which to downsample the grid for faster computation
            
        Returns:
        --------
        vx, vy : ndarray
            Components of the Bohmian velocity field
        """
        # First update the density matrix to time t
        self.evolve_density_matrix(t)
        
        # Prepare downsampled grid
        N_down = self.N_grid // grid_downsample
        vx = np.zeros((N_down, N_down))
        vy = np.zeros((N_down, N_down))
        
        # Compute velocities on the downsampled grid
        for i in range(N_down):
            for j in range(N_down):
                # Get position on original grid
                x = self.x[i * grid_downsample]
                y = self.y[j * grid_downsample]
                
                # Compute W and its gradients at this point
                W_val, dW_dx, dW_dy = self.compute_W_and_gradient_at_point(x, y, t)
                
                # Compute velocities using guidance equation
                if np.abs(W_val) > 1e-10:  # Avoid division by zero
                    vx[i, j] = self.hbar / self.mass * np.imag(dW_dx / W_val)
                    vy[i, j] = self.hbar / self.mass * np.imag(dW_dy / W_val)
        
        return vx, vy
    
    def initialize_particles(self, distribution_type='ground_state'):
        """
        Initialize particles in a specified non-equilibrium distribution.
        
        Parameters:
        -----------
        distribution_type : str
            Type of initial distribution:
            - 'ground_state': |φ₁₁|² (as in Valentini's paper)
            - 'uniform': Uniform distribution
            - 'custom': Custom distribution (requires additional parameters)
        """
        self.particles = np.zeros((self.N_particles, 2))
        
        if distribution_type == 'ground_state':
            # Generate particles according to ground state distribution
            for i in range(self.N_particles):
                accepted = False
                while not accepted:
                    # Generate uniform random points in the box
                    x_trial = np.random.uniform(0, self.L)
                    y_trial = np.random.uniform(0, self.L)
                    
                    # Calculate ground state probability at this point
                    prob = self.phi_mn(1, 1, x_trial, y_trial)**2
                    
                    # Accept with probability proportional to the ground state
                    if np.random.uniform(0, 1) < prob / np.max(self.phi_mn(1, 1, self.X, self.Y)**2):
                        self.particles[i] = [x_trial, y_trial]
                        accepted = True
        
        elif distribution_type == 'uniform':
            # Generate uniform distribution
            for i in range(self.N_particles):
                x = np.random.uniform(0, self.L)
                y = np.random.uniform(0, self.L)
                self.particles[i] = [x, y]
        
        elif distribution_type == 'custom':
            # Example of custom distribution (e.g., concentrated in a corner)
            for i in range(self.N_particles):
                x = np.random.uniform(0, self.L/2)  # Only in left half
                y = np.random.uniform(0, self.L/2)  # Only in bottom half
                self.particles[i] = [x, y]
    
    def update_particles(self, t, dt, grid_downsample=2):
        """
        Update particle positions based on the velocity field.
        
        Parameters:
        -----------
        t : float
            Current time
        dt : float
            Time step
        grid_downsample : int
            Factor by which to downsample the grid for faster computation
        """
        # Compute velocity field at current time
        vx_grid, vy_grid = self.compute_velocity_field(t, grid_downsample)
        N_down = self.N_grid // grid_downsample
        
        # Update each particle
        for i in range(self.N_particles):
            x, y = self.particles[i]
            
            # Find the nearest grid point in the downsampled grid
            ix = int((x / self.L) * (N_down - 1))
            iy = int((y / self.L) * (N_down - 1))
            ix = max(0, min(ix, N_down - 1))
            iy = max(0, min(iy, N_down - 1))
            
            # Get velocity from the grid
            vx = vx_grid[ix, iy]
            vy = vy_grid[ix, iy]
            
            # Update positions
            new_x = x + vx * dt
            new_y = y + vy * dt
            
            # Apply boundary conditions - reflection at boundaries
            if new_x < 0:
                new_x = -new_x
            elif new_x > self.L:
                new_x = 2*self.L - new_x
                
            if new_y < 0:
                new_y = -new_y
            elif new_y > self.L:
                new_y = 2*self.L - new_y
            
            self.particles[i] = [new_x, new_y]
    
    def calculate_h_function(self, epsilon=None, bins=20):
        """
        Calculate the coarse-grained H-function.
        
        Parameters:
        -----------
        epsilon : float or None
            Coarse-graining length. If None, use bins parameter instead.
        bins : int
            Number of bins for coarse-graining if epsilon is None.
            
        Returns:
        --------
        H : float
            Coarse-grained H-function value
        """
        if epsilon is not None:
            bins = int(self.L / epsilon)
        
        # Create histogram of particle positions
        hist, x_edges, y_edges = np.histogram2d(
            self.particles[:, 0], self.particles[:, 1], 
            bins=[bins, bins], range=[[0, self.L], [0, self.L]]
        )
        
        # Normalize to get empirical probability density
        hist = hist / np.sum(hist)
        
        # Resample W_diag onto the histogram grid for comparison
        W_diag_resampled = np.zeros((bins, bins))
        for i in range(bins):
            for j in range(bins):
                x_center = (x_edges[i] + x_edges[i+1]) / 2
                y_center = (y_edges[j] + y_edges[j+1]) / 2
                
                # Calculate W_diag at this point
                W_val, _, _ = self.compute_W_and_gradient_at_point(x_center, y_center, self.current_time)
                W_diag_resampled[i, j] = W_val
        
        # Normalize W_diag_resampled
        W_diag_resampled = W_diag_resampled / np.sum(W_diag_resampled)
        
        # Calculate H-function
        H = 0
        for i in range(bins):
            for j in range(bins):
                if hist[i, j] > 0 and W_diag_resampled[i, j] > 0:
                    H += hist[i, j] * np.log(hist[i, j] / W_diag_resampled[i, j])
        
        return H
    
    def visualize_density_matrix(self, t, save_path=None):
        """
        Visualize the density matrix at time t.
        Shows diagonal elements and off-diagonal slices.
        
        Parameters:
        -----------
        t : float
            Time at which to visualize the density matrix
        save_path : str or None
            If provided, save the figure to this path
        """
        # First update the density matrix to time t
        self.evolve_density_matrix(t)
        
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Diagonal elements W(x,y,x,y)
        ax1 = fig.add_subplot(2, 3, 1)
        c1 = ax1.contourf(self.X, self.Y, self.W_diag, cmap='viridis')
        plt.colorbar(c1, ax=ax1, label='|W(x,y,x,y)|')
        ax1.set_title(f'Diagonal elements at t={t:.2f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        # 2. Real part of off-diagonal slice in x
        ax2 = fig.add_subplot(2, 3, 2)
        c2 = ax2.contourf(self.X, self.Y, np.real(self.W_off_diag_x_slice), cmap='RdBu')
        plt.colorbar(c2, ax=ax2, label='Re[W(x,y₀,x₀,y₀)]')
        ax2.set_title(f'Real part of x off-diagonal at t={t:.2f}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        
        # Add a marker for the reference point
        ref_x_idx = self.N_grid // 2
        ref_y_idx = self.N_grid // 2
        ax2.plot(self.x[ref_x_idx], self.y[ref_y_idx], 'ko', markersize=5)
        
        # 3. Imaginary part of off-diagonal slice in x
        ax3 = fig.add_subplot(2, 3, 3)
        c3 = ax3.contourf(self.X, self.Y, np.imag(self.W_off_diag_x_slice), cmap='RdBu')
        plt.colorbar(c3, ax=ax3, label='Im[W(x,y₀,x₀,y₀)]')
        ax3.set_title(f'Imaginary part of x off-diagonal at t={t:.2f}')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        
        # Add a marker for the reference point
        ax3.plot(self.x[ref_x_idx], self.y[ref_y_idx], 'ko', markersize=5)
        
        # 4. Real part of off-diagonal slice in y
        ax4 = fig.add_subplot(2, 3, 5)
        c4 = ax4.contourf(self.X, self.Y, np.real(self.W_off_diag_y_slice), cmap='RdBu')
        plt.colorbar(c4, ax=ax4, label='Re[W(x₀,y,x₀,y₀)]')
        ax4.set_title(f'Real part of y off-diagonal at t={t:.2f}')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        
        # Add a marker for the reference point
        ax4.plot(self.x[ref_x_idx], self.y[ref_y_idx], 'ko', markersize=5)
        
        # 5. Imaginary part of off-diagonal slice in y
        ax5 = fig.add_subplot(2, 3, 6)
        c5 = ax5.contourf(self.X, self.Y, np.imag(self.W_off_diag_y_slice), cmap='RdBu')
        plt.colorbar(c5, ax=ax5, label='Im[W(x₀,y,x₀,y₀)]')
        ax5.set_title(f'Imaginary part of y off-diagonal at t={t:.2f}')
        ax5.set_xlabel('x')
        ax5.set_ylabel('y')
        
        # Add a marker for the reference point
        ax5.plot(self.x[ref_x_idx], self.y[ref_y_idx], 'ko', markersize=5)
        
        # Title for the whole figure
        state_type = "Pure" if self.use_pure_state else "Mixed"
        plt.suptitle(f'{state_type} State Density Matrix at t={t:.2f}', fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        plt.show()
    
    def visualize_velocity_field(self, t, save_path=None):
        """
        Visualize the Bohmian velocity field at time t.
        
        Parameters:
        -----------
        t : float
            Time at which to visualize the velocity field
        save_path : str or None
            If provided, save the figure to this path
        """
        # Compute velocity field
        grid_downsample = 4  # Downsample for better visualization
        vx, vy = self.compute_velocity_field(t, grid_downsample)
        
        # Create downsampled grid
        N_down = self.N_grid // grid_downsample
        x_down = np.linspace(0, self.L, N_down)
        y_down = np.linspace(0, self.L, N_down)
        X_down, Y_down = np.meshgrid(x_down, y_down)
        
        # Calculate velocity magnitude
        speed = np.sqrt(vx**2 + vy**2)
        
        plt.figure(figsize=(12, 10))
        
        # Plot velocity field
        plt.quiver(X_down, Y_down, vx, vy, speed, cmap='viridis', scale=30, width=0.002)
        plt.colorbar(label='Velocity magnitude')
        
        # Add contours of W_diag
        plt.contour(self.X, self.Y, self.W_diag, colors='white', alpha=0.3)
        
        state_type = "Pure" if self.use_pure_state else "Mixed"
        plt.title(f'Bohmian Velocity Field ({state_type} State) at t={t:.2f}', fontsize=14)
        plt.xlabel('x')
        plt.ylabel('y')
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        plt.show()
    
    def visualize_particle_distribution(self, t, epsilon=None, bins=20, save_path=None):
        """
        Visualize the particle distribution at time t.
        
        Parameters:
        -----------
        t : float
            Current time
        epsilon : float or None
            Coarse-graining length. If None, use bins parameter instead.
        bins : int
            Number of bins for visualization if epsilon is None.
        save_path : str or None
            If provided, save the figure to this path
        """
        if epsilon is not None:
            bins = int(self.L / epsilon)
        
        plt.figure(figsize=(12, 10))
        
        # Plot particle histogram
        h, x_edges, y_edges, img = plt.hist2d(
            self.particles[:, 0], self.particles[:, 1], 
            bins=[bins, bins], range=[[0, self.L], [0, self.L]], 
            cmap='plasma'
        )
        plt.colorbar(label='Particle count')
        
        # Add contours of W_diag
        plt.contour(self.X, self.Y, self.W_diag, colors='white', alpha=0.5)
        
        state_type = "Pure" if self.use_pure_state else "Mixed"
        plt.title(f'Particle Distribution ({state_type} State) at t={t:.2f}', fontsize=14)
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Add H-function value to the plot
        H = self.calculate_h_function(epsilon, bins)
        plt.text(0.05, 0.95, f'H = {H:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        plt.show()
        
        return H
    
    def run_simulation(self, t_max=10.0, dt=0.1, save_interval=1.0, epsilon=None, grid_downsample=2):
        """
        Run the full simulation up to time t_max.
        
        Parameters:
        -----------
        t_max : float
            Maximum simulation time
        dt : float
            Time step
        save_interval : float
            Interval at which to save results
        epsilon : float or None
            Coarse-graining length for H-function
        grid_downsample : int
            Factor by which to downsample the grid for faster computation
            
        Returns:
        --------
        times : list
            List of time points
        h_values : list
            Corresponding H-function values
        """
        n_steps = int(t_max / dt)
        save_steps = int(save_interval / dt)
        
        self.times = []
        self.h_values = []
        self.current_time = 0
        
        # Save initial state
        print(f"Initial state at t=0")
        self.visualize_density_matrix(0, save_path=f"{self.output_dir}/density_matrix_t0.png")
        self.visualize_velocity_field(0, save_path=f"{self.output_dir}/velocity_field_t0.png")
        H = self.visualize_particle_distribution(0, epsilon, save_path=f"{self.output_dir}/particles_t0.png")
        self.times.append(0)
        self.h_values.append(H)
        
        # Main simulation loop
        for step in range(n_steps):
            t = step * dt
            self.current_time = t
            
            # Update particle positions
            self.update_particles(t, dt, grid_downsample)
            
            # Calculate H-function
            H = self.calculate_h_function(epsilon)
            self.times.append(t)
            self.h_values.append(H)
            
            # Save state at regular intervals
            if step % save_steps == 0:
                print(f"Step {step}, t={t:.2f}, H={H:.4f}")
                self.visualize_density_matrix(t, save_path=f"{self.output_dir}/density_matrix_t{t:.1f}.png")
                self.visualize_velocity_field(t, save_path=f"{self.output_dir}/velocity_field_t{t:.1f}.png")
                self.visualize_particle_distribution(t, epsilon, save_path=f"{self.output_dir}/particles_t{t:.1f}.png")
        
        # Save final state
        print(f"Final state at t={t_max}")
        self.visualize_density_matrix(t_max, save_path=f"{self.output_dir}/density_matrix_final.png")
        self.visualize_velocity_field(t_max, save_path=f"{self.output_dir}/velocity_field_final.png")
        self.visualize_particle_distribution(t_max, epsilon, save_path=f"{self.output_dir}/particles_final.png")
        
        # Save H-function data
        np.savetxt(f"{self.output_dir}/h_function.csv", np.column_stack((self.times, self.h_values)), 
                 delimiter=',', header='time,h_function')
        
        # Plot H-function over time
        plt.figure(figsize=(10, 6))
        plt.plot(self.times, self.h_values, 'b-')
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('H-function')
        plt.title('Evolution of coarse-grained H-function')
        plt.savefig(f"{self.output_dir}/h_function.png", dpi=300)
        plt.show()
        
        # Fit exponential decay to H-function
        try:
            from scipy.optimize import curve_fit
            
            def exp_decay(t, a, b, c):
                return a * np.exp(-b * t) + c
            
            params, _ = curve_fit(exp_decay, self.times, self.h_values)
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.times, self.h_values, 'b-', label='Data')
            plt.plot(self.times, exp_decay(np.array(self.times), *params), 'r--', 
                    label=f'Fit: {params[0]:.2f}*exp(-{params[1]:.2f}*t) + {params[2]:.2f}')
            plt.grid(True)
            plt.xlabel('Time')
            plt.ylabel('H-function')
            plt.title('Exponential fit to H-function decay')
            plt.legend()
            plt.savefig(f"{self.output_dir}/h_function_fit.png", dpi=300)
            plt.show()
            
            print(f"Decay constant: {params[1]:.4f}")
        except:
            print("Could not fit exponential decay to H-function data")
        
        return self.times, self.h_values

def main():
    # Run pure state simulation with full density matrix
    print("Running pure state simulation with full density matrix...")
    sim_pure = FullDensityMatrixSimulation(
        N_grid=30,  # Reduced for full density matrix (memory intensive)
        N_modes=4, 
        L=np.pi, 
        N_particles=1000, 
        use_pure_state=True
    )
    times_pure, h_values_pure = sim_pure.run_simulation(
        t_max=10.0, 
        dt=0.1, 
        save_interval=2.0, 
        epsilon=np.pi/10  # Coarse-graining length
    )
    
    # Run mixed state simulation with full density matrix
    print("Running mixed state simulation with full density matrix...")
    sim_mixed = FullDensityMatrixSimulation(
        N_grid=30,  # Reduced for full density matrix (memory intensive)
        N_modes=4, 
        L=np.pi, 
        N_particles=1000, 
        use_pure_state=False
    )
    times_mixed, h_values_mixed = sim_mixed.run_simulation(
        t_max=10.0, 
        dt=0.1, 
        save_interval=2.0, 
        epsilon=np.pi/10  # Coarse-graining length
    )
    
    # Compare results
    plt.figure(figsize=(10, 6))
    plt.plot(times_pure, h_values_pure, 'b-', label='Pure State')
    plt.plot(times_mixed, h_values_mixed, 'r-', label='Mixed State')
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('H-function')
    plt.title('Comparison of H-function evolution')
    plt.legend()
    plt.savefig("density_matrix_results_comparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()