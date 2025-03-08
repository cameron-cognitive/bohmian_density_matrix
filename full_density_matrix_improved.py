import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent plots from showing

class FullDensityMatrixSimulation:
    """
    Simulation of Bohmian mechanics using the full density matrix formalism.
    Implements both pure and mixed state evolution with improved numerical stability.
    """
    
    def __init__(self, N_grid=30, N_modes=4, L=np.pi, N_particles=1000, use_pure_state=True, 
                 output_dir=None, num_offdiag_slices=3):
        """
        Initialize the simulation parameters.
        
        Parameters:
        -----------
        N_grid : int
            Number of grid points in each dimension
        N_modes : int
            Number of modes for the superposition (1 to N_modes in each dimension)
        L : float
            Box side length
        N_particles : int
            Number of particles in the simulation
        use_pure_state : bool
            If True, use a pure state density matrix
            If False, use a mixed state density matrix
        output_dir : str or None
            Directory for saving simulation results
        num_offdiag_slices : int
            Number of off-diagonal slices to track on each side of the diagonal
        """
        self.N_grid = N_grid
        self.N_modes = N_modes
        self.L = L
        self.N_particles = N_particles
        self.use_pure_state = use_pure_state
        self.num_offdiag_slices = num_offdiag_slices
        
        # Ensure the current_time is set right at the beginning
        self.current_time = None
        
        # Create output directory
        if output_dir is None:
            if use_pure_state:
                self.output_dir = "density_matrix_results_pure"
            else:
                self.output_dir = "density_matrix_results_mixed"
        else:
            self.output_dir = output_dir
            
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Create animation frames directory
        self.frames_dir = os.path.join(self.output_dir, "frames")
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)
        
        # Grid setup
        self.x = np.linspace(0, L, N_grid)
        self.y = np.linspace(0, L, N_grid)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.dx = L / (N_grid - 1)
        self.dy = L / (N_grid - 1)
        
        # Initialize storage containers for density matrix data
        self.W_diag = None
        self.W_diag_grad_x = None
        self.W_diag_grad_y = None
        self.Im_grad_x_W = None
        self.Im_grad_y_W = None
        
        # Offsets for multiple off-diagonal slices
        # We'll use a fraction of the box size as the step
        self.offset_step = L / (N_grid * 3)  # 1/3 of a grid cell
        
        # Storage for multiple off-diagonal slices
        # Each slice is a 2D array of shape (N_grid, N_grid)
        self.W_offdiag_x_slices = []  # W(x+dx, y, x, y)
        self.W_offdiag_y_slices = []  # W(x, y+dy, x, y)
        self.offset_values = []
        
        # For tracking H-function
        self.times = []
        self.h_values = []
        
        # Initialize density matrix and particles
        self.initialize_density_matrix()
        self.initialize_particles()
        
        print(f"Initialized {'pure' if use_pure_state else 'mixed'} state simulation with {N_grid}x{N_grid} grid")
        print(f"Tracking {num_offdiag_slices} off-diagonal slices on each side of the diagonal")
    
    def initialize_density_matrix(self):
        """
        Initialize the density matrix.
        
        For pure state: superposition of eigenstates with random phases
        For mixed state: equal mixture of energy eigenstates
        """
        # Create random phases for the superposition
        np.random.seed(42)  # For reproducibility
        self.phases = np.random.uniform(0, 2*np.pi, (self.N_modes, self.N_modes))
        
        # Energy eigenvalues for the 2D box
        self.energies = np.zeros((self.N_modes, self.N_modes))
        for m in range(1, self.N_modes+1):
            for n in range(1, self.N_modes+1):
                self.energies[m-1, n-1] = 0.5 * (m**2 + n**2) * (np.pi/self.L)**2
        
        # For pure state, initialize psi
        if self.use_pure_state:
            self.psi = np.zeros((self.N_grid, self.N_grid), dtype=complex)
            self.initialize_pure_state_wavefunction()
        else:
            self.psi = None
        
        # Evolve to t=0 to initialize the density matrix
        self.evolve_density_matrix(0)
        
        print("Density matrix initialized")
    
    def initialize_pure_state_wavefunction(self):
        """
        Initialize the pure state wave function as a superposition of energy eigenstates.
        """
        # Normalization factor
        norm_factor = 1.0 / self.N_modes
        
        # For each position on the grid
        for i in range(self.N_grid):
            for j in range(self.N_grid):
                x = self.x[i]
                y = self.y[j]
                
                # Sum over all modes
                for m in range(1, self.N_modes+1):
                    for n in range(1, self.N_modes+1):
                        # Energy eigenfunction with phase
                        phi_mn = np.sqrt(2/self.L) * np.sin(m*np.pi*x/self.L) * np.sqrt(2/self.L) * np.sin(n*np.pi*y/self.L)
                        phase = self.phases[m-1, n-1]
                        
                        # Add to superposition
                        self.psi[i, j] += norm_factor * phi_mn * np.exp(1j * phase)
    
    def initialize_particles(self, distribution_type='ground_state'):
        """
        Initialize particle positions according to a specified distribution.
        
        Parameters:
        -----------
        distribution_type : str
            Type of initial distribution:
            - 'ground_state': |φ₁₁|² distribution (non-equilibrium)
            - 'uniform': Uniform distribution across the box
            - 'custom': Custom distribution (e.g., Gaussian)
        """
        # Initialize particle array
        self.particles = np.zeros((self.N_particles, 2))
        
        if distribution_type == 'ground_state':
            # Ground state distribution |φ₁₁|² as in Valentini's paper
            accepted = 0
            while accepted < self.N_particles:
                # Random positions in the box
                x_trial = np.random.uniform(0, self.L)
                y_trial = np.random.uniform(0, self.L)
                
                # Ground state probability density
                p = (2/self.L)**2 * np.sin(np.pi*x_trial/self.L)**2 * np.sin(np.pi*y_trial/self.L)**2
                
                # Accept with probability proportional to p
                if np.random.uniform(0, 1) < p / (4/self.L**2):  # Normalized by maximum value
                    self.particles[accepted] = [x_trial, y_trial]
                    accepted += 1
        
        elif distribution_type == 'uniform':
            # Uniform distribution
            self.particles[:, 0] = np.random.uniform(0, self.L, self.N_particles)
            self.particles[:, 1] = np.random.uniform(0, self.L, self.N_particles)
        
        elif distribution_type == 'custom':
            # Example: Gaussian distribution centered at (L/2, L/2)
            mean = [self.L/2, self.L/2]
            cov = [[self.L/10, 0], [0, self.L/10]]
            self.particles = np.random.multivariate_normal(mean, cov, self.N_particles)
            
            # Ensure particles are within bounds
            self.particles = np.clip(self.particles, 0, self.L)
        
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")
        
        print(f"Initialized {self.N_particles} particles with '{distribution_type}' distribution")
    
    def evolve_density_matrix(self, t):
        """
        Evolve the density matrix to time t.
        Caches values for efficiency.
        
        Parameters:
        -----------
        t : float
            Time to which to evolve the density matrix
        """
        # Check if we've already evolved to this time
        if t == self.current_time and self.W_diag is not None:
            return  # Already at the right time, nothing to do
        
        # Store the current time
        self.current_time = t
        
        # Initialize diagonal elements and multiple off-diagonal slices
        self.W_diag = np.zeros((self.N_grid, self.N_grid))
        
        # Clear previous off-diagonal slices
        self.W_offdiag_x_slices = []
        self.W_offdiag_y_slices = []
        self.offset_values = []
        
        # Create offset values for off-diagonal slices
        for i in range(1, self.num_offdiag_slices + 1):
            # Both positive and negative offsets
            self.offset_values.append(i * self.offset_step)
            self.offset_values.append(-i * self.offset_step)
        
        # Sort offset values
        self.offset_values.sort()
        
        # Initialize storage for off-diagonal slices
        for _ in self.offset_values:
            self.W_offdiag_x_slices.append(np.zeros((self.N_grid, self.N_grid), dtype=complex))
            self.W_offdiag_y_slices.append(np.zeros((self.N_grid, self.N_grid), dtype=complex))
        
        if self.use_pure_state:
            # Evolve pure state wave function
            self.evolve_pure_state(t)
            
            # Calculate diagonal elements W(x,y,x,y) = |ψ(x,y,t)|^2
            self.W_diag = np.abs(self.psi)**2
            
            # Calculate off-diagonal slices with various offsets
            for idx, offset in enumerate(self.offset_values):
                # Calculate X off-diagonal slice: W(x+offset, y, x, y)
                for i in range(self.N_grid):
                    for j in range(self.N_grid):
                        x = self.x[i]
                        y = self.y[j]
                        
                        # Calculate position with offset
                        x_offset = x + offset
                        
                        # Handle boundary conditions
                        if x_offset < 0 or x_offset > self.L:
                            # Out of bounds - set to zero
                            self.W_offdiag_x_slices[idx][i, j] = 0.0
                        else:
                            # Calculate psi at original and offset positions
                            psi_xy = self.psi[i, j]
                            psi_offset = self.compute_psi_at_point(x_offset, y, t)
                            
                            # Calculate off-diagonal element
                            self.W_offdiag_x_slices[idx][i, j] = psi_xy * np.conj(psi_offset)
                
                # Calculate Y off-diagonal slice: W(x, y+offset, x, y)
                for i in range(self.N_grid):
                    for j in range(self.N_grid):
                        x = self.x[i]
                        y = self.y[j]
                        
                        # Calculate position with offset
                        y_offset = y + offset
                        
                        # Handle boundary conditions
                        if y_offset < 0 or y_offset > self.L:
                            # Out of bounds - set to zero
                            self.W_offdiag_y_slices[idx][i, j] = 0.0
                        else:
                            # Calculate psi at original and offset positions
                            psi_xy = self.psi[i, j]
                            psi_offset = self.compute_psi_at_point(x, y_offset, t)
                            
                            # Calculate off-diagonal element
                            self.W_offdiag_y_slices[idx][i, j] = psi_xy * np.conj(psi_offset)
        else:
            # Mixed state density matrix (equal mixture of energy eigenstates)
            # For diagonal elements
            for i in range(self.N_grid):
                for j in range(self.N_grid):
                    x = self.x[i]
                    y = self.y[j]
                    
                    # Diagonal elements - equal mixture of all eigenstates
                    diag_sum = 0
                    for m in range(1, self.N_modes+1):
                        for n in range(1, self.N_modes+1):
                            phi_mn = np.sqrt(2/self.L) * np.sin(m*np.pi*x/self.L) * np.sqrt(2/self.L) * np.sin(n*np.pi*y/self.L)
                            diag_sum += phi_mn**2 / (self.N_modes**2)
                    
                    self.W_diag[i, j] = diag_sum
            
            # For off-diagonal slices
            for idx, offset in enumerate(self.offset_values):
                # X off-diagonal slice
                for i in range(self.N_grid):
                    for j in range(self.N_grid):
                        x = self.x[i]
                        y = self.y[j]
                        x_offset = x + offset
                        
                        # Handle boundary conditions
                        if x_offset < 0 or x_offset > self.L:
                            self.W_offdiag_x_slices[idx][i, j] = 0.0
                        else:
                            # Calculate mixed state off-diagonal elements
                            x_slice_sum = 0
                            for m in range(1, self.N_modes+1):
                                for n in range(1, self.N_modes+1):
                                    phi_mn_xy = np.sqrt(2/self.L) * np.sin(m*np.pi*x/self.L) * np.sqrt(2/self.L) * np.sin(n*np.pi*y/self.L)
                                    phi_mn_offset = np.sqrt(2/self.L) * np.sin(m*np.pi*x_offset/self.L) * np.sqrt(2/self.L) * np.sin(n*np.pi*y/self.L)
                                    
                                    E_mn = self.energies[m-1, n-1]
                                    phase = np.exp(-1j * E_mn * t)
                                    
                                    x_slice_sum += phi_mn_xy * phi_mn_offset * phase / (self.N_modes**2)
                            
                            self.W_offdiag_x_slices[idx][i, j] = x_slice_sum
                
                # Y off-diagonal slice
                for i in range(self.N_grid):
                    for j in range(self.N_grid):
                        x = self.x[i]
                        y = self.y[j]
                        y_offset = y + offset
                        
                        # Handle boundary conditions
                        if y_offset < 0 or y_offset > self.L:
                            self.W_offdiag_y_slices[idx][i, j] = 0.0
                        else:
                            # Calculate mixed state off-diagonal elements
                            y_slice_sum = 0
                            for m in range(1, self.N_modes+1):
                                for n in range(1, self.N_modes+1):
                                    phi_mn_xy = np.sqrt(2/self.L) * np.sin(m*np.pi*x/self.L) * np.sqrt(2/self.L) * np.sin(n*np.pi*y/self.L)
                                    phi_mn_offset = np.sqrt(2/self.L) * np.sin(m*np.pi*x/self.L) * np.sqrt(2/self.L) * np.sin(n*np.pi*y_offset/self.L)
                                    
                                    E_mn = self.energies[m-1, n-1]
                                    phase = np.exp(-1j * E_mn * t)
                                    
                                    y_slice_sum += phi_mn_xy * phi_mn_offset * phase / (self.N_modes**2)
                            
                            self.W_offdiag_y_slices[idx][i, j] = y_slice_sum
        
        # Precompute gradients for efficiency
        self.W_diag_grad_x = np.zeros_like(self.W_diag)
        self.W_diag_grad_y = np.zeros_like(self.W_diag)
        
        # Calculate gradients using central differences (interior points)
        self.W_diag_grad_x[1:-1, 1:-1] = (self.W_diag[2:, 1:-1] - self.W_diag[:-2, 1:-1]) / (2 * self.dx)
        self.W_diag_grad_y[1:-1, 1:-1] = (self.W_diag[1:-1, 2:] - self.W_diag[1:-1, :-2]) / (2 * self.dy)
        
        # Handle boundaries with one-sided differences
        # Left and right boundaries
        self.W_diag_grad_x[0, :] = (self.W_diag[1, :] - self.W_diag[0, :]) / self.dx
        self.W_diag_grad_x[-1, :] = (self.W_diag[-1, :] - self.W_diag[-2, :]) / self.dx
        
        # Top and bottom boundaries
        self.W_diag_grad_y[:, 0] = (self.W_diag[:, 1] - self.W_diag[:, 0]) / self.dy
        self.W_diag_grad_y[:, -1] = (self.W_diag[:, -1] - self.W_diag[:, -2]) / self.dy
        
        # Precompute the imaginary part of the off-diagonal gradient
        self.compute_Im_grad_W()
    
    def evolve_pure_state(self, t):
        """
        Evolve the pure state wave function to time t.
        
        Parameters:
        -----------
        t : float
            Time to which to evolve the wave function
        """
        # For each position on the grid
        for i in range(self.N_grid):
            for j in range(self.N_grid):
                x = self.x[i]
                y = self.y[j]
                
                # Start with zero
                self.psi[i, j] = 0
                
                # Sum over all modes
                for m in range(1, self.N_modes+1):
                    for n in range(1, self.N_modes+1):
                        # Energy eigenfunction with phase
                        phi_mn = np.sqrt(2/self.L) * np.sin(m*np.pi*x/self.L) * np.sqrt(2/self.L) * np.sin(n*np.pi*y/self.L)
                        E_mn = self.energies[m-1, n-1]
                        phase = self.phases[m-1, n-1] - E_mn * t
                        
                        # Add to superposition with time evolution
                        self.psi[i, j] += (1.0 / self.N_modes) * phi_mn * np.exp(1j * phase)
    
    def compute_psi_at_point(self, x, y, t):
        """
        Compute the wave function at an arbitrary point (x,y) at time t.
        
        Parameters:
        -----------
        x, y : float
            Coordinates at which to compute psi
        t : float
            Time at which to compute psi
            
        Returns:
        --------
        psi_val : complex
            Wave function value at (x,y,t)
        """
        if not self.use_pure_state:
            return None
        
        psi_val = 0
        norm_factor = 1.0 / self.N_modes
        
        # Sum over all modes
        for m in range(1, self.N_modes+1):
            for n in range(1, self.N_modes+1):
                # Energy eigenfunction with phase
                phi_mn = np.sqrt(2/self.L) * np.sin(m*np.pi*x/self.L) * np.sqrt(2/self.L) * np.sin(n*np.pi*y/self.L)
                E_mn = self.energies[m-1, n-1]
                phase = self.phases[m-1, n-1] - E_mn * t
                
                # Add to superposition
                psi_val += norm_factor * phi_mn * np.exp(1j * phase)
        
        return psi_val
    
    def compute_Im_grad_W(self):
        """
        Compute the imaginary part of the gradient of W that appears in the
        Bohmian velocity equation.
        
        For the guidance equation we need:
        Im[∇_i W(q,q',t) / W(q,q',t)]|(q=q'=Q)
        
        This is computed using the off-diagonal slices of W.
        """
        # Create arrays for the velocity field components
        self.Im_grad_x_W = np.zeros_like(self.W_diag)
        self.Im_grad_y_W = np.zeros_like(self.W_diag)
        
        # Add a small regularization constant to avoid division by zero
        epsilon = 1e-10
        
        # Find the index of the slice closest to zero offset
        # This should be the middle slice when sorted
        if len(self.offset_values) > 0:
            # Find the slice with smallest magnitude offset
            closest_idx = np.argmin(np.abs(self.offset_values))
            smallest_offset = self.offset_values[closest_idx]
            
            # Use this slice for calculating the gradient
            for i in range(self.N_grid):
                for j in range(self.N_grid):
                    # For x gradient
                    W_val = self.W_diag[i, j]
                    W_offdiag = self.W_offdiag_x_slices[closest_idx][i, j]
                    
                    # Avoid division by zero
                    if abs(W_val) > epsilon:
                        # Calculate Im[∇W/W] using the smallest offset slice
                        # This is an approximation of the derivative at the diagonal
                        self.Im_grad_x_W[i, j] = np.imag((W_offdiag - W_val) / (smallest_offset * W_val))
                    else:
                        self.Im_grad_x_W[i, j] = 0.0
                    
                    # For y gradient
                    W_offdiag = self.W_offdiag_y_slices[closest_idx][i, j]
                    
                    # Avoid division by zero
                    if abs(W_val) > epsilon:
                        self.Im_grad_y_W[i, j] = np.imag((W_offdiag - W_val) / (smallest_offset * W_val))
                    else:
                        self.Im_grad_y_W[i, j] = 0.0
        
        # Replace any NaN values with zeros (extra safety)
        self.Im_grad_x_W = np.nan_to_num(self.Im_grad_x_W)
        self.Im_grad_y_W = np.nan_to_num(self.Im_grad_y_W)
    
    def compute_velocity_field(self, t, grid_downsample=1):
        """
        Compute the Bohmian velocity field on the grid at time t.
        
        Parameters:
        -----------
        t : float
            Time at which to compute the velocity field
        grid_downsample : int
            Downsampling factor for velocity field calculation
            
        Returns:
        --------
        vx, vy : ndarray
            Velocity field components on the downsampled grid
        """
        # First update density matrix to current time
        self.evolve_density_matrix(t)
        
        # Create downsampled grid
        N_down = self.N_grid // grid_downsample
        vx = np.zeros((N_down, N_down))
        vy = np.zeros((N_down, N_down))
        
        # Compute velocities on downsampled grid
        for i in range(N_down):
            for j in range(N_down):
                # Map to original grid
                i_orig = i * grid_downsample
                j_orig = j * grid_downsample
                
                # Get coordinates
                x = self.x[i_orig]
                y = self.y[j_orig]
                
                # Compute velocity at this point
                vx[i, j], vy[i, j] = self.compute_bohmian_velocity(x, y, t)
        
        return vx, vy
    
    def compute_W_and_gradient_at_point(self, x, y, t):
        """
        Compute the diagonal density matrix value W(x,y,x,y) and its gradients at point (x,y).
        Uses consistent bilinear interpolation for both value and gradient.
        
        Parameters:
        -----------
        x, y : float
            Coordinates at which to compute W and gradients
        t : float
            Time at which to compute W and gradients
            
        Returns:
        --------
        W_val : float
            Diagonal value of density matrix W(x,y,x,y)
        grad_x : float
            Gradient in x-direction
        grad_y : float
            Gradient in y-direction
        """
        # First update density matrix to current time if needed
        if t != self.current_time:
            self.evolve_density_matrix(t)
        
        # Check for invalid input
        if np.isnan(x) or np.isnan(y):
            print(f"Warning: NaN coordinates detected in compute_W_and_gradient_at_point: ({x}, {y})")
            return 0.0, 0.0, 0.0
        
        # Ensure coordinates are within bounds
        x = np.clip(x, 0, self.L)
        y = np.clip(y, 0, self.L)
        
        # Find indices and weights for interpolation
        i = min(max(0, int(x / self.dx)), self.N_grid - 2)
        j = min(max(0, int(y / self.dy)), self.N_grid - 2)
        
        # Calculate fractional position within cell
        fx = (x - i * self.dx) / self.dx
        fy = (y - j * self.dy) / self.dy
        
        # Bilinear interpolation weights
        w00 = (1 - fx) * (1 - fy)
        w10 = fx * (1 - fy)
        w01 = (1 - fx) * fy
        w11 = fx * fy
        
        # Interpolate W_diag
        W_val = (w00 * self.W_diag[i, j] + 
                w10 * self.W_diag[i+1, j] + 
                w01 * self.W_diag[i, j+1] + 
                w11 * self.W_diag[i+1, j+1])
        
        # Calculate gradients using bilinear interpolation of pre-computed gradients
        # Instead of finite differencing the interpolated value
        grad_x = (w00 * self.W_diag_grad_x[i, j] + 
                w10 * self.W_diag_grad_x[i+1, j] + 
                w01 * self.W_diag_grad_x[i, j+1] + 
                w11 * self.W_diag_grad_x[i+1, j+1])
        
        grad_y = (w00 * self.W_diag_grad_y[i, j] + 
                w10 * self.W_diag_grad_y[i+1, j] + 
                w01 * self.W_diag_grad_y[i, j+1] + 
                w11 * self.W_diag_grad_y[i+1, j+1])
        
        return W_val, grad_x, grad_y
    
    def compute_bohmian_velocity(self, x, y, t):
        """
        Compute the Bohmian velocity at point (x,y) at time t.
        Regularizes velocity near wave function nodes.
        
        Parameters:
        -----------
        x, y : float
            Coordinates at which to compute velocity
        t : float
            Time at which to compute velocity
            
        Returns:
        --------
        vx, vy : float
            Velocity components
        """
        # Ensure the density matrix is evolved to time t
        if t != self.current_time:
            self.evolve_density_matrix(t)
        
        # Check for invalid input
        if np.isnan(x) or np.isnan(y):
            print(f"Warning: NaN coordinates detected in compute_bohmian_velocity: ({x}, {y})")
            return 0.0, 0.0
        
        # Ensure coordinates are within bounds
        x = np.clip(x, 0, self.L)
        y = np.clip(y, 0, self.L)
        
        # Find indices and weights for interpolation
        i = min(max(0, int(x / self.dx)), self.N_grid - 2)
        j = min(max(0, int(y / self.dy)), self.N_grid - 2)
        
        # Calculate fractional position within cell
        fx = (x - i * self.dx) / self.dx
        fy = (y - j * self.dy) / self.dy
        
        # Bilinear interpolation weights
        w00 = (1 - fx) * (1 - fy)
        w10 = fx * (1 - fy)
        w01 = (1 - fx) * fy
        w11 = fx * fy
        
        # Interpolate W_diag for regularization
        W_val = (w00 * self.W_diag[i, j] + 
                w10 * self.W_diag[i+1, j] + 
                w01 * self.W_diag[i, j+1] + 
                w11 * self.W_diag[i+1, j+1])
        
        # Interpolate imaginary gradient components
        # Check for NaN values in the gradients
        if np.isnan(self.Im_grad_x_W[i, j]) or np.isnan(self.Im_grad_x_W[i+1, j]) or \
           np.isnan(self.Im_grad_x_W[i, j+1]) or np.isnan(self.Im_grad_x_W[i+1, j+1]) or \
           np.isnan(self.Im_grad_y_W[i, j]) or np.isnan(self.Im_grad_y_W[i+1, j]) or \
           np.isnan(self.Im_grad_y_W[i, j+1]) or np.isnan(self.Im_grad_y_W[i+1, j+1]):
            print(f"Warning: NaN gradients detected at ({x}, {y})")
            return 0.0, 0.0
        
        try:
            Im_grad_x = (w00 * self.Im_grad_x_W[i, j] + 
                        w10 * self.Im_grad_x_W[i+1, j] + 
                        w01 * self.Im_grad_x_W[i, j+1] + 
                        w11 * self.Im_grad_x_W[i+1, j+1])
            
            Im_grad_y = (w00 * self.Im_grad_y_W[i, j] + 
                        w10 * self.Im_grad_y_W[i+1, j] + 
                        w01 * self.Im_grad_y_W[i, j+1] + 
                        w11 * self.Im_grad_y_W[i+1, j+1])
        except Exception as e:
            print(f"Error in gradient interpolation: {e}")
            return 0.0, 0.0
        
        # Regularization constant to avoid division by very small numbers
        epsilon = 1e-10
        
        # Extract Im[∇W/W] with regularization
        if W_val > epsilon:
            # Normal calculation when W is not close to zero
            vx = Im_grad_x
            vy = Im_grad_y
            
            # Check for very large velocities and cap them
            vx_max = 100.0
            vy_max = 100.0
            vx = np.clip(vx, -vx_max, vx_max)
            vy = np.clip(vy, -vy_max, vy_max)
        else:
            # Near nodes, use a regularized calculation
            # Set velocity to zero when W is very small
            vx = 0.0
            vy = 0.0
        
        # Final NaN check
        if np.isnan(vx) or np.isnan(vy):
            print(f"Warning: NaN velocity detected at ({x}, {y})")
            return 0.0, 0.0
            
        return vx, vy
    
    def update_particles(self, t, dt, grid_downsample=1):
        """
        Update particle positions using RK4 integration.
        
        Parameters:
        -----------
        t : float
            Current time
        dt : float
            Time step
        grid_downsample : int
            Downsampling factor for velocity field calculation
        """
        # Update density matrix to current time
        self.evolve_density_matrix(t)
        
        # Update each particle
        for i in range(len(self.particles)):
            try:
                x, y = self.particles[i]
                
                # RK4 integration
                # k1
                vx1, vy1 = self.compute_bohmian_velocity(x, y, t)
                
                # k2
                vx2, vy2 = self.compute_bohmian_velocity(x + 0.5*dt*vx1, y + 0.5*dt*vy1, t + 0.5*dt)
                
                # k3
                vx3, vy3 = self.compute_bohmian_velocity(x + 0.5*dt*vx2, y + 0.5*dt*vy2, t + 0.5*dt)
                
                # k4
                vx4, vy4 = self.compute_bohmian_velocity(x + dt*vx3, y + dt*vy3, t + dt)
                
                # Combine steps with appropriate weights
                vx = (vx1 + 2*vx2 + 2*vx3 + vx4) / 6
                vy = (vy1 + 2*vy2 + 2*vy3 + vy4) / 6
                
                # Check for invalid velocities
                if np.isnan(vx) or np.isnan(vy) or np.isinf(vx) or np.isinf(vy):
                    print(f"Warning: Invalid velocity detected for particle {i}: ({vx}, {vy})")
                    continue  # Skip this particle
                
                new_x = x + vx * dt
                new_y = y + vy * dt
                
                # Improved boundary handling
                if new_x < 0:
                    # Reflect and ensure we don't go beyond 0
                    new_x = abs(new_x)
                elif new_x > self.L:
                    # Reflect and ensure we don't go beyond L
                    new_x = 2*self.L - new_x
                    if new_x < 0:  # Handle case of overshooting beyond reflection
                        new_x = 0
                
                # Similar for y-boundary
                if new_y < 0:
                    new_y = abs(new_y)
                elif new_y > self.L:
                    new_y = 2*self.L - new_y
                    if new_y < 0:
                        new_y = 0
                
                # Final check for valid position
                if np.isnan(new_x) or np.isnan(new_y):
                    print(f"Warning: Invalid position detected for particle {i}: ({new_x}, {new_y})")
                    continue  # Skip this particle
                
                self.particles[i] = [new_x, new_y]
                
            except Exception as e:
                print(f"Error updating particle {i}: {e}")
    
    def calculate_h_function(self, epsilon=None, bins=20):
        """
        Calculate the coarse-grained H-function.
        Improved implementation with proper normalization and numerical stability.
        
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
        
        # Calculate bin area for proper normalization
        bin_area = (self.L / bins) * (self.L / bins)
        
        # Normalize to get empirical probability density (divide by bin area as well as count)
        hist = hist / (np.sum(hist) * bin_area)
        
        # Resample W_diag onto the histogram grid for comparison
        W_diag_resampled = np.zeros((bins, bins))
        
        # Store the current time to ensure consistent time evolution
        current_time = self.current_time
        
        for i in range(bins):
            for j in range(bins):
                x_center = (x_edges[i] + x_edges[i+1]) / 2
                y_center = (y_edges[j] + y_edges[j+1]) / 2
                
                # Calculate W_diag at this point
                W_val, _, _ = self.compute_W_and_gradient_at_point(x_center, y_center, current_time)
                W_diag_resampled[i, j] = W_val
        
        # Normalize W_diag_resampled (proper normalization for probability density)
        # Integrate W_diag over the box should give 1
        W_diag_resampled = W_diag_resampled / (np.sum(W_diag_resampled) * bin_area)
        
        # Calculate H-function with numerical stability
        H = 0
        epsilon = 1e-10  # Small regularization constant
        
        for i in range(bins):
            for j in range(bins):
                if hist[i, j] > epsilon and W_diag_resampled[i, j] > epsilon:
                    H += hist[i, j] * np.log(hist[i, j] / W_diag_resampled[i, j]) * bin_area
        
        return H
    
    def validate_simulation(self, t):
        """
        Validate various aspects of the simulation at time t.
        """
        # 1. Check normalization of wave function or density matrix
        if self.use_pure_state and self.psi is not None:
            norm = np.sum(np.abs(self.psi)**2) * self.dx * self.dy
            print(f"Wave function norm at t={t}: {norm:.6f} (should be close to 1.0)")
        else:
            norm = np.sum(self.W_diag) * self.dx * self.dy
            print(f"Density matrix trace at t={t}: {norm:.6f} (should be close to 1.0)")
        
        # 2. Check if the H-function is decreasing
        if len(self.h_values) >= 2:
            diff = self.h_values[-1] - self.h_values[-2]
            if diff > 1e-4:  # Allow small fluctuations due to numerics
                print(f"Warning: H-function increased by {diff:.6f} at t={t}")
        
        # 3. Validate velocity field for reasonable values
        vx, vy = self.compute_velocity_field(t)
        max_v = max(np.max(np.abs(vx)), np.max(np.abs(vy)))
        if max_v > 100:  # Arbitrary threshold, adjust as needed
            print(f"Warning: Very large velocity detected: {max_v:.2f} at t={t}")
        
        # 4. Check for stray particles outside the box
        outside_x = np.logical_or(self.particles[:, 0] < 0, self.particles[:, 0] > self.L)
        outside_y = np.logical_or(self.particles[:, 1] < 0, self.particles[:, 1] > self.L)
        outside = np.sum(np.logical_or(outside_x, outside_y))
        if outside > 0:
            print(f"Warning: {outside} particle coordinates outside box at t={t}")
            # Fix particles outside the box
            self.particles[outside_x, 0] = np.clip(self.particles[outside_x, 0], 0, self.L)
            self.particles[outside_y, 1] = np.clip(self.particles[outside_y, 1], 0, self.L)
        
        # 5. Check for NaN values in velocity field
        if np.any(np.isnan(vx)) or np.any(np.isnan(vy)):
            print(f"Warning: NaN values detected in velocity field at t={t}")
    
    def visualize_density_matrix_block(self, t, save_path=None):
        """
        Visualize the density matrix with multiple off-diagonal slices
        to reveal the block structure.
        
        Parameters:
        -----------
        t : float
            Time at which to visualize the density matrix
        save_path : str or None
            If provided, save the figure to this path
        """
        # First update the density matrix to time t
        self.evolve_density_matrix(t)
        
        # Determine the number of subplots needed
        n_offdiags = len(self.offset_values)
        n_rows = 2  # One row for diagonal, one for selected off-diagonals
        n_cols = 1 + min(4, n_offdiags)  # Diagonal + up to 4 off-diags
        
        fig = plt.figure(figsize=(n_cols * 5, n_rows * 4))
        
        # First subplot: Diagonal elements
        ax1 = fig.add_subplot(n_rows, n_cols, 1)
        c1 = ax1.contourf(self.X, self.Y, self.W_diag, cmap='viridis')
        plt.colorbar(c1, ax=ax1, label='W(x,y,x,y)')
        ax1.set_title(f'Diagonal elements at t={t:.2f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        # Choose which off-diagonal slices to show (if there are many)
        offdiag_indices = np.linspace(0, n_offdiags-1, n_cols-1, dtype=int)
        
        # Plot selected off-diagonal X slices with real and imaginary parts
        for i, idx in enumerate(offdiag_indices):
            if idx < len(self.offset_values):
                offset = self.offset_values[idx]
                
                # Real part
                ax = fig.add_subplot(n_rows, n_cols, n_cols + i + 1)
                c = ax.contourf(self.X, self.Y, np.abs(self.W_offdiag_x_slices[idx]), cmap='plasma')
                plt.colorbar(c, ax=ax, label=f'|W(x+{offset:.3f},y,x,y)|')
                ax.set_title(f'Offset = {offset:.3f}')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
        
        # Title for the whole figure
        state_type = "Pure" if self.use_pure_state else "Mixed"
        plt.suptitle(f'{state_type} State Density Matrix Block Structure at t={t:.2f}', fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()
    
    def visualize_density_matrix_coherence_map(self, t, save_path=None):
        """
        Create a coherence map visualizing how the density matrix
        magnitude decays as a function of distance from the diagonal.
        
        Parameters:
        -----------
        t : float
            Time at which to visualize the coherence map
        save_path : str or None
            If provided, save the figure to this path
        """
        # First update the density matrix to time t
        self.evolve_density_matrix(t)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Compute the average magnitude of each off-diagonal slice
        x_offdiag_means = []
        y_offdiag_means = []
        
        for idx, offset in enumerate(self.offset_values):
            # Calculate mean magnitude
            x_mean = np.mean(np.abs(self.W_offdiag_x_slices[idx]))
            y_mean = np.mean(np.abs(self.W_offdiag_y_slices[idx]))
            
            x_offdiag_means.append(x_mean)
            y_offdiag_means.append(y_mean)
        
        # Plot coherence in X direction
        ax1.plot(self.offset_values, x_offdiag_means, 'bo-', linewidth=2)
        ax1.set_xlabel('Offset in X direction')
        ax1.set_ylabel('Average Magnitude |W(x+offset,y,x,y)|')
        ax1.set_title('X-direction Coherence')
        ax1.grid(True)
        
        # Plot coherence in Y direction
        ax2.plot(self.offset_values, y_offdiag_means, 'ro-', linewidth=2)
        ax2.set_xlabel('Offset in Y direction')
        ax2.set_ylabel('Average Magnitude |W(x,y+offset,x,y)|')
        ax2.set_title('Y-direction Coherence')
        ax2.grid(True)
        
        # Title for the whole figure
        state_type = "Pure" if self.use_pure_state else "Mixed"
        plt.suptitle(f'{state_type} State Density Matrix Coherence Map at t={t:.2f}', fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()
    
    def save_frame(self, t, frame_num):
        """
        Save a frame of the simulation for animation.
        
        Parameters:
        -----------
        t : float
            Time of the frame
        frame_num : int
            Frame number for ordering
        """
        # Update density matrix to current time
        self.evolve_density_matrix(t)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(18, 10))
        
        # Layout: 2 rows, 3 columns
        # Top row: Density matrix diagonal and coherence map
        # Bottom row: Particle distribution and selected off-diagonal slices
        
        # Diagonal elements
        ax1 = plt.subplot2grid((2, 3), (0, 0))
        c1 = ax1.contourf(self.X, self.Y, self.W_diag, cmap='viridis')
        plt.colorbar(c1, ax=ax1)
        ax1.set_title(f'Density Matrix at t={t:.2f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        # Coherence map
        ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
        
        # Compute the average magnitude of each off-diagonal slice
        x_offdiag_means = []
        y_offdiag_means = []
        
        for idx in range(len(self.offset_values)):
            # Calculate mean magnitude
            x_mean = np.mean(np.abs(self.W_offdiag_x_slices[idx]))
            y_mean = np.mean(np.abs(self.W_offdiag_y_slices[idx]))
            
            x_offdiag_means.append(x_mean)
            y_offdiag_means.append(y_mean)
        
        # Plot coherence in both directions
        ax2.plot(self.offset_values, x_offdiag_means, 'bo-', linewidth=2, label='X direction')
        ax2.plot(self.offset_values, y_offdiag_means, 'ro-', linewidth=2, label='Y direction')
        ax2.set_xlabel('Offset distance')
        ax2.set_ylabel('Average Magnitude |W|')
        ax2.set_title('Coherence Map')
        ax2.grid(True)
        ax2.legend()
        
        # Particle distribution
        ax3 = plt.subplot2grid((2, 3), (1, 0))
        h, x_edges, y_edges, img = ax3.hist2d(
            self.particles[:, 0], self.particles[:, 1], 
            bins=20, range=[[0, self.L], [0, self.L]], 
            cmap='plasma'
        )
        plt.colorbar(img, ax=ax3)
        
        # Add contours of W_diag to particle plot
        ax3.contour(self.X, self.Y, self.W_diag, colors='white', alpha=0.5)
        
        # Add H-function value to the plot
        H = self.calculate_h_function()
        ax3.text(0.05, 0.95, f'H = {H:.4f}', transform=ax3.transAxes, 
                bbox=dict(facecolor='white', alpha=0.5))
        
        ax3.set_title(f'Particle Distribution')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        
        # Selected off-diagonal slices (absolute value)
        # Choose two slices: one close to diagonal, one further away
        if len(self.offset_values) >= 2:
            # Choose a slice close to diagonal (small positive offset)
            close_idx = np.argmin(np.abs(np.array(self.offset_values) - 0.1))
            close_offset = self.offset_values[close_idx]
            
            # Choose a slice further from diagonal (large positive offset)
            far_idx = np.argmin(np.abs(np.array(self.offset_values) - 0.5))
            far_offset = self.offset_values[far_idx]
            
            # Near off-diagonal slice
            ax4 = plt.subplot2grid((2, 3), (1, 1))
            c4 = ax4.contourf(self.X, self.Y, np.abs(self.W_offdiag_x_slices[close_idx]), cmap='viridis')
            plt.colorbar(c4, ax=ax4)
            ax4.set_title(f'|W| at offset={close_offset:.3f}')
            ax4.set_xlabel('x')
            ax4.set_ylabel('y')
            
            # Far off-diagonal slice
            ax5 = plt.subplot2grid((2, 3), (1, 2))
            c5 = ax5.contourf(self.X, self.Y, np.abs(self.W_offdiag_x_slices[far_idx]), cmap='viridis')
            plt.colorbar(c5, ax=ax5)
            ax5.set_title(f'|W| at offset={far_offset:.3f}')
            ax5.set_xlabel('x')
            ax5.set_ylabel('y')
        
        # Add title
        state_type = "Pure" if self.use_pure_state else "Mixed"
        plt.suptitle(f'{state_type} State Bohmian Dynamics at t={t:.2f}', fontsize=16)
        plt.tight_layout()
        
        # Save the frame
        frame_path = os.path.join(self.frames_dir, f"frame_{frame_num:04d}.png")
        plt.savefig(frame_path, dpi=150)
        plt.close(fig)
        
        return H
    
    def create_animation(self, output_file=None):
        """
        Create an animation from saved frames using FFmpeg.
        
        Parameters:
        -----------
        output_file : str or None
            Output file path for the animation
        
        Note: Requires FFmpeg to be installed and available in the system path
        """
        import subprocess
        
        if output_file is None:
            state_type = "pure" if self.use_pure_state else "mixed"
            output_file = os.path.join(self.output_dir, f"{state_type}_state_animation.mp4")
        
        # Check if FFmpeg is available
        try:
            # Build the FFmpeg command
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-framerate', '10',  # Frames per second
                '-i', f"{self.frames_dir}/frame_%04d.png",
                '-c:v', 'libx264',
                '-profile:v', 'high',
                '-crf', '20',  # Quality (lower is better)
                '-pix_fmt', 'yuv420p',
                output_file
            ]
            
            # Execute FFmpeg
            subprocess.run(ffmpeg_cmd, check=True)
            print(f"Animation saved to {output_file}")
            
        except Exception as e:
            print(f"Failed to create animation: {e}")
            print("Make sure FFmpeg is installed and available in your system path.")
            print("Alternative: Use the saved frames in the 'frames' directory to create an animation manually.")
    
    def run_simulation_with_animation(self, t_max=10.0, dt=0.1, n_frames=100, grid_downsample=2):
        """
        Run the simulation and save frames for animation.
        
        Parameters:
        -----------
        t_max : float
            Maximum simulation time
        dt : float
            Time step for particle updates
        n_frames : int
            Number of frames to save for the animation
        grid_downsample : int
            Factor by which to downsample the grid for faster computation
        """
        n_steps = int(t_max / dt)
        steps_per_frame = max(1, n_steps // n_frames)
        
        self.times = []
        self.h_values = []
        
        # Compute frame times
        frame_times = np.linspace(0, t_max, n_frames)
        
        # Save initial state (frame 0)
        print(f"Initial state at t=0 (Frame 0/{n_frames})")
        self.current_time = 0
        self.validate_simulation(0)
        H = self.save_frame(0, 0)
        self.times.append(0)
        self.h_values.append(H)
        
        # Main simulation loop with evenly spaced frames
        for frame_idx in range(1, n_frames):
            t_target = frame_times[frame_idx]
            
            # Calculate how many simulation steps to reach target time
            steps_to_target = int(round((t_target - self.current_time) / dt))
            
            # Run simulation steps to reach target time
            for _ in range(steps_to_target):
                t_current = self.current_time + dt
                try:
                    self.update_particles(self.current_time, dt, grid_downsample)
                    self.current_time = t_current
                except Exception as e:
                    print(f"Error updating particles at t={t_current}: {e}")
            
            # Save frame at target time
            try:
                print(f"Processing frame {frame_idx}/{n_frames-1} at t={t_target:.2f}")
                H = self.save_frame(t_target, frame_idx)
                self.times.append(t_target)
                self.h_values.append(H)
                
                # Visualize coherence map periodically
                if frame_idx % 10 == 0:
                    self.visualize_density_matrix_coherence_map(
                        t_target,
                        save_path=f"{self.output_dir}/coherence_map_t{t_target:.1f}.png"
                    )
                
                # Validate every 10 frames
                if frame_idx % 10 == 0:
                    self.validate_simulation(t_target)
            except Exception as e:
                print(f"Error saving frame {frame_idx} at t={t_target}: {e}")
        
        # Save H-function data
        np.savetxt(f"{self.output_dir}/h_function.csv", np.column_stack((self.times, self.h_values)), 
                 delimiter=',', header='time,h_function', comments='')
        
        # Create animation
        self.create_animation()
        
        # Plot H-function over time
        plt.figure(figsize=(10, 6))
        plt.plot(self.times, self.h_values, 'b-')
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('H-function')
        plt.title('Evolution of coarse-grained H-function')
        plt.savefig(f"{self.output_dir}/h_function.png", dpi=300)
        plt.close()
        
        # Fit exponential decay to H-function
        try:
            from scipy.optimize import curve_fit
            
            def exp_decay(t, a, b, c):
                return a * np.exp(-b * t) + c
            
            params, _ = curve_fit(exp_decay, np.array(self.times), np.array(self.h_values))
            
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
            plt.close()
            
            print(f"Decay constant: {params[1]:.4f}")
        except Exception as e:
            print(f"Could not fit exponential decay to H-function data: {e}")
        
        return self.times, self.h_values

def main():
    # Create output directory for comparison plots
    comparison_dir = "comparison_results"
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    # Run pure state simulation with animation
    print("Running pure state simulation...")
    sim_pure = FullDensityMatrixSimulation(
        N_grid=10,
        N_modes=4, 
        L=np.pi, 
        N_particles=1000, 
        use_pure_state=True,
        output_dir="pure_state_results",
        num_offdiag_slices=5  # Track 5 off-diagonal slices on each side
    )
    times_pure, h_values_pure = sim_pure.run_simulation_with_animation(
        t_max=10.0, 
        dt=0.1, 
        n_frames=50
    )
    
    # Run mixed state simulation with animation
    print("Running mixed state simulation...")
    sim_mixed = FullDensityMatrixSimulation(
        N_grid=10,
        N_modes=4, 
        L=np.pi, 
        N_particles=1000, 
        use_pure_state=False,
        output_dir="mixed_state_results",
        num_offdiag_slices=5  # Track 5 off-diagonal slices on each side
    )
    times_mixed, h_values_mixed = sim_mixed.run_simulation_with_animation(
        t_max=10.0, 
        dt=0.1, 
        n_frames=50
    )
    
    # Compare H-function evolution
    plt.figure(figsize=(10, 6))
    plt.plot(times_pure, h_values_pure, 'b-', label='Pure State')
    plt.plot(times_mixed, h_values_mixed, 'r-', label='Mixed State')
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('H-function')
    plt.title('Comparison of H-function Evolution in Pure and Mixed States')
    plt.legend()
    plt.savefig(f"{comparison_dir}/h_function_comparison.png", dpi=300)
    plt.close()
    
    # Fit and compare decay rates
    try:
        from scipy.optimize import curve_fit
        
        def exp_decay(t, a, b, c):
            return a * np.exp(-b * t) + c
        
        # Fit pure state
        params_pure, _ = curve_fit(exp_decay, np.array(times_pure), np.array(h_values_pure))
        
        # Fit mixed state
        params_mixed, _ = curve_fit(exp_decay, np.array(times_mixed), np.array(h_values_mixed))
        
        # Plot comparison with fits
        plt.figure(figsize=(10, 6))
        
        # Data points
        plt.plot(times_pure, h_values_pure, 'bo', alpha=0.5, label='Pure State Data')
        plt.plot(times_mixed, h_values_mixed, 'ro', alpha=0.5, label='Mixed State Data')
        
        # Fit curves
        t_smooth = np.linspace(0, max(times_pure[-1], times_mixed[-1]), 10)
        plt.plot(t_smooth, exp_decay(t_smooth, *params_pure), 'b-', 
                label=f'Pure State Fit: τ = {1/params_pure[1]:.2f}')
        plt.plot(t_smooth, exp_decay(t_smooth, *params_mixed), 'r-', 
                label=f'Mixed State Fit: τ = {1/params_mixed[1]:.2f}')
        
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('H-function')
        plt.title('Comparison of Relaxation Rates')
        plt.legend()
        plt.savefig(f"{comparison_dir}/relaxation_rate_comparison.png", dpi=300)
        plt.close()
        
        # Print comparison
        print("\nRelaxation Rate Comparison:")
        print("-" * 50)
        print(f"Pure State Decay Rate: {params_pure[1]:.4f} (time constant: {1/params_pure[1]:.4f})")
        print(f"Mixed State Decay Rate: {params_mixed[1]:.4f} (time constant: {1/params_mixed[1]:.4f})")
        print(f"Ratio (Mixed/Pure): {params_mixed[1]/params_pure[1]:.4f}")
    except Exception as e:
        print(f"Could not compare decay rates: {e}")
    
    print("\nSimulations complete!")
    print(f"Pure state animation saved to: {sim_pure.output_dir}")
    print(f"Mixed state animation saved to: {sim_mixed.output_dir}")
    print(f"Comparison plots saved to: {comparison_dir}")

if __name__ == "__main__":
    main()