            vy = (vy1 + 2*vy2 + 2*vy3 + vy4) / 6
            
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
            
            self.particles[i] = [new_x, new_y]
    
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
        self.validate_simulation(0)  # Validate initial state
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
                self.validate_simulation(t)  # Validate simulation at each save point
                self.visualize_density_matrix(t, save_path=f"{self.output_dir}/density_matrix_t{t:.1f}.png")
                self.visualize_velocity_field(t, save_path=f"{self.output_dir}/velocity_field_t{t:.1f}.png")
                self.visualize_particle_distribution(t, epsilon, save_path=f"{self.output_dir}/particles_t{t:.1f}.png")
        
        # Save final state
        print(f"Final state at t={t_max}")
        self.validate_simulation(t_max)  # Validate final state
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