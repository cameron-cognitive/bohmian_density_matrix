        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/coarse_graining_comparison.png", dpi=300)
        plt.show()
        
        return {
            'epsilon_values': epsilon_values,
            'pure_h_values': pure_h_values,
            'mixed_h_values': mixed_h_values,
            'bin_sizes': bin_sizes
        }
    
    def analyze_off_diagonal_evolution(self, sim):
        """
        Analyze the evolution of off-diagonal elements over time.
        
        Parameters:
        -----------
        sim : FullDensityMatrixSimulation
            Simulation object with stored off-diagonal samples
        """
        # Select a specific point for tracking
        ref_x_idx = sim.N_grid // 2
        ref_y_idx = sim.N_grid // 2
        
        # Calculate evolution at this point for a series of times
        times = np.linspace(0, 10, 50)
        
        # Track magnitude of off-diagonal elements
        off_diag_x_mag = []
        off_diag_y_mag = []
        off_diag_x_phase = []
        off_diag_y_phase = []
        
        for t in times:
            sim.evolve_density_matrix(t)
            
            # Get magnitude and phase at reference point
            off_diag_x = sim.W_off_diag_x_slice[ref_x_idx, ref_y_idx]
            off_diag_y = sim.W_off_diag_y_slice[ref_x_idx, ref_y_idx]
            
            off_diag_x_mag.append(np.abs(off_diag_x))
            off_diag_y_mag.append(np.abs(off_diag_y))
            off_diag_x_phase.append(np.angle(off_diag_x, deg=True))
            off_diag_y_phase.append(np.angle(off_diag_y, deg=True))
        
        # Improved plot with subplots for magnitude and phase
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot magnitude
        ax1.plot(times, off_diag_x_mag, 'b-', linewidth=2, label='|W(x₀,y₀,x₁,y₀)|')
        ax1.plot(times, off_diag_y_mag, 'r-', linewidth=2, label='|W(x₀,y₀,x₀,y₁)|')
        ax1.grid(True)
        ax1.set_ylabel('Magnitude')
        ax1.set_title('Evolution of Off-diagonal Element Magnitude')
        ax1.legend()
        
        # Plot phase
        ax2.plot(times, off_diag_x_phase, 'b-', linewidth=2, label='Arg[W(x₀,y₀,x₁,y₀)]')
        ax2.plot(times, off_diag_y_phase, 'r-', linewidth=2, label='Arg[W(x₀,y₀,x₀,y₁)]')
        ax2.grid(True)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Phase (degrees)')
        ax2.set_title('Evolution of Off-diagonal Element Phase')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/off_diagonal_evolution.png", dpi=300)
        plt.show()
        
        # Calculate power spectrum to identify frequencies
        from scipy import fft
        
        def analyze_frequencies(data, times):
            """Analyze frequency components in the data"""
            n = len(times)
            dt = times[1] - times[0]
            yf = fft.fft(data)
            xf = fft.fftfreq(n, dt)[:n//2]
            
            # Get the dominant frequencies
            amplitudes = 2.0/n * np.abs(yf[:n//2])
            dominant_idx = np.argsort(amplitudes)[-3:]  # Top 3 frequencies
            dominant_freqs = xf[dominant_idx]
            dominant_amps = amplitudes[dominant_idx]
            
            return xf, amplitudes, dominant_freqs, dominant_amps
        
        # Analyze frequency components
        xf_x, amp_x, dom_freq_x, dom_amp_x = analyze_frequencies(off_diag_x_mag, times)
        xf_y, amp_y, dom_freq_y, dom_amp_y = analyze_frequencies(off_diag_y_mag, times)
        
        # Plot power spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(xf_x, amp_x, 'b-', label='X direction')
        plt.plot(xf_y, amp_y, 'r-', label='Y direction')
        
        # Mark dominant frequencies
        for freq, amp in zip(dom_freq_x, dom_amp_x):
            plt.plot(freq, amp, 'bo', markersize=10)
            plt.text(freq, amp, f'{freq:.2f} Hz', verticalalignment='bottom')
        
        for freq, amp in zip(dom_freq_y, dom_amp_y):
            plt.plot(freq, amp, 'ro', markersize=10)
            plt.text(freq, amp, f'{freq:.2f} Hz', verticalalignment='top')
        
        plt.grid(True)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('Frequency Analysis of Off-diagonal Elements')
        plt.legend()
        plt.savefig(f"{self.output_dir}/off_diagonal_frequency.png", dpi=300)
        plt.show()
        
        return {
            'times': times,
            'off_diag_x_mag': off_diag_x_mag,
            'off_diag_y_mag': off_diag_y_mag,
            'off_diag_x_phase': off_diag_x_phase,
            'off_diag_y_phase': off_diag_y_phase,
            'dominant_frequencies_x': dom_freq_x,
            'dominant_frequencies_y': dom_freq_y
        }
    
    def create_relaxation_animation(self, sim, t_max=10.0, fps=10, epsilon=None, output_file=None):
        """
        Create an animation of the relaxation process.
        
        Parameters:
        -----------
        sim : FullDensityMatrixSimulation
            Simulation object
        t_max : float
            Maximum time for the animation
        fps : int
            Frames per second
        epsilon : float or None
            Coarse-graining scale
        output_file : str or None
            If provided, save the animation to this file, otherwise use default name
        """
        # Setup figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Initialize with first frame
        sim.evolve_density_matrix(0)
        
        # Plot density matrix diagonal
        c1 = ax1.contourf(sim.X, sim.Y, sim.W_diag, cmap='viridis')
        fig.colorbar(c1, ax=ax1, label='W(x,y,x,y)')
        ax1.set_title('Density Matrix Diagonal')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        # Plot particle distribution
        h, x_edges, y_edges, img = ax2.hist2d(
            sim.particles[:, 0], sim.particles[:, 1], 
            bins=20, range=[[0, sim.L], [0, sim.L]], 
            cmap='plasma'
        )
        fig.colorbar(img, ax=ax2, label='Particle count')
        ax2.set_title('Particle Distribution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        
        # Add text for H-function
        h_text = ax2.text(0.05, 0.95, '', transform=ax2.transAxes, 
                          bbox=dict(facecolor='white', alpha=0.5))
        
        # Add title
        state_type = "Pure" if sim.use_pure_state else "Mixed"
        title = plt.suptitle(f'{state_type} State Relaxation, t=0.00')
        
        # Animation function
        def update(frame):
            t = frame / fps * t_max / 10
            
            # Clear axes
            ax1.clear()
            ax2.clear()
            
            # Update simulation
            sim.evolve_density_matrix(t)
            sim.update_particles(t, 0.1)
            
            # Plot density matrix diagonal
            c1 = ax1.contourf(sim.X, sim.Y, sim.W_diag, cmap='viridis')
            ax1.set_title('Density Matrix Diagonal')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            
            # Plot particle distribution
            h, _, _, img = ax2.hist2d(
                sim.particles[:, 0], sim.particles[:, 1], 
                bins=20, range=[[0, sim.L], [0, sim.L]], 
                cmap='plasma'
            )
            ax2.set_title('Particle Distribution')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            
            # Add contours of W_diag to particle plot
            ax2.contour(sim.X, sim.Y, sim.W_diag, colors='white', alpha=0.5)
            
            # Update H-function
            H = sim.calculate_h_function(epsilon)
            h_text = ax2.text(0.05, 0.95, f'H = {H:.4f}', transform=ax2.transAxes, 
                              bbox=dict(facecolor='white', alpha=0.5))
            
            # Update title
            title.set_text(f'{state_type} State Relaxation, t={t:.2f}')
            
            return ax1, ax2, h_text, title
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=100, blit=False)
        
        # Save animation
        if output_file is None:
            output_file = f"{self.output_dir}/{state_type.lower()}_relaxation.mp4"
            
        ani.save(output_file, fps=fps, 
                 extra_args=['-vcodec', 'libx264'])
        
        plt.close()
        
        print(f"Animation saved to {output_file}")
    
    def compare_initial_distributions(self, L=np.pi, N_particles=1000, dist_types=['ground_state', 'uniform', 'custom']):
        """
        Compare the effect of different initial particle distributions on relaxation.
        
        Parameters:
        -----------
        L : float
            Box side length
        N_particles : int
            Number of particles
        dist_types : list
            List of distribution types to compare
        """
        # Initialize simulations with different initial distributions
        simulations = []
        
        for dist_type in dist_types:
            sim = FullDensityMatrixSimulation(
                N_grid=30, 
                N_modes=4, 
                L=L, 
                N_particles=N_particles, 
                use_pure_state=True  # Use pure state for this comparison
            )
            sim.initialize_particles(distribution_type=dist_type)
            simulations.append(sim)
        
        # Run short simulations and track H-function
        t_max = 5.0
        dt = 0.1
        n_steps = int(t_max / dt)
        
        # Storage for H-function values
        times = np.linspace(0, t_max, n_steps+1)
        h_values = np.zeros((len(dist_types), n_steps+1))
        
        # Run simulations with progress indicator
        for i, sim in enumerate(simulations):
            print(f"Running simulation for {dist_types[i]} distribution...")
            for step, t in enumerate(times):
                if step % 10 == 0:  # Print progress every 10 steps
                    print(f"  Step {step}/{n_steps}, t={t:.2f}")
                
                sim.current_time = t
                
                # First calculate H-function at current state
                h_values[i, step] = sim.calculate_h_function()
                
                # Then update particles if not at the end
                if step < n_steps:
                    sim.update_particles(t, dt)
        
        # Plot comparison with improved visualization
        plt.figure(figsize=(12, 8))
        
        line_styles = ['-', '--', '-.']
        colors = ['b', 'r', 'g']
        
        for i, dist_type in enumerate(dist_types):
            plt.plot(times, h_values[i], 
                     linestyle=line_styles[i % len(line_styles)],
                     color=colors[i % len(colors)],
                     linewidth=2,
                     label=f'{dist_type}')
        
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('H-function')
        plt.title('Effect of Initial Distribution on Relaxation')
        plt.legend()
        plt.savefig(f"{self.output_dir}/initial_distribution_comparison.png", dpi=300)
        plt.show()
        
        # Fit exponential decay to each curve and compare rates
        decay_rates = []
        asymptotic_values = []
        r_squared_values = []
        
        def exp_decay(t, a, b, c):
            return a * np.exp(-b * t) + c
        
        plt.figure(figsize=(12, 8))
        
        for i, dist_type in enumerate(dist_types):
            try:
                params, cov = curve_fit(exp_decay, times, h_values[i])
                
                # Calculate R-squared
                residuals = h_values[i] - exp_decay(times, *params)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((h_values[i] - np.mean(h_values[i]))**2)
                r_squared = 1 - (ss_res / ss_tot)
                
                decay_rates.append(params[1])
                asymptotic_values.append(params[2])
                r_squared_values.append(r_squared)
                
                # Plot data and fit
                plt.plot(times, h_values[i], 'o', 
                         color=colors[i % len(colors)], 
                         alpha=0.5,
                         label=f'{dist_type} data')
                
                plt.plot(times, exp_decay(times, *params), 
                         linestyle=line_styles[i % len(line_styles)],
                         color=colors[i % len(colors)],
                         linewidth=2,
                         label=f'{dist_type} fit: τ = {1/params[1]:.2f}, R² = {r_squared:.3f}')
            except:
                print(f"Could not fit {dist_type} distribution")
                decay_rates.append(np.nan)
                asymptotic_values.append(np.nan)
                r_squared_values.append(np.nan)
        
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('H-function')
        plt.title('Exponential Fits for Different Initial Distributions')
        plt.legend()
        plt.savefig(f"{self.output_dir}/initial_distribution_fits.png", dpi=300)
        plt.show()
        
        # Create a comparison table
        print("\nComparison of Initial Distributions:")
        print("-" * 80)
        print(f"{'Distribution':<15} {'Decay Rate':<15} {'Time Constant':<15} {'Asymptotic H':<15} {'R-squared':<15}")
        print("-" * 80)
        
        for i, dist_type in enumerate(dist_types):
            if not np.isnan(decay_rates[i]):
                print(f"{dist_type:<15} {decay_rates[i]:<15.4f} {1/decay_rates[i]:<15.4f} {asymptotic_values[i]:<15.4f} {r_squared_values[i]:<15.4f}")
            else:
                print(f"{dist_type:<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        
        return {
            'times': times,
            'h_values': h_values,
            'dist_types': dist_types,
            'decay_rates': decay_rates,
            'asymptotic_values': asymptotic_values,
            'r_squared_values': r_squared_values
        }

def main():
    # Create analysis object
    analyzer = DensityMatrixRelaxationAnalysis()
    
    # Option 1: Load data from previous simulations (if available)
    try:
        analyzer.load_simulation_data(
            'density_matrix_results_pure/h_function.csv',
            'density_matrix_results_mixed/h_function.csv'
        )
        analyzer.analyze_relaxation_rates()
        print("Successfully analyzed existing simulation data")
    except Exception as e:
        print(f"Could not load previous simulation data: {e}")
        print("Running new simulations instead...")
    
    # Option 2: Run new simulations for different coarse-graining scales
    print("\nRunning new simulations for coarse-graining analysis...")
    
    # Create simulations
    sim_pure = FullDensityMatrixSimulation(
        N_grid=30,
        N_modes=4,
        L=np.pi,
        N_particles=1000,
        use_pure_state=True
    )
    
    sim_mixed = FullDensityMatrixSimulation(
        N_grid=30,
        N_modes=4,
        L=np.pi,
        N_particles=1000,
        use_pure_state=False
    )
    
    # Run short simulations
    t_max = 5.0
    dt = 0.1
    n_steps = int(t_max / dt)
    
    print("Simulating particle dynamics...")
    for step in range(n_steps):
        t = step * dt
        if step % 10 == 0:  # Print progress every 10 steps
            print(f"  Step {step}/{n_steps}, t={t:.2f}")
        
        sim_pure.current_time = t
        sim_mixed.current_time = t
        
        sim_pure.update_particles(t, dt)
        sim_mixed.update_particles(t, dt)
    
    # Compare coarse-graining effects
    print("\nAnalyzing coarse-graining effects...")
    analyzer.compare_coarse_graining_effects(
        sim_pure, 
        sim_mixed, 
        epsilon_values=[np.pi/5, np.pi/10, np.pi/20, np.pi/40]
    )
    
    # Analyze off-diagonal evolution
    print("\nAnalyzing off-diagonal element evolution...")
    analyzer.analyze_off_diagonal_evolution(sim_pure)
    
    # Compare different initial distributions
    print("\nComparing different initial distributions...")
    analyzer.compare_initial_distributions(
        dist_types=['ground_state', 'uniform', 'custom']
    )
    
    # Create relaxation animation for both pure and mixed states
    print("\nCreating relaxation animations...")
    analyzer.create_relaxation_animation(sim_pure, t_max=5.0, fps=10)
    analyzer.create_relaxation_animation(sim_mixed, t_max=5.0, fps=10)
    
    print("\nAll analyses complete!")

if __name__ == "__main__":
    main()