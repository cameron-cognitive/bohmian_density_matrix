import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from full_density_matrix import FullDensityMatrixSimulation

class DensityMatrixRelaxationAnalysis:
    """
    Analysis tools for comparing relaxation dynamics in pure and mixed state
    density matrix simulations.
    """
    
    def __init__(self, output_dir="analysis_results"):
        """
        Initialize the analysis class.
        
        Parameters:
        -----------
        output_dir : str
            Directory for saving analysis results
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def load_simulation_data(self, pure_data_path, mixed_data_path):
        """
        Load H-function data from previous simulations.
        
        Parameters:
        -----------
        pure_data_path : str
            Path to H-function data for pure state
        mixed_data_path : str
            Path to H-function data for mixed state
        """
        try:
            self.pure_data = np.genfromtxt(pure_data_path, delimiter=',', names=True)
            self.mixed_data = np.genfromtxt(mixed_data_path, delimiter=',', names=True)
            print(f"Successfully loaded data from {pure_data_path} and {mixed_data_path}")
        except Exception as e:
            print(f"Error loading simulation data: {e}")
            raise
    
    def analyze_relaxation_rates(self):
        """
        Analyze and compare relaxation rates between pure and mixed states.
        Fits exponential decay to H-function data and compares time constants.
        """
        def exp_decay(t, a, b, c):
            return a * np.exp(-b * t) + c
        
        # Fit pure state data
        try:
            self.pure_params, pure_cov = curve_fit(exp_decay, self.pure_data['time'], self.pure_data['h_function'])
            pure_decay_rate = self.pure_params[1]
            pure_asymptotic = self.pure_params[2]
            
            # Calculate error estimates for pure state
            pure_perr = np.sqrt(np.diag(pure_cov))
            pure_rate_error = pure_perr[1]
            
            # Calculate R-squared for pure state
            residuals_pure = self.pure_data['h_function'] - exp_decay(self.pure_data['time'], *self.pure_params)
            ss_res_pure = np.sum(residuals_pure**2)
            ss_tot_pure = np.sum((self.pure_data['h_function'] - np.mean(self.pure_data['h_function']))**2)
            r_squared_pure = 1 - (ss_res_pure / ss_tot_pure)
        except Exception as e:
            print(f"Could not fit exponential decay to pure state data: {e}")
            pure_decay_rate = np.nan
            pure_asymptotic = np.nan
            pure_rate_error = np.nan
            r_squared_pure = np.nan
        
        # Fit mixed state data
        try:
            self.mixed_params, mixed_cov = curve_fit(exp_decay, self.mixed_data['time'], self.mixed_data['h_function'])
            mixed_decay_rate = self.mixed_params[1]
            mixed_asymptotic = self.mixed_params[2]
            
            # Calculate error estimates for mixed state
            mixed_perr = np.sqrt(np.diag(mixed_cov))
            mixed_rate_error = mixed_perr[1]
            
            # Calculate R-squared for mixed state
            residuals_mixed = self.mixed_data['h_function'] - exp_decay(self.mixed_data['time'], *self.mixed_params)
            ss_res_mixed = np.sum(residuals_mixed**2)
            ss_tot_mixed = np.sum((self.mixed_data['h_function'] - np.mean(self.mixed_data['h_function']))**2)
            r_squared_mixed = 1 - (ss_res_mixed / ss_tot_mixed)
        except Exception as e:
            print(f"Could not fit exponential decay to mixed state data: {e}")
            mixed_decay_rate = np.nan
            mixed_asymptotic = np.nan
            mixed_rate_error = np.nan
            r_squared_mixed = np.nan
        
        # Compare relaxation rates
        plt.figure(figsize=(12, 8))
        
        # Plot data and fits
        plt.plot(self.pure_data['time'], self.pure_data['h_function'], 'bo', alpha=0.7, label='Pure State Data')
        plt.plot(self.mixed_data['time'], self.mixed_data['h_function'], 'ro', alpha=0.7, label='Mixed State Data')
        
        t = np.linspace(0, max(self.pure_data['time'].max(), self.mixed_data['time'].max()), 100)
        
        if not np.isnan(pure_decay_rate):
            plt.plot(t, exp_decay(t, *self.pure_params), 'b-', linewidth=2,
                    label=f'Pure State Fit: τ = {1/pure_decay_rate:.2f} ± {pure_rate_error/pure_decay_rate**2:.2f}, R² = {r_squared_pure:.3f}')
        
        if not np.isnan(mixed_decay_rate):
            plt.plot(t, exp_decay(t, *self.mixed_params), 'r-', linewidth=2,
                    label=f'Mixed State Fit: τ = {1/mixed_decay_rate:.2f} ± {mixed_rate_error/mixed_decay_rate**2:.2f}, R² = {r_squared_mixed:.3f}')
        
        plt.grid(True)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('H-function', fontsize=12)
        plt.title('Comparison of Relaxation Rates', fontsize=14)
        plt.legend(fontsize=10)
        plt.savefig(f"{self.output_dir}/relaxation_comparison.png", dpi=300)
        plt.show()
        
        # Print comparison with error estimates
        print("\nRelaxation Rate Analysis:")
        print("-" * 50)
        if not np.isnan(pure_decay_rate):
            print(f"Pure State Decay Rate: {pure_decay_rate:.4f} ± {pure_rate_error:.4f}")
            print(f"Pure State Time Constant: {1/pure_decay_rate:.4f} ± {pure_rate_error/pure_decay_rate**2:.4f}")
            print(f"Pure State R-squared: {r_squared_pure:.4f}")
            print(f"Pure State Asymptotic H-value: {pure_asymptotic:.6f}")
        else:
            print("Pure State: Could not fit exponential decay")
        
        if not np.isnan(mixed_decay_rate):
            print(f"Mixed State Decay Rate: {mixed_decay_rate:.4f} ± {mixed_rate_error:.4f}")
            print(f"Mixed State Time Constant: {1/mixed_decay_rate:.4f} ± {mixed_rate_error/mixed_decay_rate**2:.4f}")
            print(f"Mixed State R-squared: {r_squared_mixed:.4f}")
            print(f"Mixed State Asymptotic H-value: {mixed_asymptotic:.6f}")
        else:
            print("Mixed State: Could not fit exponential decay")
        
        if not np.isnan(pure_decay_rate) and not np.isnan(mixed_decay_rate):
            ratio = mixed_decay_rate / pure_decay_rate
            # Calculate error propagation for ratio
            ratio_error = ratio * np.sqrt((mixed_rate_error/mixed_decay_rate)**2 + (pure_rate_error/pure_decay_rate)**2)
            print(f"Ratio of Decay Rates (Mixed/Pure): {ratio:.4f} ± {ratio_error:.4f}")
        
        return {
            'pure_decay_rate': pure_decay_rate,
            'pure_decay_rate_error': pure_rate_error if not np.isnan(pure_decay_rate) else None,
            'pure_r_squared': r_squared_pure,
            'mixed_decay_rate': mixed_decay_rate,
            'mixed_decay_rate_error': mixed_rate_error if not np.isnan(mixed_decay_rate) else None,
            'mixed_r_squared': r_squared_mixed,
            'ratio': mixed_decay_rate/pure_decay_rate if not (np.isnan(pure_decay_rate) or np.isnan(mixed_decay_rate)) else None,
            'pure_asymptotic': pure_asymptotic,
            'mixed_asymptotic': mixed_asymptotic
        }
    
    def compare_coarse_graining_effects(self, sim_pure, sim_mixed, 
                                        epsilon_values=[None, np.pi/10, np.pi/20, np.pi/40]):
        """
        Compare the effect of different coarse-graining scales on the H-function.
        
        Parameters:
        -----------
        sim_pure : FullDensityMatrixSimulation
            Pure state simulation object
        sim_mixed : FullDensityMatrixSimulation
            Mixed state simulation object
        epsilon_values : list
            List of coarse-graining scales to test
        """
        # Fix a specific time for comparison
        t = 5.0  # Use a reasonable time where relaxation is partially complete
        
        # Ensure density matrices are evolved to this time
        sim_pure.evolve_density_matrix(t)
        sim_mixed.evolve_density_matrix(t)
        
        # Calculate H-function for different epsilon values
        pure_h_values = []
        mixed_h_values = []
        bin_sizes = []
        
        for epsilon in epsilon_values:
            if epsilon is None:
                bins = 50  # Default high resolution
            else:
                bins = int(sim_pure.L / epsilon)
                
            bin_sizes.append(bins)
            pure_h = sim_pure.calculate_h_function(epsilon, bins)
            mixed_h = sim_mixed.calculate_h_function(epsilon, bins)
            
            pure_h_values.append(pure_h)
            mixed_h_values.append(mixed_h)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        # Create x-axis labels
        x_labels = ["Auto (50 bins)" if e is None else f"ε = {e:.3f} ({b} bins)" 
                     for e, b in zip(epsilon_values, bin_sizes)]
        
        # Plot with improved aesthetics
        plt.plot(range(len(x_labels)), pure_h_values, 'bo-', linewidth=2, markersize=8, label='Pure State')
        plt.plot(range(len(x_labels)), mixed_h_values, 'ro-', linewidth=2, markersize=8, label='Mixed State')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right')
        plt.xlabel('Coarse-graining scale (ε)', fontsize=12)
        plt.ylabel('H-function', fontsize=12)
        plt.title(f'Effect of Coarse-graining Scale on H-function at t={t:.2f}', fontsize=14)
        plt.legend(fontsize=12)
        
        # Add text showing percentage difference between finest and coarsest graining
        pure_diff_pct = (pure_h_values[-1] - pure_h_values[0]) / pure_h_values[0] * 100
        mixed_diff_pct = (mixed_h_values[-1] - mixed_h_values[0]) / mixed_h_values[0] * 100
        
        plt.annotate(f"Pure state variation: {pure_diff_pct:.1f}%", 
                     xy=(0.02, 0.95), xycoords='axes fraction', 
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.annotate(f"Mixed state variation: {mixed_diff_pct:.1f}%", 
                     xy=(0.02, 0.89), xycoords='axes fraction', 
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
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
        
        # Track magnitude and phase of off-diagonal elements
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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot magnitude
        ax1.plot(times, off_diag_x_mag, 'b-', linewidth=2, label='|W(x₀,y₀,x₁,y₀)|')
        ax1.plot(times, off_diag_y_mag, 'r-', linewidth=2, label='|W(x₀,y₀,x₀,y₁)|')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_ylabel('Magnitude', fontsize=12)
        ax1.set_title('Evolution of Off-diagonal Element Magnitude', fontsize=14)
        ax1.legend(fontsize=10)
        
        # Plot phase
        ax2.plot(times, off_diag_x_phase, 'b-', linewidth=2, label='Arg[W(x₀,y₀,x₁,y₀)]')
        ax2.plot(times, off_diag_y_phase, 'r-', linewidth=2, label='Arg[W(x₀,y₀,x₀,y₁)]')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Phase (degrees)', fontsize=12)
        ax2.set_title('Evolution of Off-diagonal Element Phase', fontsize=14)
        ax2.legend(fontsize=10)
        
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
        plt.figure(figsize=(12, 8))
        plt.plot(xf_x, amp_x, 'b-', label='X direction')
        plt.plot(xf_y, amp_y, 'r-', label='Y direction')
        
        # Mark dominant frequencies
        for freq, amp in zip(dom_freq_x, dom_amp_x):
            if freq > 0.05:  # Only mark significant frequencies
                plt.plot(freq, amp, 'bo', markersize=10)
                plt.text(freq, amp, f'{freq:.2f} Hz', verticalalignment='bottom')
        
        for freq, amp in zip(dom_freq_y, dom_amp_y):
            if freq > 0.05:  # Only mark significant frequencies
                plt.plot(freq, amp, 'ro', markersize=10)
                plt.text(freq, amp, f'{freq:.2f} Hz', verticalalignment='top')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.title('Frequency Analysis of Off-diagonal Elements', fontsize=14)
        plt.legend(fontsize=10)
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
        try:
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
            
        except Exception as e:
            print(f"Error creating animation: {e}")
            import traceback
            traceback.print_exc()
    
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
        plt.figure(figsize=(14, 10))
        
        line_styles = ['-', '--', '-.']
        colors = ['b', 'r', 'g']
        
        for i, dist_type in enumerate(dist_types):
            plt.plot(times, h_values[i], 
                    linestyle=line_styles[i % len(line_styles)],
                    color=colors[i % len(colors)],
                    linewidth=2,
                    label=f'{dist_type}')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('H-function', fontsize=12)
        plt.title('Effect of Initial Distribution on Relaxation', fontsize=14)
        plt.legend(fontsize=10)
        plt.savefig(f"{self.output_dir}/initial_distribution_comparison.png", dpi=300)
        plt.show()
        
        # Fit exponential decay to each curve and compare rates
        decay_rates = []
        asymptotic_values = []
        r_squared_values = []
        
        def exp_decay(t, a, b, c):
            return a * np.exp(-b * t) + c
        
        plt.figure(figsize=(14, 10))
        
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
            except Exception as e:
                print(f"Could not fit {dist_type} distribution: {e}")
                decay_rates.append(np.nan)
                asymptotic_values.append(np.nan)
                r_squared_values.append(np.nan)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('H-function', fontsize=12)
        plt.title('Exponential Fits for Different Initial Distributions', fontsize=14)
        plt.legend(fontsize=10)
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