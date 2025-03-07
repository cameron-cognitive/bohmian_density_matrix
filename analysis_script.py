import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from scipy.optimize import curve_fit
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
        self.pure_data = np.genfromtxt(pure_data_path, delimiter=',', names=True)
        self.mixed_data = np.genfromtxt(mixed_data_path, delimiter=',', names=True)
    
    def analyze_relaxation_rates(self):
        """
        Analyze and compare relaxation rates between pure and mixed states.
        Fits exponential decay to H-function data and compares time constants.
        """
        def exp_decay(t, a, b, c):
            return a * np.exp(-b * t) + c
        
        # Fit pure state data
        try:
            self.pure_params, _ = curve_fit(exp_decay, self.pure_data['time'], self.pure_data['h_function'])
            pure_decay_rate = self.pure_params[1]
            pure_asymptotic = self.pure_params[2]
        except:
            print("Could not fit exponential decay to pure state data")
            pure_decay_rate = np.nan
            pure_asymptotic = np.nan
        
        # Fit mixed state data
        try:
            self.mixed_params, _ = curve_fit(exp_decay, self.mixed_data['time'], self.mixed_data['h_function'])
            mixed_decay_rate = self.mixed_params[1]
            mixed_asymptotic = self.mixed_params[2]
        except:
            print("Could not fit exponential decay to mixed state data")
            mixed_decay_rate = np.nan
            mixed_asymptotic = np.nan
        
        # Compare relaxation rates
        plt.figure(figsize=(12, 8))
        
        # Plot data and fits
        plt.plot(self.pure_data['time'], self.pure_data['h_function'], 'bo', label='Pure State Data')
        plt.plot(self.mixed_data['time'], self.mixed_data['h_function'], 'ro', label='Mixed State Data')
        
        t = np.linspace(0, max(self.pure_data['time'].max(), self.mixed_data['time'].max()), 100)
        
        if not np.isnan(pure_decay_rate):
            plt.plot(t, exp_decay(t, *self.pure_params), 'b-', 
                    label=f'Pure State Fit: τ = {1/pure_decay_rate:.2f}')
        
        if not np.isnan(mixed_decay_rate):
            plt.plot(t, exp_decay(t, *self.mixed_params), 'r-', 
                    label=f'Mixed State Fit: τ = {1/mixed_decay_rate:.2f}')
        
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('H-function')
        plt.title('Comparison of Relaxation Rates')
        plt.legend()
        plt.savefig(f"{self.output_dir}/relaxation_comparison.png", dpi=300)
        plt.show()
        
        # Print comparison
        print(f"Pure State Decay Rate: {pure_decay_rate:.4f} (time constant: {1/pure_decay_rate:.4f})")
        print(f"Mixed State Decay Rate: {mixed_decay_rate:.4f} (time constant: {1/mixed_decay_rate:.4f})")
        print(f"Ratio of Decay Rates (Mixed/Pure): {mixed_decay_rate/pure_decay_rate:.4f}")
        print(f"Pure State Asymptotic H-value: {pure_asymptotic:.6f}")
        print(f"Mixed State Asymptotic H-value: {mixed_asymptotic:.6f}")
        
        return {
            'pure_decay_rate': pure_decay_rate,
            'mixed_decay_rate': mixed_decay_rate,
            'ratio': mixed_decay_rate/pure_decay_rate,
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
        t = sim_pure.times[-1]  # Use the final time
        
        # Calculate H-function for different epsilon values
        pure_h_values = []
        mixed_h_values = []
        
        for epsilon in epsilon_values:
            if epsilon is None:
                bins = 50  # Default high resolution
            else:
                bins = int(sim_pure.L / epsilon)
                
            pure_h = sim_pure.calculate_h_function(epsilon, bins)
            mixed_h = sim_mixed.calculate_h_function(epsilon, bins)
            
            pure_h_values.append(pure_h)
            mixed_h_values.append(mixed_h)
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        
        x_labels = ["None" if e is None else f"{e:.3f}" for e in epsilon_values]
        
        plt.plot(x_labels, pure_h_values, 'bo-', label='Pure State')
        plt.plot(x_labels, mixed_h_values, 'ro-', label='Mixed State')
        
        plt.grid(True)
        plt.xlabel('Coarse-graining scale (ε)')
        plt.ylabel('H-function')
        plt.title(f'Effect of Coarse-graining Scale on H-function at t={t:.2f}')
        plt.legend()
        plt.savefig(f"{self.output_dir}/coarse_graining_comparison.png", dpi=300)
        plt.show()
        
        return {
            'epsilon_values': epsilon_values,
            'pure_h_values': pure_h_values,
            'mixed_h_values': mixed_h_values
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
        
        for t in times:
            sim.evolve_density_matrix(t)
            
            # Get magnitude at reference point
            off_diag_x = np.abs(sim.W_off_diag_x_slice[ref_x_idx, ref_y_idx])
            off_diag_y = np.abs(sim.W_off_diag_y_slice[ref_x_idx, ref_y_idx])
            
            off_diag_x_mag.append(off_diag_x)
            off_diag_y_mag.append(off_diag_y)
        
        # Plot evolution
        plt.figure(figsize=(10, 6))
        
        plt.plot(times, off_diag_x_mag, 'b-', label='|W(x₀,y₀,x₁,y₀)|')
        plt.plot(times, off_diag_y_mag, 'r-', label='|W(x₀,y₀,x₀,y₁)|')
        
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('Magnitude of off-diagonal elements')
        plt.title('Evolution of Off-diagonal Elements')
        plt.legend()
        plt.savefig(f"{self.output_dir}/off_diagonal_evolution.png", dpi=300)
        plt.show()
        
        return {
            'times': times,
            'off_diag_x_mag': off_diag_x_mag,
            'off_diag_y_mag': off_diag_y_mag
        }
    
    def create_relaxation_animation(self, sim, t_max=10.0, fps=10, epsilon=None):
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
        ani.save(f"{self.output_dir}/{state_type.lower()}_relaxation.mp4", fps=fps, 
                 extra_args=['-vcodec', 'libx264'])
        
        plt.close()
        
        print(f"Animation saved to {self.output_dir}/{state_type.lower()}_relaxation.mp4")
    
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
        
        # Run simulations
        for i, sim in enumerate(simulations):
            for step, t in enumerate(times):
                sim.current_time = t
                
                # First calculate H-function at current state
                h_values[i, step] = sim.calculate_h_function()
                
                # Then update particles if not at the end
                if step < n_steps:
                    sim.update_particles(t, dt)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        for i, dist_type in enumerate(dist_types):
            plt.plot(times, h_values[i], label=f'{dist_type}')
        
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('H-function')
        plt.title('Effect of Initial Distribution on Relaxation')
        plt.legend()
        plt.savefig(f"{self.output_dir}/initial_distribution_comparison.png", dpi=300)
        plt.show()
        
        return {
            'times': times,
            'h_values': h_values,
            'dist_types': dist_types
        }

def main():
    # Create analysis object
    analyzer = DensityMatrixRelaxationAnalysis()
    
    # Option 1: Load data from previous simulations
    try:
        analyzer.load_simulation_data(
            'density_matrix_results_pure/h_function.csv',
            'density_matrix_results_mixed/h_function.csv'
        )
        analyzer.analyze_relaxation_rates()
    except:
        print("Could not load previous simulation data")
    
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
    
    for step in range(n_steps):
        t = step * dt
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
    
    # Create relaxation animation
    print("\nCreating relaxation animations...")
    analyzer.create_relaxation_animation(sim_pure, t_max=5.0, fps=10, epsilon=np.pi/20)
    analyzer.create_relaxation_animation(sim_mixed, t_max=5.0, fps=10, epsilon=np.pi/20)
    
    # Compare different initial distributions
    print("\nComparing different initial distributions...")
    analyzer.compare_initial_distributions(
        dist_types=['ground_state', 'uniform', 'custom']
    )

if __name__ == "__main__":
    main()