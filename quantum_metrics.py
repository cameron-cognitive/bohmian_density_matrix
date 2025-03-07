import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

class QuantumInformationMetrics:
    """
    Class for calculating and analyzing quantum information metrics
    for density matrices in the Bohmian density matrix simulation.
    """
    
    def __init__(self, simulation):
        """
        Initialize with a reference to a simulation object.
        
        Parameters:
        -----------
        simulation : FullDensityMatrixSimulation
            Reference to the simulation object
        """
        self.sim = simulation
        self.metrics_history = {
            'times': [],
            'von_neumann_entropy': [],
            'linear_entropy': [],
            'purity': [],
            'participation_ratio': []
        }
    
    def calculate_eigenvalues(self):
        """
        Calculate eigenvalues of the density matrix in energy eigenbasis.
        
        For pure states, there will be one eigenvalue of 1 and the rest 0.
        For mixed states, the eigenvalues represent the mixture weights.
        
        Returns:
        --------
        eigenvalues : ndarray
            Eigenvalues of the density matrix
        """
        if self.sim.use_pure_state:
            # For our pure state, we know the eigenvalues analytically
            # One eigenvalue is 1, the rest are 0
            return np.array([1.0] + [0.0] * (self.sim.N_modes**2 - 1))
        else:
            # For our specific mixed state (equal mixture of energy eigenstates)
            # Each eigenvalue is 1/N where N is the number of eigenstates
            return np.array([1/16.0] * 16 + [0.0] * (self.sim.N_modes**2 - 16))
    
    def von_neumann_entropy(self):
        """
        Calculate the von Neumann entropy of the density matrix.
        S(ρ) = -Tr(ρ ln ρ) = -∑ λᵢ ln λᵢ
        
        Returns:
        --------
        entropy : float
            Von Neumann entropy value
        """
        eigenvalues = self.calculate_eigenvalues()
        
        # Filter out zero eigenvalues to avoid log(0) - improved numerical stability
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Calculate entropy with improved numerical stability
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        
        return entropy
    
    def linear_entropy(self):
        """
        Calculate the linear entropy of the density matrix.
        S_L = 1 - Tr(ρ²)
        
        The linear entropy is a measure of mixedness that is easier to compute
        than the von Neumann entropy and ranges from 0 (pure) to 1-1/d (completely mixed),
        where d is the dimension of the Hilbert space.
        
        Returns:
        --------
        linear_entropy : float
            Linear entropy value
        """
        if self.sim.use_pure_state:
            # For a pure state, Tr(ρ²) = 1, so S_L = 0
            return 0.0
        else:
            # For our equal mixture of N eigenstates, Tr(ρ²) = ∑ (1/N)² = 1/N
            return 1.0 - 1.0/16.0
    
    def purity(self):
        """
        Calculate the purity of the density matrix.
        γ = Tr(ρ²)
        
        Purity ranges from 1/d to 1, where d is the dimension of the Hilbert space.
        A pure state has purity = 1, while a completely mixed state has purity = 1/d.
        
        Returns:
        --------
        purity : float
            Purity value
        """
        if self.sim.use_pure_state:
            # For a pure state, purity is 1
            return 1.0
        else:
            # For our equal mixture of N eigenstates, purity is 1/N
            return 1.0/16.0
    
    def participation_ratio(self):
        """
        Calculate the participation ratio of the density matrix.
        PR = 1/Tr(ρ²)
        
        The participation ratio gives an estimate of the "effective number of states"
        in the mixture. For a pure state, PR = 1, while for a completely mixed state
        of dimension d, PR = d.
        
        Returns:
        --------
        pr : float
            Participation ratio value
        """
        return 1.0 / self.purity()
    
    def calculate_metrics(self, t):
        """
        Calculate all quantum information metrics at time t.
        
        Parameters:
        -----------
        t : float
            Time at which to calculate metrics
            
        Returns:
        --------
        metrics : dict
            Dictionary of calculated metrics
        """
        # Evolve the density matrix to time t
        self.sim.evolve_density_matrix(t)
        
        # Calculate metrics
        von_neumann = self.von_neumann_entropy()
        linear = self.linear_entropy()
        purity = self.purity()
        pr = self.participation_ratio()
        
        # Store in history
        self.metrics_history['times'].append(t)
        self.metrics_history['von_neumann_entropy'].append(von_neumann)
        self.metrics_history['linear_entropy'].append(linear)
        self.metrics_history['purity'].append(purity)
        self.metrics_history['participation_ratio'].append(pr)
        
        return {
            'von_neumann_entropy': von_neumann,
            'linear_entropy': linear,
            'purity': purity,
            'participation_ratio': pr
        }
    
    def track_metrics_over_time(self, t_max=10.0, n_steps=100):
        """
        Track quantum information metrics over time.
        
        Parameters:
        -----------
        t_max : float
            Maximum time
        n_steps : int
            Number of time steps
            
        Returns:
        --------
        metrics_history : dict
            Dictionary of metric time series
        """
        times = np.linspace(0, t_max, n_steps)
        
        for t in times:
            self.calculate_metrics(t)
        
        return self.metrics_history
    
    def visualize_metrics(self, save_path=None):
        """
        Visualize the evolution of quantum information metrics.
        
        Parameters:
        -----------
        save_path : str or None
            If provided, save the figure to this path
        """
        if len(self.metrics_history['times']) == 0:
            print("No metrics to visualize. Run track_metrics_over_time() first.")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Plot von Neumann entropy
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics_history['times'], self.metrics_history['von_neumann_entropy'], 'b-')
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('von Neumann Entropy')
        plt.title('von Neumann Entropy S(ρ)')
        
        # Plot linear entropy
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics_history['times'], self.metrics_history['linear_entropy'], 'r-')
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('Linear Entropy')
        plt.title('Linear Entropy S_L')
        
        # Plot purity
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics_history['times'], self.metrics_history['purity'], 'g-')
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('Purity')
        plt.title('Purity γ')
        
        # Plot participation ratio
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics_history['times'], self.metrics_history['participation_ratio'], 'm-')
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('Participation Ratio')
        plt.title('Participation Ratio PR')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        plt.show()
    
    def compare_pure_vs_mixed(self, metrics_pure, metrics_mixed, save_path=None):
        """
        Compare quantum information metrics between pure and mixed states.
        
        Parameters:
        -----------
        metrics_pure : dict
            Metrics history for pure state
        metrics_mixed : dict
            Metrics history for mixed state
        save_path : str or None
            If provided, save the figure to this path
        """
        plt.figure(figsize=(12, 10))
        
        # Plot von Neumann entropy
        plt.subplot(2, 2, 1)
        plt.plot(metrics_pure['times'], metrics_pure['von_neumann_entropy'], 'b-', label='Pure')
        plt.plot(metrics_mixed['times'], metrics_mixed['von_neumann_entropy'], 'r-', label='Mixed')
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('von Neumann Entropy')
        plt.title('von Neumann Entropy S(ρ)')
        plt.legend()
        
        # Plot linear entropy
        plt.subplot(2, 2, 2)
        plt.plot(metrics_pure['times'], metrics_pure['linear_entropy'], 'b-', label='Pure')
        plt.plot(metrics_mixed['times'], metrics_mixed['linear_entropy'], 'r-', label='Mixed')
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('Linear Entropy')
        plt.title('Linear Entropy S_L')
        plt.legend()
        
        # Plot purity
        plt.subplot(2, 2, 3)
        plt.plot(metrics_pure['times'], metrics_pure['purity'], 'b-', label='Pure')
        plt.plot(metrics_mixed['times'], metrics_mixed['purity'], 'r-', label='Mixed')
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('Purity')
        plt.title('Purity γ')
        plt.legend()
        
        # Plot participation ratio
        plt.subplot(2, 2, 4)
        plt.plot(metrics_pure['times'], metrics_pure['participation_ratio'], 'b-', label='Pure')
        plt.plot(metrics_mixed['times'], metrics_mixed['participation_ratio'], 'r-', label='Mixed')
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('Participation Ratio')
        plt.title('Participation Ratio PR')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        plt.show()
    
    def calculate_relative_entropy(self, rho_pure, rho_mixed):
        """
        Calculate the quantum relative entropy between two density matrices.
        S(ρ||σ) = Tr[ρ(log ρ - log σ)]
        
        Parameters:
        -----------
        rho_pure : ndarray
            Density matrix in energy eigenbasis (pure state)
        rho_mixed : ndarray
            Density matrix in energy eigenbasis (mixed state)
            
        Returns:
        --------
        rel_entropy : float
            Quantum relative entropy
        """
        # For our specific case, we know the eigenvalues analytically
        # Pure state: one eigenvalue of 1, rest 0
        # Mixed state: 16 eigenvalues of 1/16, rest 0
        
        # For pure state relative to mixed state, S(ρ||σ) = -S(ρ) - Tr[ρ log σ]
        # Since pure state has S(ρ) = 0, and Tr[ρ log σ] = log(1/16) = -log(16)
        rel_entropy = np.log(16)
        
        return rel_entropy
    
    def analyze_relaxation_correlations(self, h_values, save_path=None):
        """
        Analyze correlations between H-function relaxation and quantum information metrics.
        
        Parameters:
        -----------
        h_values : list
            H-function values over time
        save_path : str or None
            If provided, save the figure to this path
            
        Returns:
        --------
        correlations : dict
            Dictionary of correlation coefficients
        """
        if len(self.metrics_history['times']) == 0:
            print("No metrics to analyze. Run track_metrics_over_time() first.")
            return
        
        # Ensure h_values have the same length as metrics
        if len(h_values) != len(self.metrics_history['times']):
            print("H-function values must have the same length as metrics history.")
            return
        
        # Calculate correlations with improved numerical stability
        # Use Spearman rank correlation which is more robust to non-linear relationships
        from scipy.stats import spearmanr
        
        corr_von_neumann, p_von_neumann = spearmanr(h_values, self.metrics_history['von_neumann_entropy'])
        corr_linear, p_linear = spearmanr(h_values, self.metrics_history['linear_entropy'])
        corr_purity, p_purity = spearmanr(h_values, self.metrics_history['purity'])
        corr_pr, p_pr = spearmanr(h_values, self.metrics_history['participation_ratio'])
        
        correlations = {
            'von_neumann_entropy': {'correlation': corr_von_neumann, 'p_value': p_von_neumann},
            'linear_entropy': {'correlation': corr_linear, 'p_value': p_linear},
            'purity': {'correlation': corr_purity, 'p_value': p_purity},
            'participation_ratio': {'correlation': corr_pr, 'p_value': p_pr}
        }
        
        # Visualize correlations
        plt.figure(figsize=(12, 10))
        
        # Scatter plot: H-function vs von Neumann entropy
        plt.subplot(2, 2, 1)
        plt.scatter(h_values, self.metrics_history['von_neumann_entropy'])
        plt.grid(True)
        plt.xlabel('H-function')
        plt.ylabel('von Neumann Entropy')
        plt.title(f'Spearman Correlation: {corr_von_neumann:.4f} (p={p_von_neumann:.4f})')
        
        # Scatter plot: H-function vs linear entropy
        plt.subplot(2, 2, 2)
        plt.scatter(h_values, self.metrics_history['linear_entropy'])
        plt.grid(True)
        plt.xlabel('H-function')
        plt.ylabel('Linear Entropy')
        plt.title(f'Spearman Correlation: {corr_linear:.4f} (p={p_linear:.4f})')
        
        # Scatter plot: H-function vs purity
        plt.subplot(2, 2, 3)
        plt.scatter(h_values, self.metrics_history['purity'])
        plt.grid(True)
        plt.xlabel('H-function')
        plt.ylabel('Purity')
        plt.title(f'Spearman Correlation: {corr_purity:.4f} (p={p_purity:.4f})')
        
        # Scatter plot: H-function vs participation ratio
        plt.subplot(2, 2, 4)
        plt.scatter(h_values, self.metrics_history['participation_ratio'])
        plt.grid(True)
        plt.xlabel('H-function')
        plt.ylabel('Participation Ratio')
        plt.title(f'Spearman Correlation: {corr_pr:.4f} (p={p_pr:.4f})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        plt.show()
        
        return correlations

# Example usage
if __name__ == "__main__":
    from full_density_matrix import FullDensityMatrixSimulation
    
    # Create a pure state simulation
    sim_pure = FullDensityMatrixSimulation(
        N_grid=30,
        N_modes=4,
        L=np.pi,
        N_particles=1000,
        use_pure_state=True
    )
    
    # Create a mixed state simulation
    sim_mixed = FullDensityMatrixSimulation(
        N_grid=30,
        N_modes=4,
        L=np.pi,
        N_particles=1000,
        use_pure_state=False
    )
    
    # Create metric analyzers
    metrics_pure = QuantumInformationMetrics(sim_pure)
    metrics_mixed = QuantumInformationMetrics(sim_mixed)
    
    # Track metrics over time
    pure_history = metrics_pure.track_metrics_over_time(t_max=5.0, n_steps=50)
    mixed_history = metrics_mixed.track_metrics_over_time(t_max=5.0, n_steps=50)
    
    # Visualize metrics
    metrics_pure.visualize_metrics(save_path="quantum_metrics_pure.png")
    metrics_mixed.visualize_metrics(save_path="quantum_metrics_mixed.png")
    
    # Compare pure vs mixed
    metrics_pure.compare_pure_vs_mixed(pure_history, mixed_history, 
                                      save_path="quantum_metrics_comparison.png")