"""
Maximum Mean Discrepancy (MMD) Detector Module
Implements kernel-based statistical tests for distribution comparison
"""

import numpy as np
from typing import Optional
import scipy.stats as stats
from sklearn.metrics.pairwise import pairwise_kernels
import warnings
warnings.filterwarnings('ignore')


class MMDDriftDetector:
    """Maximum Mean Discrepancy based drift detection"""
    
    def __init__(self, kernel_type: str = 'rbf', gamma: Optional[float] = None, 
                 degree: int = 3, coef0: float = 1):
        """
        Initialize MMD detector
        
        Args:
            kernel_type: Type of kernel ('rbf', 'linear', 'polynomial', 'sigmoid')
            gamma: Kernel coefficient for rbf, poly and sigmoid kernels
            degree: Degree for polynomial kernel
            coef0: Independent term for polynomial and sigmoid kernels
        """
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        
        # Auto-set gamma if not provided
        if gamma is None and kernel_type in ['rbf', 'poly', 'sigmoid']:
            self.gamma = 1.0
    
    def _compute_kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix between X and Y using sklearn's implementation
        
        Args:
            X: First set of samples (m x d)
            Y: Second set of samples (n x d)
            
        Returns:
            Kernel matrix (m x n)
        """
        if self.kernel_type == 'rbf':
            return pairwise_kernels(X, Y, metric='rbf', gamma=self.gamma)
        elif self.kernel_type == 'linear':
            return pairwise_kernels(X, Y, metric='linear')
        elif self.kernel_type == 'polynomial':
            return pairwise_kernels(X, Y, metric='polynomial', 
                                  gamma=self.gamma, degree=self.degree, coef0=self.coef0)
        elif self.kernel_type == 'sigmoid':
            return pairwise_kernels(X, Y, metric='sigmoid', 
                                  gamma=self.gamma, coef0=self.coef0)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
    
    def _rbf_kernel_manual(self, X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
        """Manual implementation of RBF kernel for reference"""
        XX = np.sum(X**2, axis=1)[:, np.newaxis]
        YY = np.sum(Y**2, axis=1)[np.newaxis, :]
        XY = np.dot(X, Y.T)
        
        distances_squared = XX - 2*XY + YY
        return np.exp(-gamma * distances_squared)
    
    def compute_mmd_squared(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute squared Maximum Mean Discrepancy between two distributions
        
        Args:
            X: Reference distribution samples (m x d)
            Y: Target distribution samples (n x d)
            
        Returns:
            Squared MMD value
        """
        m, n = X.shape[0], Y.shape[0]
        
        if m == 0 or n == 0:
            return 0.0
        
        # Compute kernel matrices
        Kxx = self._compute_kernel_matrix(X, X)
        Kyy = self._compute_kernel_matrix(Y, Y)
        Kxy = self._compute_kernel_matrix(X, Y)
        
        # Compute MMD squared using unbiased estimator
        # E[k(x,x')] for x != x'
        term1 = (np.sum(Kxx) - np.trace(Kxx)) / (m * (m - 1)) if m > 1 else 0
        
        # E[k(y,y')] for y != y' 
        term2 = (np.sum(Kyy) - np.trace(Kyy)) / (n * (n - 1)) if n > 1 else 0
        
        # E[k(x,y)]
        term3 = 2 * np.sum(Kxy) / (m * n)
        
        mmd_squared = term1 + term2 - term3
        return max(0, mmd_squared)  # Ensure non-negative
    
    def compute_mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Maximum Mean Discrepancy (square root of squared MMD)
        
        Args:
            X: Reference distribution samples
            Y: Target distribution samples
            
        Returns:
            MMD value
        """
        mmd_squared = self.compute_mmd_squared(X, Y)
        return np.sqrt(mmd_squared)
    
    def compute_mmd_with_permutation_test(self, X: np.ndarray, Y: np.ndarray,
                                        n_permutations: int = 1000, 
                                        alpha: float = 0.05) -> dict:
        """
        Compute MMD with permutation test for statistical significance
        
        Args:
            X: Reference distribution samples
            Y: Target distribution samples  
            n_permutations: Number of permutations for test
            alpha: Significance level
            
        Returns:
            Dictionary with MMD value, p-value, and test result
        """
        # Compute observed MMD
        observed_mmd = self.compute_mmd_squared(X, Y)
        
        # Combine datasets
        combined = np.vstack([X, Y])
        m, n = len(X), len(Y)
        
        # Permutation test
        permutation_mmds = []
        for _ in range(n_permutations):
            # Random permutation
            perm_idx = np.random.permutation(len(combined))
            X_perm = combined[perm_idx[:m]]
            Y_perm = combined[perm_idx[m:]]
            
            # Compute MMD for permuted data
            perm_mmd = self.compute_mmd_squared(X_perm, Y_perm)
            permutation_mmds.append(perm_mmd)
        
        # Compute p-value
        p_value = np.mean(np.array(permutation_mmds) >= observed_mmd)
        
        # Test result
        is_significant = p_value < alpha
        
        return {
            'mmd': np.sqrt(observed_mmd),
            'mmd_squared': observed_mmd,
            'p_value': p_value,
            'is_significant': is_significant,
            'alpha': alpha,
            'n_permutations': n_permutations,
            'permutation_mmds': permutation_mmds
        }
    
    def compute_mmd_with_bootstrap(self, X: np.ndarray, Y: np.ndarray,
                                 n_bootstrap: int = 1000) -> dict:
        """
        Compute MMD with bootstrap confidence intervals
        
        Args:
            X: Reference distribution samples
            Y: Target distribution samples
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with MMD statistics
        """
        observed_mmd = self.compute_mmd(X, Y)
        bootstrap_mmds = []
        
        m, n = len(X), len(Y)
        
        for _ in range(n_bootstrap):
            # Bootstrap samples
            X_boot_idx = np.random.choice(m, size=m, replace=True)
            Y_boot_idx = np.random.choice(n, size=n, replace=True)
            
            X_boot = X[X_boot_idx]
            Y_boot = Y[Y_boot_idx]
            
            # Compute MMD
            boot_mmd = self.compute_mmd(X_boot, Y_boot)
            bootstrap_mmds.append(boot_mmd)
        
        bootstrap_mmds = np.array(bootstrap_mmds)
        
        return {
            'mmd': observed_mmd,
            'bootstrap_mean': np.mean(bootstrap_mmds),
            'bootstrap_std': np.std(bootstrap_mmds),
            'bootstrap_median': np.median(bootstrap_mmds),
            'confidence_interval_95': np.percentile(bootstrap_mmds, [2.5, 97.5]),
            'bootstrap_samples': bootstrap_mmds
        }
    
    def get_kernel_info(self) -> dict:
        """Get information about the kernel configuration"""
        return {
            'kernel_type': self.kernel_type,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing MMDDriftDetector...")
    
    # Generate synthetic data
    np.random.seed(42)
    
    # Reference distribution (standard normal)
    X = np.random.normal(0, 1, (100, 5))
    
    # Target distribution (shifted normal - should have drift)
    Y = np.random.normal(0.5, 1.2, (120, 5))
    
    # No drift case
    Z = np.random.normal(0, 1, (110, 5))
    
    # Test different kernels
    kernels = ['rbf', 'linear', 'polynomial']
    
    for kernel in kernels:
        print(f"\n--- Testing {kernel} kernel ---")
        detector = MMDDriftDetector(kernel_type=kernel)
        
        # Test with drift
        mmd_drift = detector.compute_mmd(X, Y)
        print(f"MMD with drift: {mmd_drift:.6f}")
        
        # Test without drift  
        mmd_no_drift = detector.compute_mmd(X, Z)
        print(f"MMD without drift: {mmd_no_drift:.6f}")
        
        # Bootstrap test
        boot_result = detector.compute_mmd_with_bootstrap(X, Y, n_bootstrap=100)
        print(f"Bootstrap mean: {boot_result['bootstrap_mean']:.6f}")
        print(f"95% CI: [{boot_result['confidence_interval_95'][0]:.6f}, {boot_result['confidence_interval_95'][1]:.6f}]")
    
    # Detailed permutation test
    print("\n--- Permutation test ---")
    detector = MMDDriftDetector(kernel_type='rbf')
    
    perm_result = detector.compute_mmd_with_permutation_test(X, Y, n_permutations=200)
    print(f"Observed MMD: {perm_result['mmd']:.6f}")
    print(f"P-value: {perm_result['p_value']:.6f}")
    print(f"Significant drift detected: {perm_result['is_significant']}")
