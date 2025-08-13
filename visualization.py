"""
Visualization Module for Drift Detection
Provides comprehensive plotting and analysis tools for drift detection results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")


class DriftVisualization:
    """
    Comprehensive visualization tools for drift detection analysis
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'seaborn-v0_8'):
        """
        Initialize visualization settings
        
        Args:
            figsize: Default figure size
            style: Matplotlib style to use
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Color schemes
        self.colors = {
            'drift': '#e74c3c',
            'no_drift': '#2ecc71', 
            'warning': '#f39c12',
            'info': '#3498db',
            'background': '#ecf0f1'
        }
    
    def plot_drift_analysis(self, results: Dict[str, Any], 
                           performance_metrics: Optional[List[float]] = None,
                           save_path: Optional[str] = None) -> None:
        """
        Create comprehensive drift analysis plots
        
        Args:
            results: Results from drift detection
            performance_metrics: Optional performance metrics over time
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Drift Detection Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Drift over time
        self._plot_drift_over_time(axes[0, 0], results)
        
        # Plot 2: Bootstrap analysis
        self._plot_bootstrap_analysis(axes[0, 1], results)
        
        # Plot 3: Drift distribution
        self._plot_drift_distribution(axes[1, 0], results)
        
        # Plot 4: Correlation with performance or drift statistics
        if performance_metrics:
            self._plot_drift_performance_correlation(axes[1, 1], results, performance_metrics)
        else:
            self._plot_drift_statistics(axes[1, 1], results)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def _plot_drift_over_time(self, ax, results: Dict[str, Any]) -> None:
        """Plot MMD drift values over time"""
        drift_history = results['drift_history']
        
        ax.plot(drift_history, color=self.colors['drift'], linewidth=2, alpha=0.8)
        ax.axhline(results['mean_drift'], color=self.colors['warning'], 
                  linestyle='--', alpha=0.7, label=f'Mean: {results["mean_drift"]:.4f}')
        
        # Highlight maximum drift point
        max_idx = results['max_drift_index']
        if max_idx < len(drift_history):
            ax.axvline(max_idx, color=self.colors['info'], linestyle=':', 
                      alpha=0.7, label=f'Max drift at: {max_idx}')
        
        ax.set_title('MMD Drift Over Time', fontweight='bold')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('MMD Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_bootstrap_analysis(self, ax, results: Dict[str, Any]) -> None:
        """Plot bootstrap median analysis"""
        bootstrap_medians = results['bootstrap_medians']
        
        ax.plot(bootstrap_medians, color=self.colors['info'], linewidth=2, alpha=0.8)
        ax.fill_between(range(len(bootstrap_medians)), bootstrap_medians, alpha=0.3, 
                       color=self.colors['info'])
        
        mean_bootstrap = np.mean(bootstrap_medians)
        ax.axhline(mean_bootstrap, color=self.colors['drift'], linestyle='--', 
                  alpha=0.7, label=f'Mean: {mean_bootstrap:.4f}')
        
        ax.set_title('Bootstrap Median MMD', fontweight='bold')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Median MMD')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_drift_distribution(self, ax, results: Dict[str, Any]) -> None:
        """Plot distribution of drift values"""
        drift_history = results['drift_history']
        
        # Histogram
        n_bins = min(30, len(drift_history) // 3)
        ax.hist(drift_history, bins=n_bins, alpha=0.7, color=self.colors['drift'], 
               density=True, edgecolor='black', linewidth=0.5)
        
        # Statistics lines
        mean_drift = results['mean_drift']
        median_drift = results.get('median_drift', np.median(drift_history))
        max_drift = results['max_drift']
        
        ax.axvline(mean_drift, color=self.colors['warning'], linestyle='--', 
                  linewidth=2, label=f'Mean: {mean_drift:.4f}')
        ax.axvline(median_drift, color=self.colors['no_drift'], linestyle='--', 
                  linewidth=2, label=f'Median: {median_drift:.4f}')
        ax.axvline(max_drift, color=self.colors['info'], linestyle=':', 
                  linewidth=2, label=f'Max: {max_drift:.4f}')
        
        ax.set_title('Distribution of MMD Values', fontweight='bold')
        ax.set_xlabel('MMD Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_drift_performance_correlation(self, ax, results: Dict[str, Any], 
                                          performance_metrics: List[float]) -> None:
        """Plot correlation between drift and performance"""
        drift_history = results['drift_history']
        
        # Align lengths
        min_len = min(len(drift_history), len(performance_metrics))
        drift_vals = drift_history[:min_len]
        perf_vals = performance_metrics[:min_len]
        
        # Scatter plot
        ax.scatter(drift_vals, perf_vals, alpha=0.6, color=self.colors['drift'])
        
        # Fit trend line
        if len(drift_vals) > 1:
            z = np.polyfit(drift_vals, perf_vals, 1)
            p = np.poly1d(z)
            ax.plot(sorted(drift_vals), p(sorted(drift_vals)), 
                   color=self.colors['info'], linestyle='--', linewidth=2)
            
            # Compute correlation
            correlation = np.corrcoef(drift_vals, perf_vals)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=ax.transAxes, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title('Drift vs Performance', fontweight='bold')
        ax.set_xlabel('MMD Value')
        ax.set_ylabel('Performance Metric')
        ax.grid(True, alpha=0.3)
    
    def _plot_drift_statistics(self, ax, results: Dict[str, Any]) -> None:
        """Plot drift statistics summary"""
        stats = results.get('drift_statistics', {})
        
        # Create bar plot of statistics
        stat_names = ['Mean', 'Median', 'Std', 'Min', 'Max', '95th %ile']
        stat_values = [
            results['mean_drift'],
            results.get('median_drift', np.median(results['drift_history'])),
            stats.get('std', 0),
            stats.get('min', 0),
            stats.get('max', 0),
            stats.get('percentile_95', 0)
        ]
        
        bars = ax.bar(stat_names, stat_values, color=self.colors['drift'], alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, stat_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Drift Statistics Summary', fontweight='bold')
        ax.set_ylabel('MMD Value')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def plot_encoder_comparison(self, encoder_results: Dict[str, float], 
                               save_path: Optional[str] = None) -> None:
        """
        Plot comparison of different encoders
        
        Args:
            encoder_results: Dictionary mapping encoder names to drift values
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        encoders = list(encoder_results.keys())
        drift_values = list(encoder_results.values())
        
        # Create bar plot
        bars = ax.bar(range(len(encoders)), drift_values, 
                     color=self.colors['drift'], alpha=0.7)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, drift_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Encoder Comparison for Drift Detection', fontsize=14, fontweight='bold')
        ax.set_xlabel('Encoder')
        ax.set_ylabel('Mean Drift (MMD)')
        ax.set_xticks(range(len(encoders)))
        ax.set_xticklabels(encoders, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_mitigation_comparison(self, mitigation_results: Dict[str, Dict], 
                                  metric_name: str = 'accuracy',
                                  save_path: Optional[str] = None) -> None:
        """
        Plot comparison of different mitigation strategies
        
        Args:
            mitigation_results: Results from different mitigation strategies
            metric_name: Name of the performance metric
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        strategies = list(mitigation_results.keys())
        
        # Plot 1: Number of additional samples
        additional_samples = [results.get('additional_samples', 0) 
                             for results in mitigation_results.values()]
        
        bars1 = ax1.bar(strategies, additional_samples, color=self.colors['info'], alpha=0.7)
        
        for bar, value in zip(bars1, additional_samples):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_title('Additional Training Samples by Strategy', fontweight='bold')
        ax1.set_ylabel('Number of Additional Samples')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Total training samples
        total_samples = [results.get('total_training_samples', 0) 
                        for results in mitigation_results.values()]
        
        bars2 = ax2.bar(strategies, total_samples, color=self.colors['no_drift'], alpha=0.7)
        
        for bar, value in zip(bars2, total_samples):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_title('Total Training Samples by Strategy', fontweight='bold')
        ax2.set_ylabel('Total Training Samples')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_time_series_drift(self, time_periods: List[str], 
                              drift_values: List[float],
                              performance_values: Optional[List[float]] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Plot drift over time periods (e.g., monthly buckets)
        
        Args:
            time_periods: List of time period labels
            drift_values: Drift values for each time period
            performance_values: Optional performance values
            save_path: Path to save the plot
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot drift values
        color1 = self.colors['drift']
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Drift (MMD)', color=color1)
        line1 = ax1.plot(time_periods, drift_values, color=color1, marker='o', 
                        linewidth=2, markersize=6, label='Drift')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Plot performance values if provided
        if performance_values:
            ax2 = ax1.twinx()
            color2 = self.colors['no_drift']
            ax2.set_ylabel('Performance', color=color2)
            line2 = ax2.plot(time_periods, performance_values, color=color2, 
                           marker='s', linewidth=2, markersize=6, label='Performance')
            ax2.tick_params(axis='y', labelcolor=color2)
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
        
        plt.title('Drift and Performance Over Time', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_drift_heatmap(self, drift_matrix: np.ndarray, 
                          x_labels: List[str], y_labels: List[str],
                          title: str = "Drift Heatmap",
                          save_path: Optional[str] = None) -> None:
        """
        Plot drift as a heatmap (e.g., comparing different time periods or datasets)
        
        Args:
            drift_matrix: 2D array of drift values
            x_labels: Labels for x-axis
            y_labels: Labels for y-axis  
            title: Plot title
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(drift_matrix, cmap='Reds', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Drift (MMD)', rotation=-90, va="bottom")
        
        # Add text annotations
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                text = ax.text(j, i, f'{drift_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_bootstrap_distribution(self, bootstrap_samples: List[float],
                                   observed_value: float,
                                   confidence_level: float = 0.95,
                                   save_path: Optional[str] = None) -> None:
        """
        Plot bootstrap distribution with confidence intervals
        
        Args:
            bootstrap_samples: Bootstrap sample values
            observed_value: Observed drift value
            confidence_level: Confidence level for intervals
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram of bootstrap samples
        ax.hist(bootstrap_samples, bins=30, alpha=0.7, color=self.colors['info'], 
               density=True, edgecolor='black', linewidth=0.5)
        
        # Confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        ci_lower = np.percentile(bootstrap_samples, lower_percentile)
        ci_upper = np.percentile(bootstrap_samples, upper_percentile)
        
        # Plot lines
        ax.axvline(observed_value, color=self.colors['drift'], linestyle='-', 
                  linewidth=2, label=f'Observed: {observed_value:.4f}')
        ax.axvline(ci_lower, color=self.colors['warning'], linestyle='--', 
                  linewidth=2, label=f'{confidence_level*100}% CI Lower: {ci_lower:.4f}')
        ax.axvline(ci_upper, color=self.colors['warning'], linestyle='--', 
                  linewidth=2, label=f'{confidence_level*100}% CI Upper: {ci_upper:.4f}')
        
        # Fill confidence interval
        ax.axvspan(ci_lower, ci_upper, alpha=0.2, color=self.colors['warning'])
        
        ax.set_title('Bootstrap Distribution of Drift Values', fontweight='bold')
        ax.set_xlabel('Drift (MMD)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_drift_report(self, results: Dict[str, Any], 
                           save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive text report of drift detection results
        
        Args:
            results: Drift detection results
            save_path: Path to save the report
            
        Returns:
            Text report string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("DRIFT DETECTION REPORT")
        report_lines.append("=" * 60)
        
        # Basic information
        report_lines.append(f"\nModel Configuration:")
        report_lines.append(f"  Encoder: {results.get('encoder_name', 'Unknown')}")
        report_lines.append(f"  Kernel: {results.get('kernel_type', 'Unknown')}")
        report_lines.append(f"  Pooling: {results.get('pooling_strategy', 'Unknown')}")
        
        # Sample information
        report_lines.append(f"\nData Information:")
        report_lines.append(f"  Reference samples: {results.get('num_reference_samples', 'Unknown')}")
        report_lines.append(f"  Target samples: {results.get('num_target_samples', 'Unknown')}")
        report_lines.append(f"  Samples processed: {results.get('num_samples_processed', 'Unknown')}")
        report_lines.append(f"  Batch size: {results.get('batch_size', 'Unknown')}")
        
        # Drift statistics
        report_lines.append(f"\nDrift Analysis:")
        report_lines.append(f"  Mean drift (MMD): {results.get('mean_drift', 0):.6f}")
        report_lines.append(f"  Median drift: {results.get('median_drift', 0):.6f}")
        report_lines.append(f"  Maximum drift: {results.get('max_drift', 0):.6f}")
        report_lines.append(f"  Max drift index: {results.get('max_drift_index', 'Unknown')}")
        
        # Additional statistics
        stats = results.get('drift_statistics', {})
        if stats:
            report_lines.append(f"\nDetailed Statistics:")
            report_lines.append(f"  Standard deviation: {stats.get('std', 0):.6f}")
            report_lines.append(f"  Minimum drift: {stats.get('min', 0):.6f}")
            report_lines.append(f"  95th percentile: {stats.get('percentile_95', 0):.6f}")
        
        # Problematic samples
        prob_ref = len(results.get('problematic_ref_indices', []))
        prob_target = len(results.get('problematic_target_indices', []))
        report_lines.append(f"\nProblematic Samples:")
        report_lines.append(f"  Reference samples flagged: {prob_ref}")
        report_lines.append(f"  Target samples flagged: {prob_target}")
        
        # Interpretation
        mean_drift = results.get('mean_drift', 0)
        report_lines.append(f"\nInterpretation:")
        if mean_drift > 0.1:
            report_lines.append("  HIGH DRIFT DETECTED - Model retraining recommended")
        elif mean_drift > 0.05:
            report_lines.append("  MODERATE DRIFT DETECTED - Monitor closely")
        else:
            report_lines.append("  LOW DRIFT - Model appears stable")
        
        report_lines.append("\n" + "=" * 60)
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


# Example usage and testing
if __name__ == "__main__":
    print("Testing DriftVisualization...")
    
    # Create sample data for testing
    np.random.seed(42)
    
    # Simulate drift detection results
    sample_results = {
        'mean_drift': 0.085,
        'median_drift': 0.078,
        'max_drift': 0.156,
        'max_drift_index': 45,
        'drift_history': np.random.exponential(0.08, 100).tolist(),
        'bootstrap_medians': np.random.exponential(0.075, 80).tolist(),
        'drift_statistics': {
            'std': 0.032,
            'min': 0.001,
            'max': 0.156,
            'percentile_95': 0.145
        },
        'problematic_ref_indices': list(range(40, 50)),
        'problematic_target_indices': list(range(40, 50)),
        'encoder_name': 'bert-base-uncased',
        'kernel_type': 'rbf',
        'pooling_strategy': 'cls',
        'num_reference_samples': 500,
        'num_target_samples': 480,
        'num_samples_processed': 100,
        'batch_size': 32
    }
    
    # Initialize visualization
    viz = DriftVisualization()
    
    # Test main drift analysis plot
    print("Creating drift analysis plot...")
    performance_metrics = [max(0, 0.9 - 0.5 * drift) for drift in sample_results['drift_history']]
    viz.plot_drift_analysis(sample_results, performance_metrics)
    
    # Test encoder comparison
    print("Creating encoder comparison plot...")
    encoder_results = {
        'bert-base-uncased': 0.085,
        'bert-large-uncased': 0.078,
        'all-MiniLM-L12-v2': 0.092
    }
    viz.plot_encoder_comparison(encoder_results)
    
    # Test time series plot
    print("Creating time series plot...")
    time_periods = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    drift_vals = [0.02, 0.035, 0.048, 0.067, 0.089, 0.112]
    perf_vals = [0.95, 0.92, 0.89, 0.85, 0.81, 0.76]
    viz.plot_time_series_drift(time_periods, drift_vals, perf_vals)
    
    # Test bootstrap distribution
    print("Creating bootstrap distribution plot...")
    bootstrap_samples = np.random.exponential(0.075, 1000)
    viz.plot_bootstrap_distribution(bootstrap_samples, 0.085)
    
    # Create text report
    print("Creating drift report...")
    report = viz.create_drift_report(sample_results)
    print("\nSample Report:")
    print(report)
    
    print("\nVisualization testing complete!")
