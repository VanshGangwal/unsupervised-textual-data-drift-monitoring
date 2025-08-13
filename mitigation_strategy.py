"""
Drift Mitigation Strategy Module
Implements strategies for mitigating model performance regression due to drift
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from drift_detection import UnsupervisedDriftDetection


class DriftMitigationStrategy:
    """
    Implements various strategies for mitigating drift-related performance regression
    """
    
    def __init__(self, drift_detector: UnsupervisedDriftDetection):
        """
        Initialize mitigation strategy with drift detector
        
        Args:
            drift_detector: Trained drift detection system
        """
        self.drift_detector = drift_detector
        self.mitigation_history = []
    
    def get_high_drift_samples(self, reference_texts: List[str], 
                              target_texts: List[str], 
                              drift_results: Dict[str, Any],
                              top_k: Optional[int] = None) -> Tuple[List[str], List[str], List[int], List[int]]:
        """
        Extract samples causing highest drift for retraining
        
        Args:
            reference_texts: Original reference texts
            target_texts: Original target texts
            drift_results: Results from drift detection
            top_k: Number of top drift samples to return (None for all)
            
        Returns:
            Tuple of (high_drift_ref_texts, high_drift_target_texts, ref_indices, target_indices)
        """
        ref_indices = drift_results['problematic_ref_indices']
        target_indices = drift_results['problematic_target_indices']
        
        # Limit to available samples
        ref_indices = [i for i in ref_indices if i < len(reference_texts)]
        target_indices = [i for i in target_indices if i < len(target_texts)]
        
        # If top_k specified, get highest drift samples
        if top_k is not None and len(ref_indices) > top_k:
            # Use drift history to identify highest drift periods
            drift_history = drift_results['drift_history']
            max_drift_idx = drift_results['max_drift_index']
            
            # Get indices around maximum drift point
            start = max(0, max_drift_idx - top_k//2)
            end = min(len(reference_texts), start + top_k)
            
            ref_indices = list(range(start, min(end, len(reference_texts))))
            target_indices = list(range(start, min(end, len(target_texts))))
        
        # Extract actual text samples
        high_drift_ref = [reference_texts[i] for i in ref_indices]
        high_drift_target = [target_texts[i] for i in target_indices]
        
        return high_drift_ref, high_drift_target, ref_indices, target_indices
    
    def create_augmented_training_data(self, original_texts: List[str], 
                                     original_labels: List[int],
                                     high_drift_texts: List[str],
                                     pseudo_labels: List[int],
                                     balance_classes: bool = True) -> Tuple[List[str], List[int]]:
        """
        Combine original training data with high-drift samples
        
        Args:
            original_texts: Original training texts
            original_labels: Original training labels
            high_drift_texts: High-drift samples to add
            pseudo_labels: Pseudo-labels for high-drift samples
            balance_classes: Whether to balance class distribution
            
        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        if len(high_drift_texts) != len(pseudo_labels):
            raise ValueError("Number of high-drift texts must match pseudo-labels")
        
        # Combine data
        augmented_texts = original_texts + high_drift_texts
        augmented_labels = original_labels + pseudo_labels
        
        # Balance classes if requested
        if balance_classes:
            augmented_texts, augmented_labels = self._balance_classes(
                augmented_texts, augmented_labels
            )
        
        return augmented_texts, augmented_labels
    
    def _balance_classes(self, texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """Balance class distribution in training data"""
        label_counts = Counter(labels)
        min_count = min(label_counts.values())
        
        balanced_texts = []
        balanced_labels = []
        
        # Group by labels
        label_groups = {}
        for text, label in zip(texts, labels):
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(text)
        
        # Sample equal number from each class
        for label, group_texts in label_groups.items():
            if len(group_texts) >= min_count:
                sampled_indices = np.random.choice(len(group_texts), min_count, replace=False)
                sampled_texts = [group_texts[i] for i in sampled_indices]
            else:
                # Oversample if needed
                sampled_indices = np.random.choice(len(group_texts), min_count, replace=True)
                sampled_texts = [group_texts[i] for i in sampled_indices]
            
            balanced_texts.extend(sampled_texts)
            balanced_labels.extend([label] * len(sampled_texts))
        
        return balanced_texts, balanced_labels
    
    def bias_reduction_mitigation(self, training_texts: List[str], 
                                training_labels: List[int],
                                production_texts: List[str],
                                n_clusters: int = 10,
                                samples_per_cluster: int = 5) -> Tuple[List[str], List[int]]:
        """
        Bias reduction method: identify absent patterns in training data
        
        Args:
            training_texts: Original training texts
            training_labels: Original training labels
            production_texts: Production texts
            n_clusters: Number of clusters to create
            samples_per_cluster: Samples to add per absent cluster
            
        Returns:
            Tuple of (additional_texts, pseudo_labels)
        """
        # Encode all texts
        training_embeddings = self.drift_detector.encoder.encode_texts(training_texts)
        production_embeddings = self.drift_detector.encoder.encode_texts(production_texts)
        
        # Cluster production data
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        production_clusters = kmeans.fit_predict(production_embeddings)
        
        # Find which clusters are present in training data
        training_clusters = kmeans.predict(training_embeddings)
        training_cluster_set = set(training_clusters)
        
        # Identify absent clusters
        absent_clusters = []
        for i in range(n_clusters):
            if i not in training_cluster_set:
                absent_clusters.append(i)
        
        # Sample from absent clusters
        additional_texts = []
        pseudo_labels = []
        
        for cluster_id in absent_clusters:
            cluster_indices = np.where(production_clusters == cluster_id)[0]
            if len(cluster_indices) > 0:
                sample_size = min(samples_per_cluster, len(cluster_indices))
                sampled_indices = np.random.choice(cluster_indices, sample_size, replace=False)
                
                for idx in sampled_indices:
                    additional_texts.append(production_texts[idx])
                    # Assign pseudo-label (could be improved with a classifier)
                    pseudo_labels.append(0)  # Default label
        
        return additional_texts, pseudo_labels
    
    def upsampling_mitigation(self, training_texts: List[str],
                            training_labels: List[int],
                            target_distribution: Dict[int, float],
                            max_samples: int = 1000) -> Tuple[List[str], List[int]]:
        """
        Upsampling method: increase frequency of underrepresented samples
        
        Args:
            training_texts: Training texts
            training_labels: Training labels
            target_distribution: Desired class distribution {label: proportion}
            max_samples: Maximum additional samples to generate
            
        Returns:
            Tuple of (additional_texts, additional_labels)
        """
        label_counts = Counter(training_labels)
        total_samples = len(training_labels)
        
        # Group texts by label
        label_groups = {}
        for text, label in zip(training_texts, training_labels):
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(text)
        
        additional_texts = []
        additional_labels = []
        samples_added = 0
        
        for label, target_prop in target_distribution.items():
            if label in label_counts:
                current_count = label_counts[label]
                target_count = int(total_samples * target_prop)
                
                if target_count > current_count and samples_added < max_samples:
                    # Upsample this class
                    deficit = min(target_count - current_count, max_samples - samples_added)
                    
                    # Sample with replacement
                    if label in label_groups:
                        sampled_texts = np.random.choice(
                            label_groups[label], deficit, replace=True
                        ).tolist()
                        
                        additional_texts.extend(sampled_texts)
                        additional_labels.extend([label] * len(sampled_texts))
                        samples_added += len(sampled_texts)
        
        return additional_texts, additional_labels
    
    def evaluate_mitigation_effectiveness(self, 
                                        original_model_predictions: List[int],
                                        mitigated_model_predictions: List[int],
                                        true_labels: List[int],
                                        test_texts: List[str]) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of mitigation strategy
        
        Args:
            original_model_predictions: Predictions from original model
            mitigated_model_predictions: Predictions from mitigated model  
            true_labels: True labels for test data
            test_texts: Test texts
            
        Returns:
            Evaluation metrics comparing original vs mitigated performance
        """
        # Calculate metrics for original model
        original_accuracy = accuracy_score(true_labels, original_model_predictions)
        
        # Calculate metrics for mitigated model
        mitigated_accuracy = accuracy_score(true_labels, mitigated_model_predictions)
        
        # Performance improvement
        accuracy_improvement = mitigated_accuracy - original_accuracy
        relative_improvement = (accuracy_improvement / original_accuracy) * 100
        
        # Detailed classification reports
        original_report = classification_report(
            true_labels, original_model_predictions, output_dict=True
        )
        mitigated_report = classification_report(
            true_labels, mitigated_model_predictions, output_dict=True
        )
        
        evaluation = {
            'original_accuracy': original_accuracy,
            'mitigated_accuracy': mitigated_accuracy,
            'accuracy_improvement': accuracy_improvement,
            'relative_improvement_percent': relative_improvement,
            'original_classification_report': original_report,
            'mitigated_classification_report': mitigated_report,
            'num_test_samples': len(true_labels)
        }
        
        return evaluation
    
    def comprehensive_mitigation_comparison(self,
                                          training_texts: List[str],
                                          training_labels: List[int],
                                          production_texts: List[str],
                                          drift_results: Dict[str, Any],
                                          pseudo_labeler=None) -> Dict[str, Any]:
        """
        Compare different mitigation strategies
        
        Args:
            training_texts: Original training texts
            training_labels: Original training labels  
            production_texts: Production texts
            drift_results: Results from drift detection
            pseudo_labeler: Function to generate pseudo-labels
            
        Returns:
            Comparison of different mitigation strategies
        """
        comparison_results = {}
        
        # Strategy 1: Proposed drift detection method
        high_drift_ref, high_drift_target, _, _ = self.get_high_drift_samples(
            training_texts, production_texts, drift_results
        )
        
        if pseudo_labeler:
            pseudo_labels = pseudo_labeler(high_drift_target)
        else:
            # Simple pseudo-labeling (random assignment)
            pseudo_labels = np.random.choice([0, 1], len(high_drift_target)).tolist()
        
        drift_augmented_texts, drift_augmented_labels = self.create_augmented_training_data(
            training_texts, training_labels, high_drift_target, pseudo_labels
        )
        
        comparison_results['proposed_drift_detection'] = {
            'additional_samples': len(high_drift_target),
            'total_training_samples': len(drift_augmented_texts),
            'texts': drift_augmented_texts,
            'labels': drift_augmented_labels
        }
        
        # Strategy 2: Bias reduction
        bias_texts, bias_labels = self.bias_reduction_mitigation(
            training_texts, training_labels, production_texts
        )
        
        bias_augmented_texts, bias_augmented_labels = self.create_augmented_training_data(
            training_texts, training_labels, bias_texts, bias_labels
        )
        
        comparison_results['bias_reduction'] = {
            'additional_samples': len(bias_texts),
            'total_training_samples': len(bias_augmented_texts),
            'texts': bias_augmented_texts,
            'labels': bias_augmented_labels
        }
        
        # Strategy 3: Upsampling
        # Assume we want balanced distribution
        unique_labels = list(set(training_labels))
        target_dist = {label: 1.0/len(unique_labels) for label in unique_labels}
        
        upsample_texts, upsample_labels = self.upsampling_mitigation(
            training_texts, training_labels, target_dist
        )
        
        upsample_augmented_texts = training_texts + upsample_texts
        upsample_augmented_labels = training_labels + upsample_labels
        
        comparison_results['upsampling'] = {
            'additional_samples': len(upsample_texts),
            'total_training_samples': len(upsample_augmented_texts),
            'texts': upsample_augmented_texts,
            'labels': upsample_augmented_labels
        }
        
        # Baseline: no mitigation
        comparison_results['baseline'] = {
            'additional_samples': 0,
            'total_training_samples': len(training_texts),
            'texts': training_texts,
            'labels': training_labels
        }
        
        return comparison_results


# Example usage and testing
if __name__ == "__main__":
    from drift_detection import UnsupervisedDriftDetection
    
    print("Testing DriftMitigationStrategy...")
    
    # Create sample data
    training_texts = ["shopping for electronics"] * 50 + ["buying clothes online"] * 50
    training_labels = [0] * 50 + [1] * 50
    
    production_texts = ["machine learning research"] * 30 + ["data science projects"] * 30
    
    # Initialize components
    detector = UnsupervisedDriftDetection()
    mitigation = DriftMitigationStrategy(detector)
    
    # Run drift detection
    print("Running drift detection...")
    drift_results = detector.detect_drift(
        training_texts, production_texts,
        batch_size=16, num_bootstraps=10, verbose=False
    )
    
    # Test mitigation strategies
    print(f"\nDetected drift: {drift_results['mean_drift']:.6f}")
    
    # Get high drift samples
    high_drift_ref, high_drift_target, ref_idx, target_idx = mitigation.get_high_drift_samples(
        training_texts, production_texts, drift_results, top_k=10
    )
    
    print(f"High drift samples - Ref: {len(high_drift_ref)}, Target: {len(high_drift_target)}")
    
    # Test augmented training data
    pseudo_labels = [0] * len(high_drift_target)  # Simple pseudo-labeling
    aug_texts, aug_labels = mitigation.create_augmented_training_data(
        training_texts, training_labels, high_drift_target, pseudo_labels
    )
    
    print(f"Augmented training data: {len(aug_texts)} samples")
    
    # Test bias reduction
    bias_texts, bias_labels = mitigation.bias_reduction_mitigation(
        training_texts, training_labels, production_texts
    )
    
    print(f"Bias reduction samples: {len(bias_texts)}")
    
    # Test comprehensive comparison
    comparison = mitigation.comprehensive_mitigation_comparison(
        training_texts, training_labels, production_texts, drift_results
    )
    
    print("\n=== Mitigation Strategy Comparison ===")
    for strategy, results in comparison.items():
        print(f"{strategy}: {results['additional_samples']} additional samples, "
              f"{results['total_training_samples']} total samples")
    
    print("\nMitigation testing complete!")
