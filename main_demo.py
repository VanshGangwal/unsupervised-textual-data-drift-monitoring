"""
Main Demo Script for Drift Detection
Demonstrates the complete pipeline from the paper with real examples
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from drift_detection import UnsupervisedDriftDetection
from mitigation_strategy import DriftMitigationStrategy
from visualization import DriftVisualization
from experiment_runner import ExperimentRunner


def demo_basic_drift_detection():
    """Basic demonstration of drift detection"""
    print("=" * 60)
    print("BASIC DRIFT DETECTION DEMO")
    print("=" * 60)
    
    # Example 1: Shopping vs Technical Domain (like the paper)
    shopping_texts = [
        "I want to buy a new smartphone online",
        "Looking for the best laptop deals today", 
        "Shopping for winter clothes with free shipping",
        "Comparing prices for digital cameras",
        "Searching for home appliances on sale",
        "Best electronics store near me",
        "Online book shopping with good reviews",
        "Fashion trends for this season",
        "Furniture shopping for new apartment",
        "Sports equipment with fast delivery"
    ] * 50  # 500 samples
    
    technical_texts = [
        "Machine learning model deployment in production",
        "Deep learning optimization techniques",
        "Data science project management best practices", 
        "Cloud computing infrastructure scaling",
        "Software engineering design patterns",
        "Artificial intelligence research methodologies",
        "Statistical analysis of large datasets",
        "Computer vision algorithm development",
        "Natural language processing frameworks",
        "Big data analytics and visualization"
    ] * 45  # 450 samples
    
    # Initialize drift detector
    detector = UnsupervisedDriftDetection(
        encoder_name="bert-base-uncased",
        kernel_type='rbf',
        pooling_strategy='cls'
    )
    
    print("Detecting drift between shopping and technical domains...")
    
    # Detect drift
    results = detector.detect_drift(
        reference_texts=shopping_texts,
        target_texts=technical_texts,
        batch_size=64,
        num_bootstraps=50,
        verbose=True
    )
    
    # Print results
    print(f"\n{'='*40}")
    print("DRIFT DETECTION RESULTS")
    print(f"{'='*40}")
    print(f"Mean Drift (MMD): {results['mean_drift']:.6f}")
    print(f"Max Drift: {results['max_drift']:.6f}")
    print(f"Drift Standard Deviation: {results['drift_statistics']['std']:.6f}")
    print(f"95th Percentile: {results['drift_statistics']['percentile_95']:.6f}")
    print(f"Samples causing highest drift: {len(results['problematic_target_indices'])}")
    
    # Interpretation
    if results['mean_drift'] > 0.1:
        print("\n🚨 HIGH DRIFT DETECTED - Immediate model retraining recommended!")
    elif results['mean_drift'] > 0.05:
        print("\n⚠️  MODERATE DRIFT DETECTED - Monitor model performance closely")
    else:
        print("\n✅ LOW DRIFT - Model appears stable")
    
    # Visualize results
    viz = DriftVisualization()
    
    # Simulate performance degradation
    performance_metrics = [max(0.5, 0.95 - 2 * drift) for drift in results['drift_history']]
    
    print("\nGenerating visualizations...")
    viz.plot_drift_analysis(results, performance_metrics)
    
    # Generate detailed report
    report = viz.create_drift_report(results)
    print(f"\n{'='*60}")
    print("DETAILED REPORT")
    print(f"{'='*60}")
    print(report)
    
    return results


def demo_mitigation_strategies():
    """Demonstrate different mitigation strategies"""
    print("\n" + "=" * 60)
    print("MITIGATION STRATEGIES DEMO")
    print("=" * 60)
    
    # Training data (e-commerce domain)
    training_texts = [
        "Buy electronics online with free shipping",
        "Best deals on smartphones and tablets",
        "Shopping for clothes and accessories",
        "Home appliances with warranty",
        "Book shopping with customer reviews"
    ] * 100  # 500 samples
    
    training_labels = [0, 1, 0, 1, 1] * 100  # Binary classification
    
    # Production data (different domain - finance)
    production_texts = [
        "Investment portfolio management strategies",
        "Financial market analysis and predictions", 
        "Banking system security protocols",
        "Insurance policy optimization techniques",
        "Cryptocurrency trading algorithms"
    ] * 80  # 400 samples
    
    # Initialize components
    detector = UnsupervisedDriftDetection()
    mitigation = DriftMitigationStrategy(detector)
    viz = DriftVisualization()
    
    # Step 1: Detect drift
    print("Step 1: Detecting drift...")
    drift_results = detector.detect_drift(
        reference_texts=training_texts,
        target_texts=production_texts,
        batch_size=32,
        num_bootstraps=30,
        verbose=False
    )
    
    print(f"Detected drift: {drift_results['mean_drift']:.6f}")
    
    # Step 2: Compare mitigation strategies
    print("Step 2: Comparing mitigation strategies...")
    
    strategies = mitigation.comprehensive_mitigation_comparison(
        training_texts=training_texts,
        training_labels=training_labels,
        production_texts=production_texts,
        drift_results=drift_results
    )
    
    # Step 3: Simulate performance improvements
    print("Step 3: Simulating model performance...")
    
    baseline_accuracy = 0.75  # Baseline model accuracy
    performance_improvements = {
        'baseline': 0.0,
        'proposed_drift_detection': 0.08,  # 8% improvement
        'bias_reduction': 0.05,            # 5% improvement
        'upsampling': 0.02                 # 2% improvement
    }
    
    print(f"\n{'Strategy':<25} {'Additional':<12} {'Total':<10} {'Accuracy':<10} {'Improvement'}")
    print(f"{'Name':<25} {'Samples':<12} {'Samples':<10} {'Score':<10} {'(%)'}")
    print("-" * 75)
    
    for strategy_name, strategy_data in strategies.items():
        additional = strategy_data['additional_samples']
        total = strategy_data['total_training_samples'] 
        improvement = performance_improvements.get(strategy_name, 0)
        accuracy = baseline_accuracy + improvement
        improvement_pct = (improvement / baseline_accuracy) * 100
        
        print(f"{strategy_name:<25} {additional:<12} {total:<10} {accuracy:<10.3f} {improvement_pct:<10.1f}")
    
    # Visualize comparison
    viz.plot_mitigation_comparison(strategies)
    
    return strategies


def demo_real_time_monitoring():
    """Demonstrate real-time drift monitoring scenario"""
    print("\n" + "=" * 60)
    print("REAL-TIME MONITORING DEMO")
    print("=" * 60)
    
    # Simulate a customer service chatbot scenario
    training_texts = [
        "How can I track my order status",
        "What is your return policy",
        "I need help with my account login", 
        "Can you help me find a product",
        "I want to cancel my order"
    ] * 60  # Training on customer service queries
    
    detector = UnsupervisedDriftDetection()
    
    # Simulate incoming production batches over time
    production_batches = [
        # Week 1: Normal customer service queries
        [
            "Where is my package delivery",
            "Help me reset my password",
            "I need to return an item",
            "Product recommendation needed",
            "Cancel my subscription"
        ] * 20,
        
        # Week 2: Starting to see technical support queries
        [
            "API integration documentation needed",
            "Database connection troubleshooting", 
            "Server configuration assistance",
            "Software installation problems",
            "Network connectivity issues"
        ] * 15 + [
            "Where is my order shipment",
            "Account access problems", 
            "Return process questions"
        ] * 10,
        
        # Week 3: Mostly technical queries (high drift)
        [
            "Machine learning model deployment",
            "Cloud infrastructure scaling",
            "Data pipeline optimization",
            "API rate limiting configuration",
            "Microservices architecture design"
        ] * 20
    ]
    
    drift_over_time = []
    week_labels = ['Week 1', 'Week 2', 'Week 3']
    
    print("Monitoring drift over 3 weeks...")
    
    for week, batch in enumerate(production_batches):
        print(f"\nProcessing {week_labels[week]}...")
        
        results = detector.detect_drift(
            reference_texts=training_texts,
            target_texts=batch,
            batch_size=32,
            num_bootstraps=20,
            verbose=False
        )
        
        drift_value = results['mean_drift']
        drift_over_time.append(drift_value)
        
        print(f"  Drift detected: {drift_value:.6f}")
        
        # Alert system
        if drift_value > 0.15:
            print("  🚨 CRITICAL ALERT: High drift detected - Immediate action required!")
        elif drift_value > 0.08:
            print("  ⚠️  WARNING: Moderate drift detected - Monitor closely")
        else:
            print("  ✅ Normal operation - No action needed")
    
    # Visualize time series
    viz = DriftVisualization()
    simulated_performance = [0.92, 0.85, 0.71]  # Declining performance
    
    print("\nGenerating time series visualization...")
    viz.plot_time_series_drift(
        time_periods=week_labels,
        drift_values=drift_over_time,
        performance_values=simulated_performance
    )
    
    return drift_over_time


def demo_encoder_comparison():
    """Demonstrate encoder comparison"""
    print("\n" + "=" * 60)
    print("ENCODER COMPARISON DEMO")
    print("=" * 60)
    
    # Test data: balanced vs imbalanced
    balanced_texts = ["positive sentiment text"] * 50 + ["negative sentiment text"] * 50
    imbalanced_texts = ["positive sentiment text"] * 80 + ["negative sentiment text"] * 20
    
    encoders_to_test = ["bert-base-uncased"]
    # Add more encoders if available in your environment
    
    encoder_results = {}
    
    for encoder_name in encoders_to_test:
        print(f"Testing {encoder_name}...")
        
        try:
            detector = UnsupervisedDriftDetection(encoder_name=encoder_name)
            
            results = detector.detect_drift(
                reference_texts=balanced_texts,
                target_texts=imbalanced_texts,
                batch_size=16,
                num_bootstraps=15,
                verbose=False
            )
            
            encoder_results[encoder_name] = results['mean_drift']
            print(f"  Mean drift detected: {results['mean_drift']:.6f}")
            
        except Exception as e:
            print(f"  Error testing {encoder_name}: {str(e)}")
    
    # Visualize if multiple encoders tested
    if len(encoder_results) > 1:
        viz = DriftVisualization()
        viz.plot_encoder_comparison(encoder_results)
    
    return encoder_results


def main():
    """Main demonstration function"""
    print("🚀 UNSUPERVISED DRIFT DETECTION FOR TEXTUAL DATA")
    print("   Implementation of the research paper methodology")
    print("   'Uncovering Drift in Textual Data: An Unsupervised Method...")
    print("   ...for Detecting and Mitigating Drift in Machine Learning Models'")
    
    # Menu for different demos
    demos = {
        '1': ("Basic Drift Detection", demo_basic_drift_detection),
        '2': ("Mitigation Strategies", demo_mitigation_strategies), 
        '3': ("Real-time Monitoring", demo_real_time_monitoring),
        '4': ("Encoder Comparison", demo_encoder_comparison),
        '5': ("Run All Demos", None),
        '6': ("Paper Experiments", None)
    }
    
    print(f"\n{'='*60}")
    print("AVAILABLE DEMONSTRATIONS:")
    print(f"{'='*60}")
    
    for key, (name, _) in demos.items():
        print(f"{key}. {name}")
    
    choice = input("\nSelect demo (1-6) or 'q' to quit: ").strip()
    
    if choice.lower() == 'q':
        print("Goodbye!")
        return
    
    if choice == '5':
        # Run all demos
        print("\n🎯 Running all demonstrations...")
        demo_basic_drift_detection()
        demo_mitigation_strategies()
        demo_real_time_monitoring()
        demo_encoder_comparison()
        
    elif choice == '6':
        # Run paper experiments
        print("\n📄 Running paper experiments...")
        runner = ExperimentRunner()
        runner.run_comprehensive_evaluation()
        
    elif choice in demos and demos[choice][1] is not None:
        demos[choice][1]()
        
    else:
        print("Invalid selection!")
        return
    
    print(f"\n{'='*60}")
    print("🎉 DEMONSTRATION COMPLETE!")
    print("   Thank you for exploring drift detection!")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Check dependencies
    try:
        import torch
        import transformers
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install required packages:")
        print("pip install transformers torch scikit-learn matplotlib seaborn numpy pandas scipy")
        exit(1)
    
    # Run main demo
    main()
