"""
NeuroIDS-Sat v2 Experiment Runner

Optimized for faster training with:
- Vectorized batch processing
- Configurable validation frequency
- Progress estimation
- Multiple configuration comparison

Usage:
    python run_experiments.py --data_path data/NSL-KDD/ --config balanced

Author: Toby R. Davis
"""

import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import time

from neuroids_sat import NeuroIDSSat, SatelliteConfig, compare_configurations
from data_loader import load_nslkdd, preprocess_data


def run_classification_experiment(X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray, y_test: np.ndarray,
                                   config: SatelliteConfig,
                                   epochs: int = 20,
                                   batch_size: int = 256) -> Dict:
    """Run main classification experiment."""
    
    print("\n" + "=" * 70)
    print("Classification Performance Experiment")
    print("=" * 70)
    
    model = NeuroIDSSat(config)
    
    print(f"\nModel Configuration:")
    print(f"  Hidden layers: {config.hidden_sizes}")
    print(f"  Time steps: {config.time_steps}")
    print(f"  TMR enabled: {config.tmr_enabled}")
    print(f"  Focal loss: {config.use_focal_loss}")
    print(f"  Oversample minority: {config.oversample_minority}")
    print(f"  Parameters: {model._count_parameters():,}")
    
    # Train with timing
    print(f"\nTraining on {len(X_train):,} samples (batch_size={batch_size})...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
        validate_every=1  # Can increase to 2-3 for speed
    )
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f} seconds ({train_time/epochs:.1f}s per epoch)")
    
    # Evaluate
    print(f"\nEvaluating on {len(X_test):,} test samples...")
    metrics = model.evaluate(X_test, y_test)
    
    # Print results
    print("\n" + "-" * 70)
    print("Classification Report")
    print("-" * 70)
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)
    
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"{class_name:<12} "
              f"{class_metrics['precision']:>10.3f} "
              f"{class_metrics['recall']:>10.3f} "
              f"{class_metrics['f1']:>10.3f} "
              f"{class_metrics['support']:>10}")
    
    print("-" * 70)
    print(f"{'Overall Accuracy:':<35} {metrics['accuracy']:.4f}")
    print(f"{'Avg spikes/sample:':<35} {metrics['energy']['avg_spikes_per_sample']:.1f}")
    print(f"{'Energy (pJ/sample):':<35} {metrics['energy']['energy_pj_per_sample']:.1f}")
    print(f"{'Training time:':<35} {train_time:.1f}s")
    
    return {
        'metrics': metrics,
        'history': history,
        'model_params': model._count_parameters(),
        'training_time': train_time
    }, model


def run_radiation_experiment(X_test: np.ndarray, y_test: np.ndarray,
                              model: NeuroIDSSat) -> Dict:
    """Run radiation tolerance experiment."""
    
    print("\n" + "=" * 70)
    print("Radiation Tolerance Experiment")
    print("=" * 70)
    
    seu_rates = [0, 1e-6, 1e-5, 1e-4]
    results = {'with_tmr': {}, 'without_tmr': {}}
    
    print(f"\n{'SEU Rate':<15} {'Without TMR':>15} {'With TMR':>15}")
    print("-" * 50)
    
    for seu_rate in seu_rates:
        # Test without TMR
        model.config.tmr_enabled = False
        model.encoder.config.tmr_enabled = False
        preds_no_tmr = model.predict(X_test, inject_seu=(seu_rate > 0), seu_rate=seu_rate)
        acc_no_tmr = np.mean(preds_no_tmr == y_test)
        
        # Test with TMR
        model.config.tmr_enabled = True
        model.encoder.config.tmr_enabled = True
        preds_tmr = model.predict(X_test, inject_seu=(seu_rate > 0), seu_rate=seu_rate)
        acc_tmr = np.mean(preds_tmr == y_test)
        
        results['without_tmr'][str(seu_rate)] = float(acc_no_tmr)
        results['with_tmr'][str(seu_rate)] = float(acc_tmr)
        
        rate_str = "Baseline" if seu_rate == 0 else f"{seu_rate:.0e}"
        print(f"{rate_str:<15} {acc_no_tmr:>14.1%} {acc_tmr:>14.1%}")
    
    return results


def run_energy_experiment(X_test: np.ndarray, y_test: np.ndarray,
                          model: NeuroIDSSat) -> Dict:
    """Run energy analysis experiment."""
    
    print("\n" + "=" * 70)
    print("Energy Analysis Experiment")
    print("=" * 70)
    
    _ = model.predict(X_test)
    
    avg_spikes = model.total_spikes / len(X_test)
    energy_per_sample = avg_spikes * model.config.energy_per_spike
    
    samples_per_sec = 1000
    power_mw = (energy_per_sample * samples_per_sec) / 1e9
    
    cpu_energy_mj = 45.2
    cpu_power_w = 2.3
    
    results = {
        'neuroids_sat': {
            'avg_spikes': float(avg_spikes),
            'energy_pj': float(energy_per_sample),
            'power_mw_continuous': float(power_mw),
            'power_mw_10pct_duty': float(power_mw * 0.1),
            'samples_per_sec': samples_per_sec
        },
        'conventional_cpu': {
            'energy_mj': cpu_energy_mj,
            'power_w': cpu_power_w,
            'samples_per_sec': 50
        },
        'efficiency_ratio': float(cpu_energy_mj * 1e9 / energy_per_sample)
    }
    
    print(f"\n{'Metric':<30} {'NeuroIDS-Sat':>15} {'Conv. CPU':>15}")
    print("-" * 60)
    print(f"{'Inference energy':<30} {energy_per_sample:>12.0f} pJ {cpu_energy_mj*1e6:>12.0f} pJ")
    print(f"{'Power (continuous)':<30} {power_mw:>12.2f} mW {cpu_power_w*1000:>12.0f} mW")
    print(f"{'Power (10% duty)':<30} {power_mw*0.1:>12.3f} mW {cpu_power_w*100:>12.0f} mW")
    print("-" * 60)
    print(f"{'Energy efficiency ratio:':<45} {results['efficiency_ratio']:,.0f}x")
    
    return results


def run_mission_lifetime_experiment(power_mw: float) -> Dict:
    """Calculate mission lifetime analysis."""
    
    print("\n" + "=" * 70)
    print("Mission Lifetime Analysis")
    print("=" * 70)
    
    battery_wh = 20
    battery_joules = battery_wh * 3600
    
    configurations = {
        'Conventional (continuous)': {'power_w': 2.3, 'duty_cycle': 1.0},
        'Conventional (10% duty)': {'power_w': 2.3, 'duty_cycle': 0.1},
        'NeuroIDS-Sat (continuous)': {'power_w': power_mw / 1000, 'duty_cycle': 1.0},
        'NeuroIDS-Sat (10% duty)': {'power_w': power_mw / 1000, 'duty_cycle': 0.1}
    }
    
    results = {}
    
    print(f"\n3U CubeSat with {battery_wh} Wh Battery")
    print("-" * 60)
    print(f"{'Configuration':<30} {'Power':>12} {'Runtime':>15}")
    print("-" * 60)
    
    for name, params in configurations.items():
        effective_power = params['power_w'] * params['duty_cycle']
        runtime_hours = battery_joules / (effective_power * 3600) if effective_power > 0 else float('inf')
        runtime_days = runtime_hours / 24
        
        results[name] = {
            'power_w': float(effective_power),
            'runtime_hours': float(runtime_hours),
            'runtime_days': float(runtime_days)
        }
        
        if runtime_days < 1:
            runtime_str = f"{runtime_hours:.1f} hours"
        else:
            runtime_str = f"{runtime_days:,.0f} days"
        
        if effective_power >= 1:
            power_str = f"{effective_power:.1f} W"
        elif effective_power >= 0.001:
            power_str = f"{effective_power*1000:.2f} mW"
        else:
            power_str = f"{effective_power*1000:.3f} mW"
        
        print(f"{name:<30} {power_str:>12} {runtime_str:>15}")
    
    return results


def compare_all_configs(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Compare all configuration presets."""
    
    print("\n" + "=" * 70)
    print("Configuration Comparison (Training each for 10 epochs)")
    print("=" * 70)
    
    configs = {
        'Minimal': SatelliteConfig.minimal(),
        'Balanced': SatelliteConfig.balanced(),
        'Full': SatelliteConfig.full()
    }
    
    results = {}
    
    for name, config in configs.items():
        print(f"\n--- {name} Configuration ---")
        model = NeuroIDSSat(config)
        
        start = time.time()
        model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=False)
        train_time = time.time() - start
        
        metrics = model.evaluate(X_test, y_test)
        
        results[name] = {
            'accuracy': metrics['accuracy'],
            'energy_pj': metrics['energy']['energy_pj_per_sample'],
            'training_time': train_time,
            'per_class_recall': {k: v['recall'] for k, v in metrics['per_class'].items()}
        }
        
        recall_str = " | ".join([f"{k[:3]}:{v['recall']:.2f}" for k, v in metrics['per_class'].items()])
        print(f"  Accuracy: {metrics['accuracy']:.4f} | Energy: {metrics['energy']['energy_pj_per_sample']:.0f} pJ | Time: {train_time:.1f}s")
        print(f"  Recall: [{recall_str}]")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run NeuroIDS-Sat v2 experiments')
    parser.add_argument('--data_path', type=str, default='data/NSL-KDD/',
                        help='Path to NSL-KDD dataset')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Output directory for results')
    parser.add_argument('--config', type=str, default='balanced',
                        choices=['minimal', 'balanced', 'full', 'compare'],
                        help='Configuration preset')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("NeuroIDS-Sat v2 Experiment Suite")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Configuration: {args.config}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    # Compare architectures
    compare_configurations()
    
    # Load data
    print("\n" + "=" * 70)
    print("Loading Dataset")
    print("=" * 70)
    
    try:
        X_train, y_train, X_test, y_test = load_nslkdd(args.data_path)
        X_train, X_test = preprocess_data(X_train, X_test)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nGenerating synthetic data...")
        X_train, y_train, X_test, y_test = generate_synthetic_data()
    
    print(f"Train samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Class distribution (train): {np.bincount(y_train, minlength=5)}")
    
    results = {}
    
    # Select configuration
    if args.config == 'minimal':
        config = SatelliteConfig.minimal()
    elif args.config == 'balanced':
        config = SatelliteConfig.balanced()
    elif args.config == 'full':
        config = SatelliteConfig.full()
    elif args.config == 'compare':
        results['comparison'] = compare_all_configs(X_train, y_train, X_test, y_test)
        config = SatelliteConfig.balanced()  # Use balanced for remaining tests
    
    # Run experiments
    classification_results, model = run_classification_experiment(
        X_train, y_train, X_test, y_test, config,
        epochs=args.epochs, batch_size=args.batch_size
    )
    results['classification'] = classification_results
    
    results['radiation'] = run_radiation_experiment(X_test, y_test, model)
    results['energy'] = run_energy_experiment(X_test, y_test, model)
    
    power_mw = results['energy']['neuroids_sat']['power_mw_continuous']
    results['mission_lifetime'] = run_mission_lifetime_experiment(power_mw)
    
    # Save results
    results_file = output_dir / f'neuroids_sat_v2_{args.config}_results.json'
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Save model
    model.save(str(output_dir / f'neuroids_sat_v2_{args.config}_model.pkl'))
    
    print("\n" + "=" * 70)
    print("All experiments completed!")
    print("=" * 70)


def generate_synthetic_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic NSL-KDD-like data."""
    
    n_train = 10000
    n_test = 2000
    n_features = 41
    
    class_weights = [0.53, 0.36, 0.09, 0.015, 0.005]
    
    def generate_samples(n, weights):
        X, y = [], []
        
        for c, w in enumerate(weights):
            n_c = int(n * w)
            mean = np.random.rand(n_features) * (c + 1) / 5
            std = 0.2 + 0.1 * c
            X_c = np.random.randn(n_c, n_features) * std + mean
            X.append(X_c)
            y.extend([c] * n_c)
        
        X = np.vstack(X)
        y = np.array(y)
        idx = np.random.permutation(len(y))
        return X[idx], y[idx]
    
    X_train, y_train = generate_samples(n_train, class_weights)
    X_test, y_test = generate_samples(n_test, class_weights)
    
    X_min, X_max = X_train.min(axis=0), X_train.max(axis=0)
    X_range = X_max - X_min + 1e-10
    
    X_train = np.clip((X_train - X_min) / X_range, 0, 1)
    X_test = np.clip((X_test - X_min) / X_range, 0, 1)
    
    return X_train.astype(np.float32), y_train, X_test.astype(np.float32), y_test


if __name__ == "__main__":
    main()
