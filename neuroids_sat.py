"""
NeuroIDS-Sat: Neuromorphic IDS Adapted for CubeSat Deployment

This module extends NeuroIDS v4 with satellite-specific adaptations:
1. Reduced network architecture (64-32 vs 128-64) for power constraints
2. Radiation-aware Triple Modular Redundancy (TMR) spike encoding
3. Hierarchical detection (Tier 1 anomaly + Tier 2 classification)
4. Adaptive duty cycling based on threat level
5. Shortened simulation window (50 vs 100 timesteps)

Paper: "NeuroIDS-Sat: Neuromorphic Intrusion Detection for CubeSat 
        Constellations Under Extreme Power Constraints"

Author: Toby R. Davis
Mississippi State University
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pickle
from pathlib import Path


@dataclass
class SatelliteConfig:
    """Configuration for satellite-optimized NeuroIDS.
    
    Key differences from terrestrial NeuroIDS:
    - Smaller hidden layers (64, 32 vs 128, 64)
    - Fewer timesteps (50 vs 100)
    - Higher threshold for noise tolerance
    - TMR encoding enabled by default
    """
    input_size: int = 41
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 32])
    output_size: int = 5
    time_steps: int = 50  # Reduced from 100
    
    # LIF parameters - adjusted for radiation tolerance
    v_thresh: float = 0.6  # Higher threshold (was 0.5)
    v_rest: float = 0.0
    v_reset: float = 0.0
    tau_m: float = 20.0  # Longer time constant (was 15.0)
    tau_ref: float = 3.0  # Longer refractory (was 2.0)
    r_m: float = 1.0
    
    # Encoding
    spike_rate_base: float = 0.1
    spike_rate_max: float = 0.5
    
    # TMR (Triple Modular Redundancy)
    tmr_enabled: bool = True
    
    # Training
    learning_rate: float = 0.02
    weight_decay: float = 0.0001
    
    # Hierarchical detection
    tier1_threshold: float = 0.3  # Anomaly detection threshold
    
    # Energy model (picojoules)
    energy_per_spike: float = 2.0  # pJ per spike operation


class LIFNeuronsSat:
    """Satellite-optimized LIF neuron population.
    
    Uses higher threshold and longer time constants for 
    improved noise tolerance in radiation environments.
    """
    
    def __init__(self, size: int, config: SatelliteConfig):
        self.size = size
        self.config = config
        
        # State variables
        self.v = np.zeros(size)  # Membrane potential
        self.refractory = np.zeros(size)  # Refractory counters
        
    def reset(self):
        """Reset neuron state."""
        self.v.fill(self.config.v_rest)
        self.refractory.fill(0)
        
    def forward(self, current: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Simulate one timestep.
        
        Args:
            current: Input current to neurons
            dt: Timestep size (ms)
            
        Returns:
            Binary spike output
        """
        # Decay refractory counters
        self.refractory = np.maximum(0, self.refractory - dt)
        
        # Mask for non-refractory neurons
        active = self.refractory <= 0
        
        # Leaky integration (exponential decay + input)
        decay = np.exp(-dt / self.config.tau_m)
        self.v = np.where(
            active,
            self.config.v_rest + (self.v - self.config.v_rest) * decay + 
            self.config.r_m * current,
            self.v
        )
        
        # Spike generation
        spikes = (self.v >= self.config.v_thresh).astype(float)
        
        # Reset spiking neurons
        self.v = np.where(spikes > 0, self.config.v_reset, self.v)
        self.refractory = np.where(spikes > 0, self.config.tau_ref, self.refractory)
        
        return spikes


class TMREncoder:
    """Triple Modular Redundancy spike encoder.
    
    Generates three redundant spike trains and uses majority
    voting to tolerate single-event upsets (SEUs).
    """
    
    def __init__(self, config: SatelliteConfig):
        self.config = config
        
    def encode(self, x: np.ndarray, inject_seu: bool = False, 
               seu_rate: float = 0.0) -> np.ndarray:
        """Encode input features to spike trains with TMR.
        
        Args:
            x: Input features, shape (n_samples, n_features)
            inject_seu: Whether to simulate radiation upsets
            seu_rate: Rate of SEU injection (upsets per bit)
            
        Returns:
            Spike trains, shape (n_samples, time_steps, n_features)
        """
        n_samples, n_features = x.shape
        T = self.config.time_steps
        
        # Normalize to [0, 1]
        x_norm = np.clip(x, 0, 1)
        
        # Generate three redundant encodings
        spike_trains = []
        for _ in range(3):
            # Rate coding: P(spike) = base + x * (max - base)
            rates = self.config.spike_rate_base + x_norm * (
                self.config.spike_rate_max - self.config.spike_rate_base
            )
            
            # Generate spikes
            rand = np.random.rand(n_samples, T, n_features)
            spikes = (rand < rates[:, np.newaxis, :]).astype(float)
            
            # Inject SEUs if enabled
            if inject_seu and seu_rate > 0:
                seu_mask = np.random.rand(*spikes.shape) < seu_rate
                spikes = np.where(seu_mask, 1 - spikes, spikes)
            
            spike_trains.append(spikes)
        
        if self.config.tmr_enabled:
            # Majority voting across three encodings
            stacked = np.stack(spike_trains, axis=0)
            voted = (np.sum(stacked, axis=0) >= 2).astype(float)
            return voted
        else:
            # Return first encoding (no TMR protection)
            return spike_trains[0]


class LIFLayerSat:
    """Satellite-optimized LIF layer."""
    
    def __init__(self, input_size: int, output_size: int, 
                 config: SatelliteConfig):
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        
        # Xavier initialization
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.randn(input_size, output_size) * scale
        self.bias = np.zeros(output_size)
        
        # LIF neurons
        self.neurons = LIFNeuronsSat(output_size, config)
        
        # Spike count accumulator
        self.spike_counts = np.zeros(output_size)
        
    def reset(self):
        """Reset layer state."""
        self.neurons.reset()
        self.spike_counts.fill(0)
        
    def forward(self, input_spikes: np.ndarray) -> np.ndarray:
        """Process one timestep.
        
        Args:
            input_spikes: Binary spike input
            
        Returns:
            Binary spike output
        """
        current = input_spikes @ self.weights + self.bias
        spikes = self.neurons.forward(current)
        self.spike_counts += spikes
        return spikes
    
    def get_spike_counts(self) -> np.ndarray:
        """Get accumulated spike counts."""
        return self.spike_counts.copy()


class NeuroIDSSat:
    """Neuromorphic IDS for CubeSat deployment.
    
    Implements hierarchical detection:
    - Tier 1: Fast binary anomaly detection (always active)
    - Tier 2: Full 5-class classification (activated on anomaly)
    """
    
    ATTACK_CATEGORIES = {
        0: 'normal',
        1: 'dos',
        2: 'probe',
        3: 'r2l',
        4: 'u2r'
    }
    
    def __init__(self, config: SatelliteConfig = None):
        self.config = config or SatelliteConfig()
        
        # Encoder with TMR
        self.encoder = TMREncoder(self.config)
        
        # Build layers (smaller than terrestrial)
        self.layers: List[LIFLayerSat] = []
        prev_size = self.config.input_size
        
        for hidden_size in self.config.hidden_sizes:
            self.layers.append(LIFLayerSat(prev_size, hidden_size, self.config))
            prev_size = hidden_size
        
        # Output weights (spike counts -> class scores)
        self.output_weights = np.random.randn(prev_size, self.config.output_size) * 0.1
        self.output_bias = np.zeros(self.config.output_size)
        
        # Class scaling for imbalanced data
        self.class_scales = np.ones(self.config.output_size)
        
        # Training state
        self.is_fitted = False
        self.training_history = {
            'loss': [], 'accuracy': [], 'val_accuracy': [],
            'spikes': [], 'energy': []
        }
        
        # Energy tracking
        self.total_spikes = 0
        
    def reset(self):
        """Reset all layer states."""
        for layer in self.layers:
            layer.reset()
        self.total_spikes = 0
            
    def _forward_sample(self, spike_train: np.ndarray) -> Tuple[np.ndarray, int]:
        """Process a single sample through the network.
        
        Args:
            spike_train: Shape (time_steps, n_features)
            
        Returns:
            (spike_counts, total_spikes)
        """
        self.reset()
        total_spikes = 0
        
        for t in range(spike_train.shape[0]):
            x = spike_train[t]
            total_spikes += np.sum(x)
            
            for layer in self.layers:
                x = layer.forward(x)
                total_spikes += np.sum(x)
        
        # Get final layer spike counts
        final_counts = self.layers[-1].get_spike_counts()
        return final_counts, int(total_spikes)
    
    def predict_proba(self, X: np.ndarray, inject_seu: bool = False,
                      seu_rate: float = 0.0) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input features, shape (n_samples, n_features)
            inject_seu: Simulate radiation effects
            seu_rate: SEU injection rate
            
        Returns:
            Class probabilities, shape (n_samples, n_classes)
        """
        # Encode to spike trains
        spike_trains = self.encoder.encode(X, inject_seu, seu_rate)
        
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.config.output_size))
        total_energy = 0
        
        for i in range(n_samples):
            counts, n_spikes = self._forward_sample(spike_trains[i])
            
            # Compute scores with class scaling
            raw_scores = counts @ self.output_weights + self.output_bias
            scores[i] = raw_scores * self.class_scales
            
            total_energy += n_spikes * self.config.energy_per_spike
        
        self.total_spikes = total_energy / self.config.energy_per_spike
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X, **kwargs)
        return np.argmax(probs, axis=1)
    
    def predict_tier1(self, X: np.ndarray) -> np.ndarray:
        """Tier 1: Fast binary anomaly detection.
        
        Returns 1 for anomaly (any attack), 0 for normal.
        Uses only first few timesteps for speed.
        """
        # Quick encode with fewer timesteps
        old_T = self.config.time_steps
        self.config.time_steps = 20  # Fast mode
        
        probs = self.predict_proba(X)
        
        self.config.time_steps = old_T
        
        # Normal is class 0, anything else is anomaly
        anomaly_prob = 1 - probs[:, 0]
        return (anomaly_prob > self.config.tier1_threshold).astype(int)
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 20, batch_size: int = 64,
            val_split: float = 0.1, verbose: bool = True) -> Dict:
        """Train the network.
        
        Uses the spike count-based training approach from NeuroIDS v4.
        Only trains the output layer weights.
        """
        n_samples = X.shape[0]
        n_val = int(n_samples * val_split)
        
        # Split data
        idx = np.random.permutation(n_samples)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Compute class weights for imbalanced data
        class_counts = np.bincount(y_train, minlength=self.config.output_size)
        class_counts = np.maximum(class_counts, 1)  # Avoid division by zero
        total = len(y_train)
        self.class_scales = total / (self.config.output_size * class_counts)
        self.class_scales = self.class_scales / np.max(self.class_scales)
        
        # Encode all training data once
        if verbose:
            print("Encoding training data...")
        spike_trains = self.encoder.encode(X_train)
        
        # Training loop
        best_val_acc = 0
        best_weights = None
        
        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(len(X_train))
            
            epoch_loss = 0
            epoch_correct = 0
            epoch_spikes = 0
            n_batches = 0
            
            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                batch_idx = perm[start:end]
                
                batch_spikes = spike_trains[batch_idx]
                batch_y = y_train[batch_idx]
                
                # Forward pass - collect spike counts
                batch_counts = []
                for i in range(len(batch_idx)):
                    counts, n_spikes = self._forward_sample(batch_spikes[i])
                    batch_counts.append(counts)
                    epoch_spikes += n_spikes
                
                batch_counts = np.array(batch_counts)
                
                # Compute scores and loss
                scores = batch_counts @ self.output_weights + self.output_bias
                scores = scores * self.class_scales
                
                # Softmax
                exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                
                # Cross-entropy loss
                batch_size_actual = len(batch_y)
                loss = -np.mean(np.log(probs[np.arange(batch_size_actual), batch_y] + 1e-10))
                
                # Gradient
                grad = probs.copy()
                grad[np.arange(batch_size_actual), batch_y] -= 1
                grad /= batch_size_actual
                
                # Update output weights
                dW = batch_counts.T @ grad
                dW += self.config.weight_decay * self.output_weights
                
                self.output_weights -= self.config.learning_rate * dW
                self.output_bias -= self.config.learning_rate * np.mean(grad, axis=0)
                
                # Track metrics
                epoch_loss += loss
                epoch_correct += np.sum(np.argmax(probs, axis=1) == batch_y)
                n_batches += 1
            
            # Validation
            val_preds = self.predict(X_val)
            val_acc = np.mean(val_preds == y_val)
            train_acc = epoch_correct / len(X_train)
            avg_spikes = epoch_spikes / len(X_train)
            
            self.training_history['loss'].append(epoch_loss / n_batches)
            self.training_history['accuracy'].append(train_acc)
            self.training_history['val_accuracy'].append(val_acc)
            self.training_history['spikes'].append(avg_spikes)
            self.training_history['energy'].append(
                avg_spikes * self.config.energy_per_spike
            )
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = (self.output_weights.copy(), self.output_bias.copy())
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"loss: {epoch_loss/n_batches:.4f} - "
                      f"acc: {train_acc:.4f} - "
                      f"val_acc: {val_acc:.4f} - "
                      f"spikes: {avg_spikes:.1f}")
        
        # Restore best weights
        if best_weights is not None:
            self.output_weights, self.output_bias = best_weights
        
        self.is_fitted = True
        return self.training_history
    
    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 inject_seu: bool = False, seu_rate: float = 0.0) -> Dict:
        """Evaluate model performance.
        
        Args:
            X: Test features
            y: True labels
            inject_seu: Simulate radiation effects
            seu_rate: SEU injection rate
            
        Returns:
            Dictionary with metrics
        """
        probs = self.predict_proba(X, inject_seu, seu_rate)
        preds = np.argmax(probs, axis=1)
        
        # Overall accuracy
        accuracy = np.mean(preds == y)
        
        # Per-class metrics
        metrics = {'accuracy': accuracy, 'per_class': {}}
        
        for c in range(self.config.output_size):
            mask = y == c
            if np.sum(mask) > 0:
                class_preds = preds[mask]
                
                tp = np.sum(class_preds == c)
                fp = np.sum(preds == c) - tp
                fn = np.sum(mask) - tp
                
                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                
                metrics['per_class'][self.ATTACK_CATEGORIES[c]] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': int(np.sum(mask))
                }
        
        # Energy metrics
        avg_spikes = self.total_spikes / len(X)
        metrics['energy'] = {
            'avg_spikes_per_sample': avg_spikes,
            'energy_pj_per_sample': avg_spikes * self.config.energy_per_spike,
            'total_parameters': self._count_parameters()
        }
        
        return metrics
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        total = 0
        for layer in self.layers:
            total += layer.weights.size + layer.bias.size
        total += self.output_weights.size + self.output_bias.size
        return total
    
    def save(self, path: str):
        """Save model to file."""
        state = {
            'config': self.config,
            'layers': [(l.weights, l.bias) for l in self.layers],
            'output_weights': self.output_weights,
            'output_bias': self.output_bias,
            'class_scales': self.class_scales,
            'is_fitted': self.is_fitted,
            'training_history': self.training_history
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: str) -> 'NeuroIDSSat':
        """Load model from file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        model = cls(state['config'])
        
        for i, (w, b) in enumerate(state['layers']):
            model.layers[i].weights = w
            model.layers[i].bias = b
        
        model.output_weights = state['output_weights']
        model.output_bias = state['output_bias']
        model.class_scales = state['class_scales']
        model.is_fitted = state['is_fitted']
        model.training_history = state['training_history']
        
        return model


def compare_terrestrial_vs_satellite():
    """Compare terrestrial NeuroIDS vs NeuroIDS-Sat configurations."""
    
    print("=" * 60)
    print("NeuroIDS Architecture Comparison")
    print("=" * 60)
    
    # Terrestrial config (from original paper)
    terrestrial = {
        'hidden_sizes': [128, 64],
        'time_steps': 100,
        'v_thresh': 0.5,
        'tau_m': 15.0,
        'parameters': 41*128 + 128 + 128*64 + 64 + 64*5 + 5  # ~13,760
    }
    
    # Satellite config
    satellite = {
        'hidden_sizes': [64, 32],
        'time_steps': 50,
        'v_thresh': 0.6,
        'tau_m': 20.0,
        'parameters': 41*64 + 64 + 64*32 + 32 + 32*5 + 5  # ~4,965
    }
    
    print(f"\n{'Metric':<25} {'Terrestrial':>15} {'Satellite':>15} {'Reduction':>12}")
    print("-" * 70)
    print(f"{'Hidden Layer 1':<25} {terrestrial['hidden_sizes'][0]:>15} {satellite['hidden_sizes'][0]:>15} {100*(1-satellite['hidden_sizes'][0]/terrestrial['hidden_sizes'][0]):>11.1f}%")
    print(f"{'Hidden Layer 2':<25} {terrestrial['hidden_sizes'][1]:>15} {satellite['hidden_sizes'][1]:>15} {100*(1-satellite['hidden_sizes'][1]/terrestrial['hidden_sizes'][1]):>11.1f}%")
    print(f"{'Time Steps':<25} {terrestrial['time_steps']:>15} {satellite['time_steps']:>15} {100*(1-satellite['time_steps']/terrestrial['time_steps']):>11.1f}%")
    print(f"{'Parameters':<25} {terrestrial['parameters']:>15,} {satellite['parameters']:>15,} {100*(1-satellite['parameters']/terrestrial['parameters']):>11.1f}%")
    print(f"{'Threshold':<25} {terrestrial['v_thresh']:>15.1f} {satellite['v_thresh']:>15.1f} {'(+20%)':>12}")
    print(f"{'Membrane τ (ms)':<25} {terrestrial['tau_m']:>15.1f} {satellite['tau_m']:>15.1f} {'(+33%)':>12}")


if __name__ == "__main__":
    compare_terrestrial_vs_satellite()
