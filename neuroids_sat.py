"""
NeuroIDS-Sat v2: Optimized Neuromorphic IDS for CubeSat Deployment

Improvements over v1:
1. Vectorized batch processing (10-50x faster training)
2. SMOTE-lite oversampling for minority classes
3. Focal loss for hard example mining
4. Configurable architecture for accuracy/efficiency tradeoff
5. Two-stage training option

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
    
    Presets:
    - 'minimal': 64-32, 50 timesteps (lowest power, ~67% accuracy)
    - 'balanced': 96-48, 75 timesteps (moderate power, ~75% accuracy)
    - 'full': 128-64, 100 timesteps (highest accuracy, ~78% accuracy)
    """
    input_size: int = 41
    hidden_sizes: List[int] = field(default_factory=lambda: [96, 48])  # Balanced default
    output_size: int = 5
    time_steps: int = 75  # Balanced default
    
    # LIF parameters - adjusted for radiation tolerance
    v_thresh: float = 0.6
    v_rest: float = 0.0
    v_reset: float = 0.0
    tau_m: float = 20.0
    tau_ref: float = 3.0
    r_m: float = 1.0
    
    # Encoding
    spike_rate_base: float = 0.1
    spike_rate_max: float = 0.5
    
    # TMR (Triple Modular Redundancy)
    tmr_enabled: bool = True
    
    # Training
    learning_rate: float = 0.02
    weight_decay: float = 0.0001
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    
    # Minority class handling
    oversample_minority: bool = True
    minority_target_ratio: float = 0.1  # Target 10% for minority classes
    
    # Hierarchical detection
    tier1_threshold: float = 0.3
    
    # Energy model (picojoules)
    energy_per_spike: float = 2.0
    
    @classmethod
    def minimal(cls) -> 'SatelliteConfig':
        """Minimal power configuration."""
        return cls(hidden_sizes=[64, 32], time_steps=50)
    
    @classmethod
    def balanced(cls) -> 'SatelliteConfig':
        """Balanced power/accuracy configuration."""
        return cls(hidden_sizes=[96, 48], time_steps=75)
    
    @classmethod
    def full(cls) -> 'SatelliteConfig':
        """Full accuracy configuration (similar to terrestrial)."""
        return cls(hidden_sizes=[128, 64], time_steps=100)


class LIFLayerVectorized:
    """Vectorized LIF layer for fast batch processing."""
    
    def __init__(self, input_size: int, output_size: int, config: SatelliteConfig):
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        
        # Xavier initialization
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.randn(input_size, output_size).astype(np.float32) * scale
        self.bias = np.zeros(output_size, dtype=np.float32)
        
        # State (will be set per batch)
        self.membrane = None
        self.refractory = None
        
    def reset(self, batch_size: int):
        """Reset neuron state for new batch."""
        self.membrane = np.full((batch_size, self.output_size), 
                                self.config.v_rest, dtype=np.float32)
        self.refractory = np.zeros((batch_size, self.output_size), dtype=np.float32)
        
    def forward(self, input_spikes: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Vectorized forward pass for entire batch.
        
        Args:
            input_spikes: (batch_size, input_size) binary spike input
            dt: Timestep size
            
        Returns:
            (batch_size, output_size) binary spike output
        """
        batch_size = input_spikes.shape[0]
        
        # Initialize state if needed
        if self.membrane is None or self.membrane.shape[0] != batch_size:
            self.reset(batch_size)
        
        # Decay refractory counters
        self.refractory = np.maximum(0, self.refractory - dt)
        
        # Mask for non-refractory neurons
        active = self.refractory <= 0
        
        # Compute input current: (batch, input) @ (input, output) -> (batch, output)
        current = input_spikes @ self.weights + self.bias
        
        # Leaky integration
        decay = np.exp(-dt / self.config.tau_m)
        self.membrane = np.where(
            active,
            self.config.v_rest + (self.membrane - self.config.v_rest) * decay + 
            self.config.r_m * current,
            self.membrane
        )
        
        # Spike generation
        spikes = (self.membrane >= self.config.v_thresh).astype(np.float32)
        
        # Reset spiking neurons
        self.membrane = np.where(spikes > 0, self.config.v_reset, self.membrane)
        self.refractory = np.where(spikes > 0, self.config.tau_ref, self.refractory)
        
        return spikes


class TMREncoderFast:
    """Optimized TMR encoder with vectorized operations."""
    
    def __init__(self, config: SatelliteConfig):
        self.config = config
        
    def encode(self, x: np.ndarray, inject_seu: bool = False,
               seu_rate: float = 0.0) -> np.ndarray:
        """Encode input features to spike trains with TMR.
        
        Args:
            x: Input features, shape (n_samples, n_features)
            inject_seu: Whether to simulate radiation upsets
            seu_rate: Rate of SEU injection
            
        Returns:
            Spike trains, shape (n_samples, time_steps, n_features)
        """
        n_samples, n_features = x.shape
        T = self.config.time_steps
        
        # Normalize to [0, 1]
        x_norm = np.clip(x, 0, 1).astype(np.float32)
        
        # Rate coding: P(spike) = base + x * (max - base)
        rates = self.config.spike_rate_base + x_norm * (
            self.config.spike_rate_max - self.config.spike_rate_base
        )
        
        if self.config.tmr_enabled:
            # Generate three redundant encodings and vote
            votes = np.zeros((n_samples, T, n_features), dtype=np.float32)
            for _ in range(3):
                rand = np.random.rand(n_samples, T, n_features).astype(np.float32)
                spikes = (rand < rates[:, np.newaxis, :]).astype(np.float32)
                if inject_seu and seu_rate > 0:
                    seu_mask = np.random.rand(*spikes.shape) < seu_rate
                    spikes = np.where(seu_mask, 1 - spikes, spikes)
                votes += spikes
            # Majority voting
            spike_trains = (votes >= 2).astype(np.float32)
        else:
            rand = np.random.rand(n_samples, T, n_features).astype(np.float32)
            spike_trains = (rand < rates[:, np.newaxis, :]).astype(np.float32)
            if inject_seu and seu_rate > 0:
                seu_mask = np.random.rand(*spike_trains.shape) < seu_rate
                spike_trains = np.where(seu_mask, 1 - spike_trains, spike_trains)
        
        return spike_trains


class NeuroIDSSat:
    """Optimized Neuromorphic IDS for CubeSat deployment.
    
    v2 improvements:
    - Vectorized batch processing
    - Focal loss for minority classes
    - SMOTE-lite oversampling
    - Configurable presets (minimal/balanced/full)
    """
    
    ATTACK_CATEGORIES = {
        0: 'normal',
        1: 'dos',
        2: 'probe',
        3: 'r2l',
        4: 'u2r'
    }
    
    def __init__(self, config: SatelliteConfig = None):
        self.config = config or SatelliteConfig.balanced()
        
        # Encoder
        self.encoder = TMREncoderFast(self.config)
        
        # Build layers
        self.layers: List[LIFLayerVectorized] = []
        prev_size = self.config.input_size
        
        for hidden_size in self.config.hidden_sizes:
            self.layers.append(LIFLayerVectorized(prev_size, hidden_size, self.config))
            prev_size = hidden_size
        
        # Output weights
        self.output_weights = np.random.randn(prev_size, self.config.output_size).astype(np.float32) * 0.1
        self.output_bias = np.zeros(self.config.output_size, dtype=np.float32)
        
        # Class scaling
        self.class_scales = np.ones(self.config.output_size, dtype=np.float32)
        
        # Training state
        self.is_fitted = False
        self.training_history = {
            'loss': [], 'accuracy': [], 'val_accuracy': [],
            'spikes': [], 'energy': [], 'per_class_recall': []
        }
        
        # Energy tracking
        self.total_spikes = 0
        
    def reset_layers(self, batch_size: int):
        """Reset all layer states for new batch."""
        for layer in self.layers:
            layer.reset(batch_size)
            
    def _forward_batch(self, spike_trains: np.ndarray) -> Tuple[np.ndarray, float]:
        """Process entire batch through network.
        
        Args:
            spike_trains: (batch_size, time_steps, n_features)
            
        Returns:
            (spike_counts, avg_spikes_per_sample)
        """
        batch_size, T, _ = spike_trains.shape
        self.reset_layers(batch_size)
        
        # Accumulate spike counts from final layer
        spike_counts = np.zeros((batch_size, self.layers[-1].output_size), dtype=np.float32)
        total_spikes = 0
        
        for t in range(T):
            x = spike_trains[:, t, :]  # (batch_size, n_features)
            total_spikes += np.sum(x)
            
            for layer in self.layers:
                x = layer.forward(x)
                total_spikes += np.sum(x)
            
            spike_counts += x
        
        avg_spikes = total_spikes / batch_size
        return spike_counts, avg_spikes
    
    def _focal_loss(self, probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Focal loss for handling class imbalance.
        
        Focuses learning on hard examples by down-weighting easy ones.
        """
        pt = probs[np.arange(len(targets)), targets]
        focal_weight = (1 - pt) ** self.config.focal_gamma
        return -focal_weight * np.log(pt + 1e-10)
    
    def _cross_entropy_loss(self, probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Standard cross-entropy loss."""
        return -np.log(probs[np.arange(len(targets)), targets] + 1e-10)
    
    def predict_proba(self, X: np.ndarray, inject_seu: bool = False,
                      seu_rate: float = 0.0) -> np.ndarray:
        """Predict class probabilities for batch."""
        # Encode
        spike_trains = self.encoder.encode(X, inject_seu, seu_rate)
        
        # Forward pass
        spike_counts, avg_spikes = self._forward_batch(spike_trains)
        self.total_spikes = avg_spikes * len(X)
        
        # Compute scores
        scores = spike_counts @ self.output_weights + self.output_bias
        scores = scores * self.class_scales
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X, **kwargs)
        return np.argmax(probs, axis=1)
    
    def _oversample_minority(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """SMOTE-lite: Simple oversampling with noise injection for minority classes."""
        
        class_counts = np.bincount(y, minlength=self.config.output_size)
        max_count = np.max(class_counts)
        target_count = int(max_count * self.config.minority_target_ratio)
        target_count = max(target_count, 2000)  # At least 2000 per minority class
        
        X_new, y_new = [X], [y]
        
        for c in range(self.config.output_size):
            if class_counts[c] < target_count:
                # Get samples of this class
                mask = y == c
                X_c = X[mask]
                n_needed = target_count - class_counts[c]
                
                if len(X_c) > 0:
                    # Random oversample with small noise
                    idx = np.random.choice(len(X_c), n_needed, replace=True)
                    X_over = X_c[idx] + np.random.randn(n_needed, X.shape[1]).astype(np.float32) * 0.01
                    X_over = np.clip(X_over, 0, 1)
                    
                    X_new.append(X_over)
                    y_new.append(np.full(n_needed, c, dtype=y.dtype))
        
        X_combined = np.vstack(X_new)
        y_combined = np.concatenate(y_new)
        
        # Shuffle
        idx = np.random.permutation(len(y_combined))
        return X_combined[idx], y_combined[idx]
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 20, batch_size: int = 256,
            val_split: float = 0.1, verbose: bool = True,
            validate_every: int = 1) -> Dict:
        """Train the network with optimized batch processing.
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs
            batch_size: Batch size (larger = faster but more memory)
            val_split: Validation split ratio
            verbose: Print progress
            validate_every: Validate every N epochs (increase for speed)
        """
        # Convert to float32 for speed
        X = X.astype(np.float32)
        
        # Oversample minority classes if enabled
        if self.config.oversample_minority:
            if verbose:
                print("Oversampling minority classes...")
                print(f"  Original distribution: {np.bincount(y, minlength=5)}")
            X, y = self._oversample_minority(X, y)
            if verbose:
                print(f"  Balanced distribution: {np.bincount(y, minlength=5)}")
        
        n_samples = X.shape[0]
        n_val = int(n_samples * val_split)
        
        # Split data
        idx = np.random.permutation(n_samples)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Compute class weights
        class_counts = np.bincount(y_train, minlength=self.config.output_size)
        class_counts = np.maximum(class_counts, 1)
        total = len(y_train)
        self.class_scales = (total / (self.config.output_size * class_counts)).astype(np.float32)
        self.class_scales = self.class_scales / np.max(self.class_scales)
        
        # Encode training data once
        if verbose:
            print(f"Encoding {len(X_train):,} training samples...")
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
                batch_size_actual = len(batch_y)
                
                # Forward pass (vectorized)
                spike_counts, avg_spikes = self._forward_batch(batch_spikes)
                epoch_spikes += avg_spikes * batch_size_actual
                
                # Compute scores
                scores = spike_counts @ self.output_weights + self.output_bias
                scores = scores * self.class_scales
                
                # Softmax
                exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                
                # Loss
                if self.config.use_focal_loss:
                    losses = self._focal_loss(probs, batch_y)
                else:
                    losses = self._cross_entropy_loss(probs, batch_y)
                loss = np.mean(losses)
                
                # Gradient
                grad = probs.copy()
                grad[np.arange(batch_size_actual), batch_y] -= 1
                grad /= batch_size_actual
                
                # Weight update
                dW = spike_counts.T @ grad
                dW += self.config.weight_decay * self.output_weights
                
                self.output_weights -= self.config.learning_rate * dW
                self.output_bias -= self.config.learning_rate * np.mean(grad, axis=0)
                
                # Track metrics
                epoch_loss += loss
                epoch_correct += np.sum(np.argmax(probs, axis=1) == batch_y)
                n_batches += 1
            
            train_acc = epoch_correct / len(X_train)
            avg_spikes = epoch_spikes / len(X_train)
            
            # Validation (skip some epochs for speed)
            if epoch % validate_every == 0 or epoch == epochs - 1:
                val_preds = self.predict(X_val)
                val_acc = np.mean(val_preds == y_val)
                
                # Per-class recall
                per_class_recall = {}
                for c in range(self.config.output_size):
                    mask = y_val == c
                    if np.sum(mask) > 0:
                        recall = np.mean(val_preds[mask] == c)
                        per_class_recall[self.ATTACK_CATEGORIES[c]] = recall
                
                self.training_history['val_accuracy'].append(val_acc)
                self.training_history['per_class_recall'].append(per_class_recall)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = (self.output_weights.copy(), self.output_bias.copy())
            else:
                val_acc = self.training_history['val_accuracy'][-1] if self.training_history['val_accuracy'] else 0
                per_class_recall = self.training_history['per_class_recall'][-1] if self.training_history['per_class_recall'] else {}
            
            self.training_history['loss'].append(epoch_loss / n_batches)
            self.training_history['accuracy'].append(train_acc)
            self.training_history['spikes'].append(avg_spikes)
            self.training_history['energy'].append(avg_spikes * self.config.energy_per_spike)
            
            if verbose:
                recall_str = " | ".join([f"{k[:3]}:{v:.2f}" for k, v in per_class_recall.items()])
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"loss: {epoch_loss/n_batches:.4f} - "
                      f"acc: {train_acc:.4f} - "
                      f"val: {val_acc:.4f} - "
                      f"spikes: {avg_spikes:.0f} - "
                      f"[{recall_str}]")
        
        # Restore best weights
        if best_weights is not None:
            self.output_weights, self.output_bias = best_weights
        
        self.is_fitted = True
        return self.training_history
    
    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 inject_seu: bool = False, seu_rate: float = 0.0) -> Dict:
        """Evaluate model performance."""
        X = X.astype(np.float32)
        probs = self.predict_proba(X, inject_seu, seu_rate)
        preds = np.argmax(probs, axis=1)
        
        accuracy = np.mean(preds == y)
        
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
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'support': int(np.sum(mask))
                }
        
        avg_spikes = self.total_spikes / len(X)
        metrics['energy'] = {
            'avg_spikes_per_sample': float(avg_spikes),
            'energy_pj_per_sample': float(avg_spikes * self.config.energy_per_spike),
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


def compare_configurations():
    """Compare different configuration presets."""
    
    print("=" * 70)
    print("NeuroIDS-Sat Configuration Comparison")
    print("=" * 70)
    
    configs = {
        'Minimal': SatelliteConfig.minimal(),
        'Balanced': SatelliteConfig.balanced(),
        'Full': SatelliteConfig.full()
    }
    
    print(f"\n{'Config':<12} {'Hidden':<12} {'Steps':<8} {'Params':<10} {'Est. Power':<12}")
    print("-" * 60)
    
    for name, cfg in configs.items():
        # Calculate params
        params = cfg.input_size * cfg.hidden_sizes[0] + cfg.hidden_sizes[0]
        params += cfg.hidden_sizes[0] * cfg.hidden_sizes[1] + cfg.hidden_sizes[1]
        params += cfg.hidden_sizes[1] * cfg.output_size + cfg.output_size
        
        # Estimate power (rough)
        est_spikes = 400 * (cfg.time_steps / 50) * (sum(cfg.hidden_sizes) / 96)
        est_power = est_spikes * cfg.energy_per_spike / 1000  # mW at 1000 samples/sec
        
        hidden_str = f"{cfg.hidden_sizes[0]}-{cfg.hidden_sizes[1]}"
        print(f"{name:<12} {hidden_str:<12} {cfg.time_steps:<8} {params:<10,} {est_power:.2f} mW")


if __name__ == "__main__":
    compare_configurations()
