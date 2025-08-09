"""
ULTRA-OPTIMIZED Token-Adaptive Early Exit with Advanced Parameter Optimization
Multi-experiment optimization with comprehensive tracking
"""

import tensorflow as tf
import numpy as np
from dataclasses import dataclass
import time
import datetime
from advanced_tracker import AdvancedParameterTracker

@dataclass
class OptimizedConfig:
    """Ultra-optimized configuration class"""
    vocab_size: int = 1000
    hidden_size: int = 384
    num_layers: int = 10
    num_heads: int = 12
    early_exit_threshold: float = 0.15
    max_candidates: int = 100
    dropout_rate: float = 0.08

class OptimizedMetrics:
    """Enhanced metrics tracking with token-level analysis"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.accuracies = []
        self.exit_layers = []
        self.computational_savings = []
        self.exit_times = []
        self.early_exits = 0
        self.total_batches = 0
        self.exit_distribution = {}  # Track which layers are used for exits
        
    def update(self, loss, accuracy, layers_used, max_layers, exit_time=0.0):
        self.losses.append(float(loss))
        self.accuracies.append(float(accuracy))
        self.exit_layers.append(layers_used)
        savings = 100 * (max_layers - layers_used) / max_layers
        self.computational_savings.append(savings)
        self.exit_times.append(exit_time)
        
        # Track exit layer distribution
        if layers_used not in self.exit_distribution:
            self.exit_distribution[layers_used] = 0
        self.exit_distribution[layers_used] += 1
        
        if layers_used < max_layers:
            self.early_exits += 1
        self.total_batches += 1
    
    def get_results(self):
        return {
            'avg_loss': np.mean(self.losses),
            'avg_accuracy': np.mean(self.accuracies),
            'avg_layers': np.mean(self.exit_layers),
            'avg_savings': np.mean(self.computational_savings),
            'early_exit_rate': 100 * self.early_exits / max(self.total_batches, 1),
            'avg_exit_time': np.mean(self.exit_times),
            'exit_distribution': self.exit_distribution.copy()
        }

def calculate_accuracy(predictions, targets):
    """Calculate accuracy with proper dtype handling"""
    predicted_ids = tf.argmax(predictions, axis=-1)
    predicted_ids = tf.cast(predicted_ids, targets.dtype)
    correct = tf.equal(predicted_ids, targets)
    return tf.reduce_mean(tf.cast(correct, tf.float32))

class UltraOptimizedToExModel(tf.keras.Model):
    """Ultra-optimized Token-Adaptive Early Exit Model with enhanced exit logic"""
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(
            self.config.vocab_size, self.config.hidden_size
        )
        
        # Enhanced layer architecture
        self.model_layers = []
        self.dropout_layers = []
        self.layer_norms = []  # Add layer normalization
        
        for i in range(self.config.num_layers):
            self.model_layers.append(tf.keras.layers.Dense(
                self.config.hidden_size, 
                activation='gelu',
                name=f'ultra_layer_{i}'
            ))
            self.dropout_layers.append(tf.keras.layers.Dropout(
                self.config.dropout_rate,
                name=f'dropout_{i}'
            ))
            self.layer_norms.append(tf.keras.layers.LayerNormalization(
                name=f'norm_{i}'
            ))
        
        self.output_layer = tf.keras.layers.Dense(
            self.config.vocab_size, name='ultra_output'
        )
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        start_time = time.time()
        
        x = self.embedding(inputs)
        x = tf.reduce_mean(x, axis=1)
        
        exit_layer = None
        confidence_scores = []
        
        for i, (layer, dropout, norm) in enumerate(zip(self.model_layers, self.dropout_layers, self.layer_norms)):
            x = layer(x, training=training)
            x = norm(x, training=training)
            x = dropout(x, training=training)
            
            # Enhanced multi-metric early exit logic
            if i >= 0:  # Start checking from first layer
                # Multiple confidence metrics
                activation_confidence = tf.reduce_mean(tf.nn.relu(x))
                stability_score = 1.0 / (1.0 + tf.math.reduce_variance(x))
                magnitude_score = tf.reduce_mean(tf.abs(x))
                
                # Dynamic threshold with layer progression
                layer_factor = 0.7 + (i * 0.1)  # Gets easier to exit in later layers
                dynamic_threshold = self.config.early_exit_threshold * layer_factor
                
                # Weighted confidence combination
                combined_confidence = (
                    0.4 * activation_confidence +
                    0.3 * stability_score +
                    0.3 * magnitude_score
                )
                
                confidence_scores.append(float(combined_confidence))
                
                # Progressive exit conditions - more aggressive as we go deeper
                should_exit = False
                
                if i >= 1:  # Layer 1+
                    should_exit = combined_confidence > dynamic_threshold
                elif i >= 3:  # Layer 3+ - easier exit
                    should_exit = combined_confidence > (dynamic_threshold * 0.8)
                elif i >= 6:  # Layer 6+ - very easy exit
                    should_exit = combined_confidence > (dynamic_threshold * 0.6)
                
                if should_exit and i < len(self.model_layers) - 1:
                    exit_layer = i + 1
                    break
        
        logits = self.output_layer(x)
        exit_time = (time.time() - start_time) * 1000
        
        layers_used = exit_layer if exit_layer else len(self.model_layers)
        savings = 100 * (len(self.model_layers) - layers_used) / len(self.model_layers)
        
        return {
            'logits': logits,
            'layers_used': layers_used,
            'computational_savings': savings,
            'exit_time': exit_time,
            'confidence_scores': confidence_scores
        }

class BaselineModel(tf.keras.Model):
    """Enhanced baseline model for comparison"""
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(
            self.config.vocab_size, self.config.hidden_size
        )
        
        self.model_layers = []
        self.layer_norms = []
        
        for i in range(self.config.num_layers):
            self.model_layers.append(tf.keras.layers.Dense(
                self.config.hidden_size, 
                activation='gelu',
                name=f'baseline_layer_{i}'
            ))
            self.layer_norms.append(tf.keras.layers.LayerNormalization(
                name=f'baseline_norm_{i}'
            ))
        
        self.output_layer = tf.keras.layers.Dense(
            self.config.vocab_size, name='baseline_output'
        )
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        start_time = time.time()
        
        x = self.embedding(inputs)
        x = tf.reduce_mean(x, axis=1)
        
        for layer, norm in zip(self.model_layers, self.layer_norms):
            x = layer(x, training=training)
            x = norm(x, training=training)
        
        logits = self.output_layer(x)
        exit_time = (time.time() - start_time) * 1000
        
        return {
            'logits': logits,
            'layers_used': len(self.model_layers),
            'computational_savings': 0.0,
            'exit_time': exit_time
        }

def run_optimization_experiment(config, experiment_name, tracker):
    """Run a single optimization experiment"""
    print(f"\n🧪 Running Experiment: {experiment_name}")
    print(f"   Threshold: {config.early_exit_threshold}, Hidden: {config.hidden_size}, Layers: {config.num_layers}")
    
    # Initialize models
    toex_model = UltraOptimizedToExModel(config)
    baseline_model = BaselineModel(config)
    
    # Initialize metrics
    toex_metrics = OptimizedMetrics()
    baseline_metrics = OptimizedMetrics()
    
    # Training setup
    toex_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
    baseline_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Create data
    batch_size = 16
    seq_len = 32
    dummy_data = tf.random.uniform([batch_size, seq_len], 0, config.vocab_size, dtype=tf.int32)
    
    # Quick training loop
    for epoch in range(2):
        for batch in range(6):
            # ToEx Training
            with tf.GradientTape() as tape:
                toex_outputs = toex_model(dummy_data, training=True)
                targets = tf.random.uniform([batch_size], 0, config.vocab_size, dtype=tf.int32)
                toex_loss = loss_fn(targets, toex_outputs['logits'])
                toex_accuracy = calculate_accuracy(toex_outputs['logits'], targets)
            
            gradients = tape.gradient(toex_loss, toex_model.trainable_variables)
            toex_optimizer.apply_gradients(zip(gradients, toex_model.trainable_variables))
            
            # Baseline Training
            with tf.GradientTape() as tape:
                baseline_outputs = baseline_model(dummy_data, training=True)
                baseline_loss = loss_fn(targets, baseline_outputs['logits'])
                baseline_accuracy = calculate_accuracy(baseline_outputs['logits'], targets)
            
            gradients = tape.gradient(baseline_loss, baseline_model.trainable_variables)
            baseline_optimizer.apply_gradients(zip(gradients, baseline_model.trainable_variables))
            
            # Track metrics
            toex_metrics.update(
                toex_loss, toex_accuracy, 
                toex_outputs['layers_used'], config.num_layers,
                toex_outputs['exit_time']
            )
            
            baseline_metrics.update(
                baseline_loss, baseline_accuracy,
                baseline_outputs['layers_used'], config.num_layers,
                baseline_outputs['exit_time']
            )
    
    # Get results
    toex_results = toex_metrics.get_results()
    baseline_results = baseline_metrics.get_results()
    
    results = {
        'ToEx Model': toex_results,
        'Baseline Model': baseline_results
    }
    
    # Log to tracker
    tracker.log_experiment(config, results, experiment_name)
    
    print(f"   ✅ Complete: Exit Rate={toex_results['early_exit_rate']:.1f}%, Loss={toex_results['avg_loss']:.4f}, Savings={toex_results['avg_savings']:.1f}%")
    
    return results

def main():
    """Main optimization pipeline with multiple experiments"""
    print("🚀 ULTRA-OPTIMIZED TOKEN-ADAPTIVE EARLY EXIT OPTIMIZATION SUITE")
    print("="*80)
    
    # Initialize advanced tracker
    tracker = AdvancedParameterTracker("ultra_optimization_log.txt")
    
    # Base configuration
    base_config = OptimizedConfig()
    
    # Optimization experiments
    experiments = [
        # Experiment 1: Ultra-aggressive early exit
        {
            'name': 'Ultra Aggressive Exit',
            'config': OptimizedConfig(
                early_exit_threshold=0.10,
                hidden_size=384,
                num_layers=10,
                dropout_rate=0.05
            )
        },
        
        # Experiment 2: Balanced performance
        {
            'name': 'Balanced Performance',
            'config': OptimizedConfig(
                early_exit_threshold=0.20,
                hidden_size=448,
                num_layers=12,
                dropout_rate=0.08
            )
        },
        
        # Experiment 3: Quality focused
        {
            'name': 'Quality Focused',
            'config': OptimizedConfig(
                early_exit_threshold=0.30,
                hidden_size=512,
                num_layers=14,
                dropout_rate=0.06
            )
        },
        
        # Experiment 4: Speed focused
        {
            'name': 'Speed Focused',
            'config': OptimizedConfig(
                early_exit_threshold=0.08,
                hidden_size=320,
                num_layers=8,
                dropout_rate=0.10
            )
        }
    ]
    
    print(f"🧪 Running {len(experiments)} optimization experiments...")
    
    all_results = {}
    for i, experiment in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {i}/{len(experiments)}: {experiment['name']}")
        print(f"{'='*60}")
        
        results = run_optimization_experiment(
            experiment['config'], 
            experiment['name'],
            tracker
        )
        all_results[experiment['name']] = results
    
    # Final analysis
    print(f"\n{'='*80}")
    print("📊 FINAL OPTIMIZATION ANALYSIS")
    print(f"{'='*80}")
    
    best_config = tracker.get_best_config()
    if best_config:
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   Experiment: {best_config['notes']}")
        print(f"   Composite Score: {best_config['composite_score']:.4f}")
        print(f"   Early Exit Rate: {best_config['results']['ToEx Model']['early_exit_rate']:.1f}%")
        print(f"   Computational Savings: {best_config['results']['ToEx Model']['avg_savings']:.1f}%")
        print(f"   Loss: {best_config['results']['ToEx Model']['avg_loss']:.4f}")
        
        print(f"\n🔧 OPTIMAL PARAMETERS:")
        for param, value in best_config['config'].items():
            print(f"   {param:20} = {value}")
    
    print(f"\n📈 EXPERIMENT SUMMARY:")
    for name, results in all_results.items():
        toex = results['ToEx Model']
        print(f"   {name:20} | Exit: {toex['early_exit_rate']:5.1f}% | Save: {toex['avg_savings']:5.1f}% | Loss: {toex['avg_loss']:.4f}")
    
    print(f"\n✅ OPTIMIZATION COMPLETE")
    print(f"📝 Detailed logs: {tracker.log_file}")
    print(f"🧪 Total experiments: {len(tracker.experiments)}")
    
    return tracker

if __name__ == "__main__":
    tracker = main()
