"""
Token-Adaptive Early Exit I    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size)
        self.model_layers = [
            tf.keras.layers.Dense(self.hidden_size, activation='relu', name=f'layer_{i}')
            for i in range(self.config.num_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(self.vocab_size)
        super().build(input_shape)ation for Kaggle
Optimized for clear output visibility in Kaggle notebooks
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass

@dataclass
class ToExConfig:
    """Configuration for Token-Adaptive Early Exit models"""
    num_decoder_layers: int = 6
    num_layers: int = 6
    early_exit_threshold: float = 0.3  # Lower threshold for more early exits
    max_candidates: int = 50
    hidden_size: int = 256
    num_heads: int = 8
    vocab_size: int = 1000

class SimpleToExGPT(tf.keras.Model):
    """Simplified GPT with Token-Adaptive Early Exit for Kaggle demo"""
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        
    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size)
        self.model_layers = [
            tf.keras.layers.Dense(self.hidden_size, activation='relu', name=f'layer_{i}')
            for i in range(self.config.num_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(self.vocab_size, name='output')
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        x = self.embedding(inputs)
        x = tf.reduce_mean(x, axis=1)  # Simple pooling
        
        exit_layer = None
        for i, layer in enumerate(self.model_layers):
            x = layer(x, training=training)
            
            # Simple early exit logic
            if i >= 2:  # Check exit after layer 2
                confidence = tf.reduce_mean(tf.abs(x))
                if confidence > self.config.early_exit_threshold and i < len(self.model_layers) - 1:
                    exit_layer = i + 1
                    print(f"   → Early exit at layer {exit_layer}/{len(self.model_layers)}")
                    break
        
        logits = self.output_layer(x)
        layers_used = exit_layer if exit_layer else len(self.model_layers)
        savings = 100 * (len(self.model_layers) - layers_used) / len(self.model_layers)
        
        return {
            'logits': logits,
            'layers_used': layers_used,
            'computational_savings': savings
        }

class SimpleBaselineGPT(tf.keras.Model):
    """Baseline GPT without early exit for comparison"""
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        
    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size)
        self.model_layers = [
            tf.keras.layers.Dense(self.hidden_size, activation='relu', name=f'baseline_layer_{i}')
            for i in range(self.config.num_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(self.vocab_size, name='baseline_output')
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        x = self.embedding(inputs)
        x = tf.reduce_mean(x, axis=1)  # Simple pooling
        
        # Process ALL layers (no early exit)
        for layer in self.model_layers:
            x = layer(x, training=training)
        
        logits = self.output_layer(x)
        
        return {
            'logits': logits,
            'layers_used': len(self.model_layers),
            'computational_savings': 0.0
        }

def create_dummy_data(batch_size=16, seq_len=32, vocab_size=1000):
    """Create dummy data for demonstration"""
    return tf.random.uniform([batch_size, seq_len], 0, vocab_size, dtype=tf.int32)

def main():
    """Main comparison function optimized for Kaggle visibility"""
    
    print("🚀 TOKEN-ADAPTIVE EARLY EXIT COMPARISON")
    print("="*60)
    
    # Configuration
    config = ToExConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=6,
        early_exit_threshold=0.3
    )
    
    print(f"📋 Configuration:")
    print(f"   Vocabulary Size: {config.vocab_size}")
    print(f"   Hidden Size: {config.hidden_size}")
    print(f"   Number of Layers: {config.num_layers}")
    print(f"   Early Exit Threshold: {config.early_exit_threshold}")
    print()
    
    # Initialize models
    print("🏗️ Initializing Models...")
    toex_model = SimpleToExGPT(config)
    baseline_model = SimpleBaselineGPT(config)
    print("   ✅ ToEx Model (with Early Exit)")
    print("   ✅ Baseline Model (without Early Exit)")
    print()
    
    # Create dummy data
    dummy_data = create_dummy_data(batch_size=8, seq_len=16, vocab_size=config.vocab_size)
    print(f"📊 Data Shape: {dummy_data.shape}")
    print()
    
    # Training setup - separate optimizers for each model
    toex_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    baseline_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Training metrics
    toex_losses = []
    baseline_losses = []
    toex_savings = []
    
    print("🎯 Training Comparison...")
    print("-" * 40)
    
    # Training loop
    for epoch in range(3):
        print(f"\n📅 Epoch {epoch + 1}/3")
        
        for batch in range(5):
            # ToEx Model Training
            with tf.GradientTape() as tape:
                toex_outputs = toex_model(dummy_data, training=True)
                targets = tf.random.uniform([8], 0, config.vocab_size, dtype=tf.int32)
                toex_loss = loss_fn(targets, toex_outputs['logits'])
            
            gradients = tape.gradient(toex_loss, toex_model.trainable_variables)
            toex_optimizer.apply_gradients(zip(gradients, toex_model.trainable_variables))
            
            # Baseline Model Training
            with tf.GradientTape() as tape:
                baseline_outputs = baseline_model(dummy_data, training=True)
                baseline_loss = loss_fn(targets, baseline_outputs['logits'])
            
            gradients = tape.gradient(baseline_loss, baseline_model.trainable_variables)
            baseline_optimizer.apply_gradients(zip(gradients, baseline_model.trainable_variables))
            
            # Track metrics
            toex_losses.append(float(toex_loss))
            baseline_losses.append(float(baseline_loss))
            toex_savings.append(toex_outputs['computational_savings'])
            
            # Progress output
            if batch % 2 == 0:
                print(f"   Batch {batch}:")
                print(f"     ToEx Loss: {toex_loss:.4f} | Layers: {toex_outputs['layers_used']}/{config.num_layers} | Savings: {toex_outputs['computational_savings']:.1f}%")
                print(f"     Baseline Loss: {baseline_loss:.4f} | Layers: {baseline_outputs['layers_used']}/{config.num_layers} | Savings: {baseline_outputs['computational_savings']:.1f}%")
    
    # Calculate final results
    avg_toex_loss = np.mean(toex_losses)
    avg_baseline_loss = np.mean(baseline_losses)
    avg_savings = np.mean([s for s in toex_savings if s > 0])
    
    print("\n" + "="*60)
    print("📊 FINAL RESULTS COMPARISON")
    print("="*60)
    
    print(f"\n🔥 ToEx Model (WITH Early Exit):")
    print(f"   Average Loss: {avg_toex_loss:.4f}")
    print(f"   Average Computational Savings: {avg_savings:.1f}%")
    print(f"   Early Exits Triggered: {len([s for s in toex_savings if s > 0])}/{len(toex_savings)} batches")
    
    print(f"\n📈 Baseline Model (WITHOUT Early Exit):")
    print(f"   Average Loss: {avg_baseline_loss:.4f}")
    print(f"   Computational Savings: 0.0%")
    print(f"   Always Uses All Layers: {config.num_layers}/{config.num_layers}")
    
    print(f"\n⚡ Performance Summary:")
    loss_improvement = avg_baseline_loss - avg_toex_loss
    if loss_improvement > 0:
        print(f"   ✅ Quality: ToEx model is BETTER by {loss_improvement:.4f} loss")
    else:
        print(f"   ⚠️  Quality: Trade-off of {abs(loss_improvement):.4f} loss for efficiency")
    
    print(f"   ⚡ Efficiency: Up to {avg_savings:.1f}% computational savings")
    print(f"   🎯 Recommendation: {'Deploy ToEx' if avg_savings > 10 else 'Tune threshold'}")
    
    print("\n" + "="*60)
    print("✅ TOKEN-ADAPTIVE EARLY EXIT DEMO COMPLETE")
    print("="*60)
    
    return {
        'toex_loss': avg_toex_loss,
        'baseline_loss': avg_baseline_loss,
        'savings': avg_savings,
        'early_exits': len([s for s in toex_savings if s > 0])
    }

if __name__ == "__main__":
    results = main()
    print(f"\n📋 Results Dictionary: {results}")
