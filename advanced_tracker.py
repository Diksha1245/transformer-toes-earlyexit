"""
ADVANCED Token-Adaptive Early Exit Parameter Optimization and Tracking System
"""

import json
import datetime
from typing import Dict, List, Any

class AdvancedParameterTracker:
    """Advanced parameter tracking and optimization system"""
    
    def __init__(self, log_file="advanced_toex_log.txt"):
        self.log_file = log_file
        self.experiments = []
        self.best_config = None
        self.best_score = float('-inf')
        
    def calculate_composite_score(self, results):
        """Calculate composite performance score"""
        toex_metrics = results.get('ToEx Model', {})
        baseline_metrics = results.get('Baseline Model', {})
        
        if not toex_metrics or not baseline_metrics:
            return 0.0
            
        # Weighted composite score
        loss_improvement = (baseline_metrics['avg_loss'] - toex_metrics['avg_loss']) / baseline_metrics['avg_loss']
        accuracy_preservation = max(0, 1 - abs(toex_metrics['avg_accuracy'] - baseline_metrics['avg_accuracy']))
        efficiency_gain = toex_metrics['avg_savings'] / 100.0
        exit_rate = toex_metrics['early_exit_rate'] / 100.0
        
        # Composite score (normalized 0-1)
        score = (
            0.3 * max(0, loss_improvement) +
            0.3 * accuracy_preservation +
            0.2 * efficiency_gain +
            0.2 * exit_rate
        )
        
        return score
        
    def log_experiment(self, config, results, notes=""):
        """Log detailed experiment results"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        composite_score = self.calculate_composite_score(results)
        
        experiment = {
            'timestamp': timestamp,
            'config': {
                'vocab_size': config.vocab_size,
                'hidden_size': config.hidden_size,
                'num_layers': config.num_layers,
                'num_heads': config.num_heads,
                'early_exit_threshold': config.early_exit_threshold,
                'max_candidates': config.max_candidates,
                'dropout_rate': config.dropout_rate
            },
            'results': results,
            'composite_score': composite_score,
            'notes': notes
        }
        
        self.experiments.append(experiment)
        
        # Update best configuration
        if composite_score > self.best_score:
            self.best_score = composite_score
            self.best_config = experiment
            
        self._write_to_file(experiment)
        
    def _write_to_file(self, experiment):
        """Write experiment to log file"""
        with open(self.log_file, "a", encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"📊 EXPERIMENT #{len(self.experiments)} - {experiment['timestamp']}\n")
            f.write(f"{'='*80}\n")
            
            # Parameters
            config = experiment['config']
            f.write(f"\n🔧 CONFIGURATION:\n")
            for key, value in config.items():
                f.write(f"   {key:20} = {value}\n")
                
            # Results
            f.write(f"\n📈 RESULTS:\n")
            for model, metrics in experiment['results'].items():
                f.write(f"   {model}:\n")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"     {metric:18} = {value:.4f}\n")
                    else:
                        f.write(f"     {metric:18} = {value}\n")
                f.write("\n")
                
            # Analysis
            f.write(f"🎯 ANALYSIS:\n")
            f.write(f"   Composite Score    = {experiment['composite_score']:.4f}\n")
            
            toex = experiment['results'].get('ToEx Model', {})
            baseline = experiment['results'].get('Baseline Model', {})
            
            if toex and baseline:
                loss_diff = baseline['avg_loss'] - toex['avg_loss']
                acc_diff = toex['avg_accuracy'] - baseline['avg_accuracy']
                
                f.write(f"   Loss Improvement   = {loss_diff:+.4f}\n")
                f.write(f"   Accuracy Change    = {acc_diff:+.4f}\n")
                f.write(f"   Exit Rate          = {toex['early_exit_rate']:.1f}%\n")
                f.write(f"   Computational Save = {toex['avg_savings']:.1f}%\n")
                f.write(f"   Avg Exit Time      = {toex['avg_exit_time']:.2f}ms\n")
                
            # Next optimization suggestions
            f.write(f"\n💡 OPTIMIZATION SUGGESTIONS:\n")
            self._generate_suggestions(experiment, f)
            
            if experiment['notes']:
                f.write(f"\n📝 NOTES: {experiment['notes']}\n")
                
    def _generate_suggestions(self, experiment, file_handle):
        """Generate parameter optimization suggestions"""
        config = experiment['config']
        toex_results = experiment['results'].get('ToEx Model', {})
        
        suggestions = []
        
        # Exit rate analysis
        exit_rate = toex_results.get('early_exit_rate', 0)
        if exit_rate < 20:
            new_threshold = max(0.05, config['early_exit_threshold'] - 0.05)
            suggestions.append(f"Lower exit threshold to {new_threshold:.2f} (current: {config['early_exit_threshold']:.2f})")
        elif exit_rate > 95:
            new_threshold = min(0.8, config['early_exit_threshold'] + 0.1)
            suggestions.append(f"Raise exit threshold to {new_threshold:.2f} for quality balance")
            
        # Model capacity analysis
        if toex_results.get('avg_loss', 10) > 6.5:
            suggestions.append(f"Increase hidden_size to {config['hidden_size'] + 64} for better representation")
            suggestions.append(f"Consider increasing num_layers to {config['num_layers'] + 2}")
            
        # Efficiency analysis
        savings = toex_results.get('avg_savings', 0)
        if savings < 30:
            suggestions.append("Consider more aggressive early exit conditions")
            
        for suggestion in suggestions[:3]:  # Limit to top 3
            file_handle.write(f"   • {suggestion}\n")
            
    def get_best_config(self):
        """Return the best performing configuration"""
        return self.best_config
        
    def get_optimization_history(self):
        """Return optimization history"""
        return self.experiments
        
    def suggest_next_config(self, base_config):
        """Suggest next configuration to try"""
        if not self.experiments:
            return base_config
            
        # Analyze trends and suggest improvements
        latest = self.experiments[-1]
        config = latest['config'].copy()
        
        exit_rate = latest['results'].get('ToEx Model', {}).get('early_exit_rate', 0)
        loss = latest['results'].get('ToEx Model', {}).get('avg_loss', 10)
        
        # Adaptive suggestions based on performance
        if exit_rate < 30:
            config['early_exit_threshold'] = max(0.05, config['early_exit_threshold'] - 0.03)
        
        if loss > 6.5:
            config['hidden_size'] = min(512, config['hidden_size'] + 32)
            
        return config

# Usage example function
def setup_advanced_tracking():
    """Setup advanced parameter tracking"""
    tracker = AdvancedParameterTracker()
    
    # Initialize log file with header
    with open(tracker.log_file, "w", encoding='utf-8') as f:
        f.write("ADVANCED TOKEN-ADAPTIVE EARLY EXIT OPTIMIZATION LOG\n")
        f.write("="*80 + "\n")
        f.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
        f.write("\nObjective: Optimize Token-Adaptive Early Exit parameters for:\n")
        f.write("• Maximum computational savings (target: >50%)\n")
        f.write("• Minimal accuracy loss (target: <5%)\n")
        f.write("• High early exit rate (target: 30-70%)\n")
        f.write("• Low inference latency\n")
        f.write("• Robust token-level exit patterns\n\n")
        
    return tracker

if __name__ == "__main__":
    # Demo usage
    tracker = setup_advanced_tracking()
    print("✅ Advanced parameter tracking system initialized")
    print(f"📝 Log file: {tracker.log_file}")
