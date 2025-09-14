#!/usr/bin/env python3
"""
Example script showing how to use the debug training features.

This demonstrates various debugging scenarios and configurations.
"""

import subprocess
import sys
from pathlib import Path

def run_debug_training():
    """Run training with debug mode enabled."""
    
    # Example: Basic debug mode
    print("🔍 Running training with basic debug mode...")
    cmd = [
        sys.executable, "train_debug.py",
        "--config", "../configs/gpt-2_mla.json",  # Update path as needed
        "--debug",
        "--debug-steps", "5",  # Log every 5 steps for intensive debugging
        "--gradient-threshold", "5.0",  # Lower threshold for early detection
        "--loss-threshold", "50.0"  # Lower threshold for quicker detection
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Debug training completed successfully!")
        print("Debug artifacts saved to ./debug_artifacts/")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Debug training failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        
        # Check if it's an early stopping condition
        if "Early stopping" in e.stderr:
            print("🛑 Training was stopped early due to debug conditions.")
            print("Check the debug report and plots for more information.")
        
    except FileNotFoundError:
        print("❌ train_debug.py not found. Make sure you're in the correct directory.")

def show_debug_features():
    """Display available debug features."""
    
    features = """
    🔍 AVAILABLE DEBUG FEATURES:
    
    📊 Real-time Monitoring:
    - Gradient norms (total and per-layer)
    - Loss statistics with moving averages
    - Weight update ratios
    - Activation statistics
    - GPU memory usage
    
    📈 Visualizations:
    - Loss curves with log scale
    - Gradient norm trends
    - Memory usage plots
    - Layer-wise gradient breakdown
    
    🚨 Early Stopping Conditions:
    - NaN detection in loss/gradients/activations
    - Loss explosion detection
    - Gradient explosion warnings
    - Excessive weight updates
    
    📝 Debug Artifacts:
    - ./debug_artifacts/debug_plots_step_N.png (every 50 steps)
    - ./debug_artifacts/debug_report.txt (final report)
    - Enhanced wandb logs with debug metrics
    - Console output with step-by-step statistics
    
    ⚙️  Automatic Enhancements in Debug Mode:
    - Gradient clipping enabled (max_grad_norm=1.0)
    - More frequent logging (every 10 steps vs 50)
    - Comprehensive monitoring hooks
    - Early stopping safeguards
    """
    
    print(features)

if __name__ == "__main__":
    print("Debug Training Example")
    print("=" * 50)
    
    # Show available features
    show_debug_features()
    
    print("\n" + "=" * 50)
    print("To run debug training, uncomment the line below:")
    print("# run_debug_training()")
    
    # Uncomment to actually run training
    # run_debug_training()
