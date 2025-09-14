# -*- coding: utf-8 -*-
# Enhanced debugging training script for DeepSeek V3 with attention output subspace

"""# subspace_decoder/scripts/train_debug.py

Enhanced Training Script with Comprehensive Debugging

This script includes comprehensive debugging features to help diagnose training issues:

Features:
- 🔍 Gradient norm monitoring with per-layer breakdown
- 📊 Loss tracking with moving averages and anomaly detection  
- 📈 Real-time plotting of training metrics
- 🧠 Activation monitoring for detecting NaN/explosion issues
- 💾 Weight update ratio tracking
- 🚨 Early stopping on problematic conditions
- 📝 Detailed debug reports and logs

Usage:
  python train_debug.py --config path/to/config.json --debug
  
  Optional flags:
  --debug-steps N          Log debug info every N steps (default: 10)
  --gradient-threshold X   Warning threshold for gradient norms (default: 10.0)
  --loss-threshold Y       Loss explosion threshold (default: 100.0)

Debug artifacts are saved to ./debug_artifacts/ including:
- Training plots updated every 50 steps
- Comprehensive debug report at end of training
- Console logs with real-time statistics

The debug trainer automatically enables:
- Gradient clipping (max_grad_norm=1.0)
- More frequent logging
- Enhanced monitoring hooks
"""


import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"              # older check some codepaths still honor
# Optional: if Keras 3 is on the system and ever gets touched, force non-TF backend
os.environ.setdefault("KERAS_BACKEND", "torch")

from transformers.utils import is_tf_available
print("TF available (Transformers thinks):", is_tf_available())  # should be False


print("Importing Packages...\n")

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
import time
import warnings
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)

from utils import summarize_parameters, format_size
# To disable a warning.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Make sure we can import modules from the decoder package
PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("PROJECT_ROOT", PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.shared_space_config import SharedSpaceDecoderConfig, get_config
from layers.task_heads import SharedSpaceDecoderForCausalLM


class DebugConfig:
    """Configuration class for debugging settings."""
    def __init__(self):
        # Gradient monitoring
        self.log_gradient_norms = True
        self.log_gradient_histograms = True
        self.gradient_norm_threshold = 10.0  # Log warning if grad norm exceeds this
        
        # Loss monitoring
        self.log_loss_components = True
        self.plot_loss_curves = True
        self.loss_explosion_threshold = 100.0  # Stop if loss exceeds this
        
        # Weight monitoring
        self.log_weight_stats = True
        self.log_weight_histograms = True
        self.weight_update_ratio_threshold = 0.1  # Warn if weight updates are too large
        
        # Activation monitoring
        self.log_activations = True
        self.log_activation_histograms = True
        self.nan_detection = True
        
        # Early stopping
        self.early_stop_on_nan = True
        self.early_stop_on_loss_explosion = True
        self.patience_steps = 100  # Stop if no improvement for this many steps
        
        # Debugging frequency
        self.debug_log_steps = 10  # Log debug info every N steps
        self.plot_update_steps = 50  # Update plots every N steps
        
        # Memory monitoring
        self.log_memory_usage = True
        
        # Save debug artifacts
        self.save_debug_plots = True
        self.debug_output_dir = "debug_artifacts"


class GradientMonitor:
    """Monitor gradient norms and statistics."""
    def __init__(self, config: DebugConfig):
        self.config = config
        self.gradient_norms = deque(maxlen=1000)
        self.layer_gradient_norms = defaultdict(lambda: deque(maxlen=1000))
        self.gradient_stats = []
        
    def log_gradients(self, model: nn.Module, step: int) -> Dict[str, float]:
        """Log gradient statistics for the model."""
        total_norm = 0.0
        param_count = 0
        layer_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
                
                # Track per-layer gradients
                layer_name = '.'.join(name.split('.')[:-1])  # Remove parameter name
                if layer_name not in layer_norms:
                    layer_norms[layer_name] = []
                layer_norms[layer_name].append(param_norm)
                
                # Check for NaN gradients
                if torch.isnan(param.grad).any():
                    warnings.warn(f"NaN gradient detected in {name} at step {step}")
                    
        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append(total_norm)
        
        # Aggregate layer norms
        for layer_name, norms in layer_norms.items():
            layer_norm = np.sqrt(sum(n**2 for n in norms))
            self.layer_gradient_norms[layer_name].append(layer_norm)
            
        # Create statistics
        stats = {
            'gradient_norm/total': total_norm,
            'gradient_norm/param_count': param_count,
            'gradient_norm/avg_per_param': total_norm / max(param_count, 1),
        }
        
        # Add layer-specific norms
        for layer_name, norms in self.layer_gradient_norms.items():
            if norms:
                stats[f'gradient_norm/{layer_name}'] = norms[-1]
                
        # Warning for high gradient norms (only print once per 10 steps to avoid spam)
        if total_norm > self.config.gradient_norm_threshold and step % 10 == 0:
            print(f"⚠️  High gradient norm: {total_norm:.4f} (threshold: {self.config.gradient_norm_threshold})")
            
        return stats


class LossMonitor:
    """Monitor loss statistics and detect anomalies."""
    def __init__(self, config: DebugConfig):
        self.config = config
        self.losses = deque(maxlen=1000)
        self.loss_components = defaultdict(lambda: deque(maxlen=1000))
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        
    def log_loss(self, loss: float, step: int, components: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Log loss and check for anomalies."""
        self.losses.append(loss)
        
        # Track loss components if provided
        if components:
            for name, value in components.items():
                self.loss_components[name].append(value)
                
        # Check for loss explosion
        if loss > self.config.loss_explosion_threshold:
            warnings.warn(f"Loss explosion detected: {loss:.4f} at step {step}")
            
        # Check for NaN loss
        if np.isnan(loss):
            warnings.warn(f"NaN loss detected at step {step}")
            
        # Track improvement
        if loss < self.best_loss:
            self.best_loss = loss
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1
            
        stats = {
            'loss/current': loss,
            'loss/best': self.best_loss,
            'loss/steps_without_improvement': self.steps_without_improvement,
        }
        
        # Add moving averages
        if len(self.losses) >= 10:
            stats['loss/ma_10'] = np.mean(list(self.losses)[-10:])
        if len(self.losses) >= 50:
            stats['loss/ma_50'] = np.mean(list(self.losses)[-50:])
            
        return stats


class WeightMonitor:
    """Monitor weight statistics and updates."""
    def __init__(self, config: DebugConfig):
        self.config = config
        self.previous_weights = {}
        self.weight_stats = []
        
    def log_weights(self, model: nn.Module, step: int) -> Dict[str, float]:
        """Log weight statistics."""
        stats = {}
        
        for name, param in model.named_parameters():
            if param.data is not None:
                weight_data = param.data
                
                # Basic statistics
                stats[f'weights/{name}/mean'] = weight_data.mean().item()
                stats[f'weights/{name}/std'] = weight_data.std().item()
                stats[f'weights/{name}/norm'] = weight_data.norm().item()
                
                # Check for NaN weights
                if torch.isnan(weight_data).any():
                    warnings.warn(f"NaN weights detected in {name} at step {step}")
                    
                # Track weight updates if we have previous weights
                if name in self.previous_weights:
                    weight_diff = weight_data - self.previous_weights[name]
                    update_norm = weight_diff.norm().item()
                    param_norm = weight_data.norm().item()
                    
                    if param_norm > 0:
                        update_ratio = update_norm / param_norm
                        stats[f'weight_updates/{name}/ratio'] = update_ratio
                        stats[f'weight_updates/{name}/norm'] = update_norm
                        
                        if update_ratio > self.config.weight_update_ratio_threshold:
                            print(f"⚠️  Large weight update in {name}: {update_ratio:.4f} (threshold: {self.config.weight_update_ratio_threshold})")
                            
                # Store current weights for next comparison
                self.previous_weights[name] = weight_data.clone().detach()
                
        return stats


class ActivationHook:
    """Hook to monitor activations during forward pass."""
    def __init__(self, name: str, config: DebugConfig):
        self.name = name
        self.config = config
        self.activations = deque(maxlen=100)
        
    def __call__(self, module, input, output):
        if isinstance(output, torch.Tensor):
            activation = output.detach()
            
            # Check for NaN activations
            if self.config.nan_detection and torch.isnan(activation).any():
                warnings.warn(f"NaN activation detected in {self.name}")
                
            # Store statistics
            stats = {
                'mean': activation.mean().item(),
                'std': activation.std().item(),
                'min': activation.min().item(),
                'max': activation.max().item(),
                'norm': activation.norm().item(),
            }
            self.activations.append(stats)


def check_bf16_support():
    """Check if BFloat16 is supported on the current hardware and PyTorch version."""
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. BFloat16 training requires CUDA.")
        return False
    
    # Check if the GPU supports BFloat16
    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
        print("✓ BFloat16 is supported on this hardware")
        return True
    
    # Fallback check for older PyTorch versions
    try:
        # Try to create a small BFloat16 tensor on GPU
        test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device='cuda')
        print("✓ BFloat16 is supported on this hardware")
        return True
    except Exception as e:
        print(f"Warning: BFloat16 not supported on this hardware: {e}")
        return False

class DebugTrainer(Trainer):
    """Enhanced Trainer with debugging capabilities."""
    def __init__(self, debug_config: DebugConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_config = debug_config
        self.gradient_monitor = GradientMonitor(debug_config)
        self.loss_monitor = LossMonitor(debug_config)
        self.weight_monitor = WeightMonitor(debug_config)
        self.activation_hooks = {}
        self.debug_step = 0
        
        # Create debug output directory
        os.makedirs(debug_config.debug_output_dir, exist_ok=True)
        
        # Setup activation hooks
        self._setup_activation_hooks()
        
    def _setup_activation_hooks(self):
        """Setup hooks to monitor activations."""
        if not self.debug_config.log_activations:
            return
            
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
                hook = ActivationHook(name, self.debug_config)
                self.activation_hooks[name] = hook
                module.register_forward_hook(hook)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step to add debugging."""
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
            
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            
        # Check for loss explosion before backpropagation
        if self._should_early_stop(loss.item()):
            print(f"Early stopping triggered at step {self.debug_step}")
            raise KeyboardInterrupt("Early stopping triggered by debug conditions")
        
        self.accelerator.backward(loss)
            
        # Debug logging
        if self.debug_step % self.debug_config.debug_log_steps == 0:
            self._log_debug_info(loss.item())
            
        self.debug_step += 1
        return loss.detach() / self.args.gradient_accumulation_steps
    
    def _log_debug_info(self, loss: float):
        """Log comprehensive debugging information."""
        debug_stats = {}
        
        # Log loss statistics
        if self.debug_config.log_loss_components:
            loss_stats = self.loss_monitor.log_loss(loss, self.debug_step)
            debug_stats.update(loss_stats)
            
        # Log gradient statistics
        if self.debug_config.log_gradient_norms:
            grad_stats = self.gradient_monitor.log_gradients(self.model, self.debug_step)
            debug_stats.update(grad_stats)
            
        # Log weight statistics (less frequently to avoid overhead)
        if self.debug_config.log_weight_stats and self.debug_step % 100 == 0:
            weight_stats = self.weight_monitor.log_weights(self.model, self.debug_step)
            debug_stats.update(weight_stats)
            
        # Log memory usage
        if self.debug_config.log_memory_usage and torch.cuda.is_available():
            debug_stats.update({
                'memory/allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'memory/cached_gb': torch.cuda.memory_reserved() / 1e9,
            })
            
        # Log activation statistics
        if self.debug_config.log_activations:
            for name, hook in self.activation_hooks.items():
                if hook.activations:
                    latest = hook.activations[-1]
                    for stat_name, value in latest.items():
                        debug_stats[f'activations/{name.replace(".", "_")}/{stat_name}'] = value
                        
        # Print important stats to console (only if there are issues)
        grad_norm = debug_stats.get('gradient_norm/total', 0)
        if grad_norm > self.debug_config.gradient_norm_threshold or loss > 15.0:
            print(f"🚨 Step {self.debug_step}: Loss={loss:.4f}, GradNorm={grad_norm:.4f}")
            if torch.cuda.is_available():
                print(f"   Memory: {debug_stats.get('memory/allocated_gb', 0):.2f}GB")
        elif self.debug_step % 50 == 0:  # Less frequent normal updates
            print(f"📊 Step {self.debug_step}: Loss={loss:.4f}, GradNorm={grad_norm:.4f}")
                
        # Log to wandb
        if debug_stats:
            wandb.log(debug_stats, step=self.debug_step)
            
        # Create plots
        if (self.debug_step % self.debug_config.plot_update_steps == 0 and 
            self.debug_config.plot_loss_curves):
            self._create_debug_plots()
    
    def _should_early_stop(self, loss: float) -> bool:
        """Check if training should be stopped early due to debug conditions."""
        # Stop on NaN loss
        if self.debug_config.early_stop_on_nan and np.isnan(loss):
            print(f"Early stopping: NaN loss detected at step {self.debug_step}")
            return True
            
        # Stop on loss explosion
        if (self.debug_config.early_stop_on_loss_explosion and 
            loss > self.debug_config.loss_explosion_threshold):
            print(f"Early stopping: Loss explosion detected ({loss:.4f}) at step {self.debug_step}")
            return True
            
        return False
    
    def _create_debug_plots(self):
        """Create and save debugging plots."""
        if not self.debug_config.save_debug_plots:
            return
            
        try:
            plt.style.use('seaborn-v0_8')
        except:
            # Fallback if seaborn style not available
            pass
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curve
        if self.loss_monitor.losses:
            axes[0, 0].plot(list(self.loss_monitor.losses))
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_yscale('log')
            
        # Gradient norms
        if self.gradient_monitor.gradient_norms:
            axes[0, 1].plot(list(self.gradient_monitor.gradient_norms))
            axes[0, 1].set_title('Gradient Norms')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Gradient Norm')
            axes[0, 1].set_yscale('log')
            
        # Memory usage
        if torch.cuda.is_available():
            memory_data = [torch.cuda.memory_allocated() / 1e9]
            axes[1, 0].plot(memory_data)
            axes[1, 0].set_title('GPU Memory Usage')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Memory (GB)')
            
        # Layer gradient norms
        if self.gradient_monitor.layer_gradient_norms:
            for layer_name, norms in list(self.gradient_monitor.layer_gradient_norms.items())[:5]:  # Top 5 layers
                if norms:
                    axes[1, 1].plot(list(norms), label=layer_name.split('.')[-1])
            axes[1, 1].set_title('Layer Gradient Norms (Top 5)')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            
        plt.tight_layout()
        plot_path = f"{self.debug_config.debug_output_dir}/debug_plots_step_{self.debug_step}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Debug plots saved to {plot_path}")
    
    def generate_debug_report(self):
        """Generate a comprehensive debug report."""
        report = []
        report.append("=" * 60)
        report.append("DEBUG TRAINING REPORT")
        report.append("=" * 60)
        
        # Loss statistics
        if self.loss_monitor.losses:
            losses = list(self.loss_monitor.losses)
            report.append(f"Loss Statistics (last {len(losses)} steps):")
            report.append(f"  Current: {losses[-1]:.4f}")
            report.append(f"  Best: {self.loss_monitor.best_loss:.4f}")
            report.append(f"  Average (last 10): {np.mean(losses[-10:]):.4f}")
            report.append(f"  Steps without improvement: {self.loss_monitor.steps_without_improvement}")
            report.append("")
        
        # Gradient statistics
        if self.gradient_monitor.gradient_norms:
            grad_norms = list(self.gradient_monitor.gradient_norms)
            report.append(f"Gradient Statistics (last {len(grad_norms)} steps):")
            report.append(f"  Current norm: {grad_norms[-1]:.4f}")
            report.append(f"  Average norm: {np.mean(grad_norms):.4f}")
            report.append(f"  Max norm: {np.max(grad_norms):.4f}")
            report.append("")
        
        # Layer gradient breakdown
        if self.gradient_monitor.layer_gradient_norms:
            report.append("Layer Gradient Norms (latest):")
            for layer_name, norms in self.gradient_monitor.layer_gradient_norms.items():
                if norms:
                    report.append(f"  {layer_name}: {norms[-1]:.4f}")
            report.append("")
        
        # Memory usage
        if torch.cuda.is_available():
            report.append("Memory Usage:")
            report.append(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            report.append(f"  Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed monitoring")
    parser.add_argument("--debug-steps", type=int, default=10, help="Log debug info every N steps")
    parser.add_argument("--gradient-threshold", type=float, default=10.0, help="Gradient norm warning threshold")
    parser.add_argument("--loss-threshold", type=float, default=100.0, help="Loss explosion threshold")
    return parser.parse_args()


def main(config_path: str, enable_debug: bool = True, debug_args: Optional[Dict] = None):
    """Run pre-training using the provided configuration path."""
    
    # Load configuration
    full_cfg, model_cfg = get_config(config_path)
    
    # Setup debug configuration
    debug_config = None
    if enable_debug:
        debug_config = DebugConfig()
        if debug_args:
            debug_config.debug_log_steps = debug_args.get('debug_steps', 10)
            debug_config.gradient_norm_threshold = debug_args.get('gradient_threshold', 10.0)
            debug_config.loss_explosion_threshold = debug_args.get('loss_threshold', 100.0)
        
        print("🔍 Debug mode enabled - monitoring for training issues")
        print(f"  📊 Logging every {debug_config.debug_log_steps} steps")
        print(f"  ⚠️  Gradient norm threshold: {debug_config.gradient_norm_threshold}")
        print(f"  💥 Loss explosion threshold: {debug_config.loss_explosion_threshold}")
        print(f"  📈 Debug plots: {debug_config.debug_output_dir}")
        print("  🚨 Will highlight problematic steps with 🚨")
        print("  📊 Normal progress shown every 50 steps")

    ptrain_cfg = full_cfg['pre_train']

    # Print out its shorthand name.
    print(full_cfg["shorthand"])

    # Initialize the optional stats dictionary so later assignments don't fail.
    if "stats" not in full_cfg:
        full_cfg["stats"] = {}
    
    # Validate mixed precision settings
    if ptrain_cfg["bf16"] and ptrain_cfg["fp16"]:
        raise ValueError("Cannot enable both bf16 and fp16 simultaneously. Please choose one.")
    
    # Check BFloat16 compatibility if enabled
    if ptrain_cfg["bf16"]:
        if not check_bf16_support():
            print("BFloat16 requested but not supported. Falling back to FP16.")
            ptrain_cfg["bf16"] = False
            ptrain_cfg["fp16"] = True
    
        # Disable torch.compile for debugging (causes issues with hooks)
    if ptrain_cfg["torch_compile"]:
            print("⚠️  Disabling torch.compile for debugging (incompatible with monitoring hooks)")
            ptrain_cfg["torch_compile"] = False

    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # gpt2 has no pad by default; use EOS for padding in causal LM
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
        
    # Verify vocab size matches
    assert model_cfg.vocab_size == tokenizer.vocab_size

    # Set random seed for reproducibility
    set_seed(ptrain_cfg["seed"])

    # Setup Weights & Biases
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"

    wandb_api_key = os.environ.get("WANDB_API_KEY")

    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    # ======================
    #    Prepare Dataset
    # ======================
    

        dataset_name = ptrain_cfg["dataset_name"]
        dataset_config = ptrain_cfg["dataset_config"]
        
        
            # Original logic for wikitext and other datasets
            dataset = load_dataset(dataset_name, dataset_config)

        print(dataset)
        
        block_size = ptrain_cfg["max_seq_length"]
        eos_id = tokenizer.eos_token_id
        
        # 1) Tokenize without truncation/padding
        def tokenize_function(examples):
            # add_special_tokens=False keeps things raw; we'll insert EOS between docs
            return tokenizer(
                examples["text"],
                add_special_tokens=False,
            )
        
        # 2) Group into contiguous blocks (concat + chunk)
        def group_texts(examples):
            # Flatten and insert EOS between documents to avoid cross-article bleed
            input_ids = []
            for ids in examples["input_ids"]:
                if len(ids) > 0:
                    input_ids.extend(ids)
                # add an EOS fencepost between docs
                input_ids.append(eos_id)
        
            # Drop the trailing partial block so every example is full length
            total_length = (len(input_ids) // block_size) * block_size
            input_ids = input_ids[:total_length]
        
            # Split into equal blocks
            result_input_ids = [input_ids[i:i + block_size] for i in range(0, total_length, block_size)]
            # Labels are next-token targets; Trainer/model will do the shift
            return {
                "input_ids": result_input_ids,
                "labels": [ids.copy() for ids in result_input_ids],
                # Optional attention masks (all ones because no padding)
                "attention_mask": [[1] * block_size for _ in result_input_ids],
            }
        
        # Tokenize
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=8,
            remove_columns=dataset["train"].column_names,  # drop raw "text"
        )
        
        # Concatenate + chunk
        tokenized = tokenized.map(
            group_texts,
            batched=True,
            num_proc=8,
        )
    
    # Use a simple collator; we already created labels and have no pads
    from transformers import default_data_collator
    data_collator = default_data_collator


    # ========================
    #    Initialize Model
    # ========================

    print("Initializing model...")

    model = SharedSpaceDecoderForCausalLM(model_cfg)

    # ================================
    #       Review Configuration
    # ================================

    # Display architecture
    print(model)

    print("\n======== Model ========")
    print(model_cfg)

    print("\n======== Pre-Train ========")
    print(json.dumps(ptrain_cfg, indent=2))

    # Calculate and display effective batch size
    device_batch_size = ptrain_cfg["train_batch_size"]
    gradient_accumulation_steps = ptrain_cfg["gradient_accumulation_steps"]
    effective_batch_size = device_batch_size * gradient_accumulation_steps
    
    print(f"\n======== Batch Size Configuration ========")
    print(f"Device batch size: {device_batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")

    print("=============================\n")

    """## Parameter Summary"""

    print("\n======== Parameters ========")

    ## Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The model has {:} different named parameters.\n'.format(len(params)))

    total_params = 0
    for p_name, p in params:
        total_params += p.numel()

    full_cfg["stats"]["total_elements"] = format_size(total_params)

    print(f"Total elements: {full_cfg['stats']['total_elements']}\n")

    # Display a full parameter breakdown using the shared utility
    summarize_parameters(model)

    # ========================================
    #   Format Settings for WandB Run Name
    # ========================================

    # Format the cfg learning rate as a scientific notation string like 5e-4
    lr_str = '{:.0e}'.format(ptrain_cfg['learning_rate'])

    # Attention configuration

    ptrain_cfg["run_name"] = full_cfg["stats"]["total_elements"] + " - " + full_cfg["shorthand"]

    print(ptrain_cfg["run_name"])

    """## wandb and TrainingArguments"""

    wandb.init(
        project=ptrain_cfg["wandb_project"],
        name=ptrain_cfg["run_name"],
        config=full_cfg
    )

    # ===============================
    #       Training Arguments
    # ===============================

    training_args = TrainingArguments(
        output_dir=ptrain_cfg["output_dir"],

        per_device_train_batch_size=ptrain_cfg["train_batch_size"],
        per_device_eval_batch_size=ptrain_cfg["eval_batch_size"],
        gradient_accumulation_steps=ptrain_cfg["gradient_accumulation_steps"],

        bf16=ptrain_cfg["bf16"],
        fp16=ptrain_cfg["fp16"],

        learning_rate=ptrain_cfg["learning_rate"],
        max_steps=ptrain_cfg["num_train_steps"], 

        # Debug-specific gradient clipping (helps with exploding gradients)
        max_grad_norm=1.0 if enable_debug else ptrain_cfg.get("max_grad_norm", None),
        
        # The dataloader is a bottleneck without these.
        dataloader_num_workers=ptrain_cfg["num_workers"],
        dataloader_pin_memory=ptrain_cfg["pin_memory"],
        # The prefetch factor didn't appear to help.
        #dataloader_prefetch_factor = ptrain_cfg["prefetch_factor"],

        weight_decay=ptrain_cfg["weight_decay"],  

        # Learning rate warmup (10% of total steps)
        warmup_steps=int(0.1 * ptrain_cfg["num_train_steps"]),  
        lr_scheduler_type="linear",  # Linear warmup then decay

        # Evaluate every 2,000 steps
        # Note: Recent versions of Trainer changed the name from 
        # `evaluation_strategy` to `eval_strategy`.
        batch_eval_metrics = True, # To avoid OOM
        eval_strategy="steps",
        eval_steps=ptrain_cfg["eval_steps"],
        eval_accumulation_steps=4,  # Process eval in smaller chunks to save memory

        logging_steps=10 if enable_debug else 50,  # More frequent logging in debug mode
        metric_for_best_model="eval_loss",
        save_steps=2000,
        save_total_limit=2,           # Optional: keeps last 2 checkpoints
        save_strategy="steps",
        report_to=["wandb"],
        
        run_name=ptrain_cfg["run_name"],
        
        remove_unused_columns=False,  # Optional: avoid dropping custom model inputs
    )

    print(training_args)

    import numpy as np

    class PerplexityMetric:
        """
        A stateful class to compute perplexity in a batch-wise manner to avoid OOM.
        Similar to the MLMAccuracyMetric from the encoder training.
        """
        def __init__(self):
            # Initialize state variables to store running totals
            self.total_loss = 0.0
            self.total_tokens = 0

        def __call__(self, eval_pred, compute_result=False):
            """
            This method will be called by the Trainer.
            """
            predictions, labels = eval_pred

            # For causal LM, we compute perplexity
            # Shift predictions and labels for next token prediction
            shift_logits = predictions[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Create a mask for valid tokens (not padding, typically -100)
            mask = shift_labels != -100
            
            if mask.sum() > 0:  # Only compute if there are valid tokens
                # Compute loss only on valid tokens
                loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                batch_loss = loss_fct(shift_logits[mask], shift_labels[mask])
                
                # Add to running totals
                self.total_loss += batch_loss.item()
                self.total_tokens += mask.sum().item()

            # If this is the final call after all batches are processed
            if compute_result:
                # Avoid division by zero
                if self.total_tokens == 0:
                    avg_loss = 0.0
                    perplexity = float('inf')
                else:
                    avg_loss = self.total_loss / self.total_tokens
                    perplexity = np.exp(avg_loss)

                # Prepare the final metrics dictionary
                metrics = {
                    "perplexity": perplexity,
                    "loss": avg_loss,
                }

                # Reset state for the next evaluation run
                self.total_loss = 0.0
                self.total_tokens = 0

                return metrics

            # For intermediate calls, return an empty dict
            return {}

    # Instantiate your stateful metric computer
    perplexity_metric = PerplexityMetric()

    # ===============================
    #           Trainer
    # ===============================
    if enable_debug and debug_config:
        trainer = DebugTrainer(
            debug_config=debug_config,
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            compute_metrics=perplexity_metric,
            processing_class=tokenizer,
            data_collator=data_collator,
        )
        print("🔍 Using DebugTrainer with enhanced monitoring")
    else:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=perplexity_metric,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
        print("Using standard Trainer")

    """## Loop"""

    # =====================
    #     Run Training
    # =====================

    # Do inside a try/finally so that if the run aborts, we still call wandb.finish().
    try:
        trainer.train()

        metrics = trainer.evaluate()

        wandb.log(metrics)

        # Store wandb ids into the config.
        full_cfg["pre_train"]["run_id"] = wandb.run.id
        full_cfg["pre_train"]["run_url"] = wandb.run.url
        full_cfg["pre_train"]["run_name"] = wandb.run.name

        # Save the best checkpoint.
        full_cfg["pre_train"]["best_checkpoint"] = trainer.state.best_model_checkpoint

        # Save the json back to disk
        with open(ptrain_cfg["output_dir"] + "/full_config.json", "w") as f:
            json.dump(full_cfg, f, indent=2)
   
        # Generate debug report if in debug mode
        if enable_debug and hasattr(trainer, 'generate_debug_report'):
            debug_report = trainer.generate_debug_report()
            print("\n" + debug_report)
            
            # Save debug report to file
            debug_report_path = os.path.join(ptrain_cfg["output_dir"], "debug_report.txt")
            with open(debug_report_path, "w") as f:
                f.write(debug_report)
            print(f"Debug report saved to {debug_report_path}")
            
            # Print summary of issues detected
            if hasattr(trainer, 'gradient_monitor') and trainer.gradient_monitor.gradient_norms:
                high_grad_steps = sum(1 for norm in trainer.gradient_monitor.gradient_norms 
                                    if norm > trainer.debug_config.gradient_norm_threshold)
                total_steps = len(trainer.gradient_monitor.gradient_norms)
                print(f"\n📊 Training Summary:")
                print(f"  Steps with high gradient norms: {high_grad_steps}/{total_steps} ({high_grad_steps/total_steps*100:.1f}%)")
                print(f"  Max gradient norm: {max(trainer.gradient_monitor.gradient_norms):.4f}")
                print(f"  Final loss: {trainer.loss_monitor.losses[-1]:.4f}")
                if high_grad_steps > total_steps * 0.1:  # More than 10% of steps
                    print("  ⚠️  Consider: Lower learning rate, gradient clipping, or model initialization")

    finally:
        # End the wandb run.
        wandb.finish()

    
if __name__ == "__main__":
    args = parse_args()
    
    # Prepare debug arguments
    debug_args = {
        'debug_steps': args.debug_steps,
        'gradient_threshold': args.gradient_threshold,
        'loss_threshold': args.loss_threshold,
    }
    
    main(args.config, enable_debug=args.debug, debug_args=debug_args)
