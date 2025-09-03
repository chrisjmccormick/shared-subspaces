# -*- coding: utf-8 -*-
# Fine-tune (SST-2) via LM-head label words, optional LoRA

# ============================================================================
# IMPORTANT: STYLE GUIDE
# This is a **prototype**. Don't polish it prematurely.
# 
# This means:
# - Prioritize legibility over re-use and robustness. 
#     - DO NOT use get_attr. All settings are required. If something is missing,
#       that was a programmer error, and we want the code to **crash**. 
#     - Do not clutter the code with "raise ValueError", that's premature.
#     - We can add those things in later when the code is mature. 
# 
# Minimize boiler plate and safety checks.
# Don't create silent bugs by implementing fallbacks.
# Comment heavily. 
# Avoid packing operations into dense, single lines.
# Prefer flat code to small helper functions. Factorization is for mature code,
# not prototypes. It requires the developer to invest time and energy into 
# becoming familiar with what the function does and how to use it.
#
# Note that some of the existing code is agent-written and doesn't currently
# follow these guidelines.
# ============================================================================


import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import wandb
from collections import Counter

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from utils import summarize_parameters, format_size

# Project import path (same pattern as your train.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layers.patch_o_proj import load_checkpoint_state_dict, load_and_patch_model, Variant

from transformers import DeepseekV3Config, DeepseekV3ForCausalLM 
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


# Setup Weights & Biases
if "WANDB_MODE" not in os.environ:
    os.environ["WANDB_MODE"] = "offline"

wandb_api_key = os.environ.get("WANDB_API_KEY")

if wandb_api_key:
    wandb.login(key=wandb_api_key)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to JSON config")
    return ap.parse_args()


def _bf16_ok():
    return torch.cuda.is_available() and (
        getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        or _try_bf16_tensor()
    )

def _try_bf16_tensor():
    try:
        _ = torch.tensor([1.0], dtype=torch.bfloat16, device="cuda")
        return True
    except Exception:
        return False

def build_label_vocab(tokenizer, ft_cfg):
    # Default label words if not provided
    label_words = ft_cfg.get("label_words") or {"0": " negative", "1": " positive"}
    lw0 = label_words.get("0", " negative")
    lw1 = label_words.get("1", " positive")

    toks0 = tokenizer.tokenize(lw0)
    toks1 = tokenizer.tokenize(lw1)
    
    if len(toks0) != 1 or len(toks1) != 1:
        raise ValueError(
            f"Label words must be single tokens for this script. Got: "
            f"{lw0} -> {toks0}, {lw1} -> {toks1}"
        )
    id0 = tokenizer.convert_tokens_to_ids(toks0[0])
    id1 = tokenizer.convert_tokens_to_ids(toks1[0])

    return {0: lw0, 1: lw1}, {0: id0, 1: id1}


def make_prompt_template(ft_cfg):
    # You can override via fine_tune.prompt_template in config
    default_prompt = (
        " Examples of movie review sentiment labeled as ' positive' or ' negative':\n"
        " This movie was awful. - negative\n"
        " I absolutely loved this film. - positive\n"
        " {sentence} -{label_word}"
    )
    prompt = ft_cfg.get("prompt_template", default_prompt)
        
    return prompt


def map_with_prompt(ds, tokenizer, label_val_to_word, label_val_to_token_id, prompt_template, max_seq_length=128):
    def add_prompt(ex):
        lw = label_val_to_word[ex["label"]]
        ex["labeled_sentence"] = prompt_template.format(sentence=ex["sentence"], label_word=lw)
        ex["label_id"] = label_val_to_token_id[ex["label"]]
        return ex

    ds = ds.map(add_prompt)

    def tok_and_targets(batch):
        enc = tokenizer(
            batch["labeled_sentence"],
            padding=False,  # We'll handle padding in the collator
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]

        labels = []
        label_pos = []
                
        # Find the label word position (should be the last non-pad token)
        for i, (ids, am) in enumerate(zip(input_ids, attn)):
            # Find the last non-pad token position
            last_non_pad = len(am) - 1 - am[::-1].index(1)
            
            # Create labels array with -100 everywhere except the label position
            lab = [-100] * len(ids)
            
            # Ensure the label position is within bounds
            if last_non_pad < len(ids):
                lab[last_non_pad] = batch["label_id"][i]
                label_pos.append(last_non_pad)
            else:
                # Fallback: use the last position
                lab[-1] = batch["label_id"][i]
                label_pos.append(len(ids) - 1)
            
            # Debug: Print label positioning for first few examples (only for small batches)
            if i < 3 and len(batch["labeled_sentence"]) <= 32:
                print(f"  Example {i}: last_non_pad={last_non_pad}, label_pos={label_pos[-1]}, label_id={batch['label_id'][i]}")
                print(f"    Last few tokens: {ids[-5:] if len(ids) >= 5 else ids}")
                print(f"    Last few labels: {lab[-5:] if len(lab) >= 5 else lab}")
            
            labels.append(lab)

        enc["labels"] = labels
        enc["label_pos"] = label_pos
        enc["true_label_ids"] = batch["label_id"]
        return enc

    cols = ["input_ids", "attention_mask", "labels", "label_pos", "true_label_ids"]
    ds = ds.map(tok_and_targets, batched=True, remove_columns=ds["train"].column_names)
    ds.set_format(type="torch", columns=cols)
    return ds


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = json.load(f)

    full_cfg = cfg
    
    model_cfg = cfg["model"]
    ft = cfg.get("fine_tune", {})
    ptrain_cfg = cfg["pre_train"]
    seed = ft.get("seed", 42)
    set_seed(seed)

    # Tokenizer selection
    tok_name = ft.get("tokenizer_name_or_path")  # recommended
    if tok_name is None:
        # Fallback heuristic by vocab size
        tok_name = "gpt2" if model_cfg.get("vocab_size", 0) <= 60000 else "deepseek-ai/DeepSeek-V3"
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Label words & prompt
    label_val_to_word, label_val_to_token_id = build_label_vocab(tokenizer, ft)
    prompt_template = make_prompt_template(ft)
    
    print(f"Label words: {label_val_to_word}")
    print(f"Label token IDs: {label_val_to_token_id}")
    print(f"Prompt template: {repr(prompt_template)}")

    # Dataset
    from datasets import load_dataset, DatasetDict
    from torch.utils.data import Subset
    
    # Load SST-2 and keep only labeled splits
    ds_all = load_dataset("glue", "sst2")
    ds = DatasetDict({k: v for k, v in ds_all.items() if k in ("train", "validation")})
    
    # Safety guard: filter any unlabeled rows (label == -1)
    ds = ds.filter(lambda ex: ex["label"] != -1)

    max_seq_length = ft.get("max_seq_length", 128)
    
    ds = map_with_prompt(ds, tokenizer, label_val_to_word, label_val_to_token_id, prompt_template, max_seq_length)
    
    # Because the SST-2 dataset does not provide labels for the test samples,
    # We'll set aside 10% of the training data to use for validation during 
    # training, (to assess overfitting and determine our best checkpoint)
    # and then use the provided "validation" set as our test set.
    train_split = int(0.9 * len(ds["train"]))
    train_dataset = Subset(ds["train"], range(train_split))
    val_dataset = Subset(ds["train"], range(train_split, len(ds["train"])))
    test_dataset = ds["validation"]
    
    print(f"\nDataset splits:")
    print(f"  Training set: {len(train_dataset)} examples")
    print(f"  Validation set: {len(val_dataset)} examples") 
    print(f"  Test set: {len(test_dataset)} examples")
    
    # ========================
    #    Initialize Model
    # ========================
    print("Initializing model...")
    
    # Identify the added parameter names by looking at the safetensors .json directly
    ckpt_dict = load_checkpoint_state_dict(full_cfg['pre_train']['best_checkpoint'])

    # Load the model and patch its implementation and weights.
    model = load_and_patch_model(model_cfg, ckpt_dict)

    # Optional LoRA
    lora_cfg = ft.get("lora", {})
    if lora_cfg.get("enabled", False):
        if not PEFT_AVAILABLE:
            raise RuntimeError("PEFT not available. Install `peft` to use LoRA.")
        peft_config = LoraConfig(
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            bias=lora_cfg.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    # ================================
    #       Review Configuration
    # ================================
    # While we're prototyping, printing out these architecture details (and skimming them 
    # to sanity check) is a good way to confirm that we're running the intended experiment,
    # as well as a good reference to go back to if we suspect something was wrong.
    
    # Display architecture
    print(model)

    print("\n======== Model ========")
    print(json.dumps(model_cfg, indent=2))

    print("\n======== Pre-Train ========")
    print(json.dumps(ptrain_cfg, indent=2))

    print("=============================\n")

    """## Parameter Summary"""

    print("\n======== Parameters ========")

    ## Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The model has {:} different named parameters.\n'.format(len(params)))

    total_params = 0
    for p_name, p in params:
        total_params += p.numel()

    print(f"Total elements: {total_params}\n")

    # Display a full parameter breakdown using the shared utility
    summarize_parameters(model)

    
    # Precision / compile
    bf16 = ft.get("bf16", True) and _bf16_ok()
    fp16 = ft.get("fp16", False) if not bf16 else False

    # Custom collator to handle variable-length sequences properly
    import torch.nn.functional as F
    
    class CustomDataCollator(DataCollatorForLanguageModeling):
        def __call__(self, features):
            # Expect 1-D LongTensors for each example
            input_ids      = [f["input_ids"]      for f in features]
            attention_mask = [f["attention_mask"] for f in features]
            labels         = [f["labels"]         for f in features]
    
            # lengths
            lengths = [ids.size(0) for ids in input_ids]
            max_len = max(lengths)
    
            # optional round-up for efficiency
            m = getattr(self, "pad_to_multiple_of", None)
            if m:
                max_len = ((max_len + m - 1) // m) * m
    
            pad_id = self.tokenizer.pad_token_id
    
            def pad_1d(x, tgt_len, value):
                pad = tgt_len - x.size(0)
                return x if pad == 0 else F.pad(x, (0, pad), value=value)
    
            input_ids      = [pad_1d(x, max_len, pad_id) for x in input_ids]
            attention_mask = [pad_1d(x, max_len, 0)      for x in attention_mask]
            labels         = [pad_1d(x, max_len, -100)   for x in labels]
    
            batch = {
                "input_ids":      torch.stack(input_ids, dim=0),
                "attention_mask": torch.stack(attention_mask, dim=0),
                "labels":         torch.stack(labels, dim=0),
            }
            return batch

    
    # Use custom collator
    collator = CustomDataCollator(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8  # Ensure consistent padding for efficiency
    )

    # Output dir
    run_name = cfg.get("pre_train", {}).get("run_name", "pretrain")
    out_dir = ft.get("output_dir", f"checkpoints/finetune_sst2_{run_name}")

    # Steps vs epochs
    max_steps = ft.get("max_steps")  # preferred for speed/consistency

    # Warmup
    warmup_ratio = ft.get("warmup_ratio", 0.1)
    warmup_steps = ft.get("warmup_steps")  # override if provided

    # Metrics: accuracy on the label token using LM head
    pos_id = label_val_to_token_id[1]
    neg_id = label_val_to_token_id[0]

    class AccuracyMetric:
        """
        A stateful class to compute accuracy in a batch-wise manner to avoid OOM.
        Similar to the PerplexityMetric from the pre-training script.
        """
        def __init__(self):
            # Initialize state variables to store running totals
            self.correct = 0
            self.total = 0

        def __call__(self, eval_pred, compute_result=False):
            """
            This method will be called by the Trainer.
            """
            predictions, labels = eval_pred
            
            # preds: [bsz, seq, vocab]; labels: [bsz, seq] with one position != -100
            for i in range(labels.shape[0]):
                # position of our supervised label token
                # Use torch operations instead of numpy to keep on GPU
                poss = torch.where(labels[i] != -100)[0]
                if len(poss) == 0:
                    continue
                p = poss[0].item()  # Convert to Python int
                if p - 1 < 0:
                    continue
                logits = predictions[i, p - 1]  # LM predicts token at t from logits[t-1]
                # compare only our two label tokens
                pred_token = pos_id if logits[pos_id] > logits[neg_id] else neg_id
                true_token = labels[i, p].item()  # Convert to Python int
                self.correct += int(pred_token == true_token)
                self.total += 1

            # If this is the final call after all batches are processed
            if compute_result:
                acc = (self.correct / max(self.total, 1)) * 100.0
                
                # Prepare the final metrics dictionary
                metrics = {"accuracy": acc}

                # Reset state for the next evaluation run
                self.correct = 0
                self.total = 0

                return metrics

            # For intermediate calls, return an empty dict
            return {}

    # Instantiate your stateful metric computer
    accuracy_metric = AccuracyMetric()

    # ========================================
    #   Format Settings for WandB Run Name
    # ========================================

    wandb.init(
        project="decoder-finetune-sst2",
        name=ft.get("run_name", f"ft-sst2-{run_name}"),
        config=full_cfg
    )
    
    # Training args
    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=ft["train_batch_size"],
        per_device_eval_batch_size=ft["eval_batch_size"],
        gradient_accumulation_steps=ft.get("gradient_accumulation_steps", 1),
        learning_rate=ft.get("learning_rate", 1e-4),
        weight_decay=ft.get("weight_decay", 0.01),
        bf16=bf16,
        fp16=fp16,
        # TODO - Not planning to use this for now. Also, torch_compile auto-sets to true
        #        if you specify (the backend?).
        #torch_compile=ft.get("torch_compile", True),
        #torch_compile_backend=ft.get("torch_compile_backend", "inductor"),
        #torch_compile_mode=ft.get("torch_compile_mode", "default"),
        eval_strategy="steps",
        eval_steps=ft.get("eval_steps", 100),
        save_strategy=ft.get("save_strategy", "steps"),
        save_steps=ft.get("save_steps", 500) if ft.get("save_strategy", "steps") != "no" else None,
        save_total_limit=ft.get("save_total_limit", 2),
        logging_steps=ft.get("logging_steps", 20),
        report_to=["wandb"] if ft.get("report_to_wandb", True) else [],
        run_name=ft.get("run_name", f"ft-sst2-{run_name}"),
        remove_unused_columns=False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        max_steps=max_steps,
        seed=seed,
        # Memory optimization settings from pre-training script
        batch_eval_metrics=True,  # To avoid OOM
        eval_accumulation_steps=4,  # Process eval in smaller chunks to save memory
    )

    # Print out the arguments--good way to sanity check things.
    print(targs)
    
    # Warmup resolution
    if warmup_steps is None:
        # ~10% by default
        targs.warmup_steps = int(warmup_ratio * max_steps)
    else:
        targs.warmup_steps = warmup_steps

    # Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=accuracy_metric,
    )

    # ======================
    #        Training
    # ======================
    # We'll use a try / finally block to ensure wandb.finish() is called.
    try:
        trainer.train()

        # Evaluate the trained model on the validation set
        print("\n--- Evaluating on Validation Set ---")
        val_results = trainer.evaluate()
        print(f"Validation Accuracy: {val_results['eval_accuracy']:.4f}")

        # Evaluate the trained model on the test set
        print("\n--- Evaluating on Test Set ---")
        test_results = trainer.predict(test_dataset)
        print(f"Test Accuracy: {test_results.metrics['test_accuracy']:.4f}")

        # Record the fine-tuning run with the checkpoint.
        full_cfg["fine_tune"]["run_id"] = wandb.run.id
        full_cfg["fine_tune"]["run_url"] = wandb.run.url

        # Save the json back to disk
        with open(os.path.join(out_dir, "full_config.json"), "w") as f:
            json.dump(full_cfg, f, indent=2)
   
    # Ensure we get to call `finish`, even if training is interrupted.
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
