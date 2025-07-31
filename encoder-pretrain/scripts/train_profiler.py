import json
import argparse
from pathlib import Path
import sys
import os

import torch
from torch import profiler as torch_profiler

import wandb
from datasets import load_dataset
import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# This file exists in the 'scripts' subdirectory, go up a level to find 'models'.
from models.custom_bert import SubspaceBertForMaskedLM, SubspaceBertConfig
from utils import summarize_parameters, format_size

# Make sure we can import modules from the encoder-pretrain package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Disable tokenizer parallelism since we're using num_workers > 0.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config")
    return parser.parse_args()

"""Helper utilities are imported from utils."""


from torch.utils.data import DataLoader


class CustomTrainer(Trainer):
    """Trainer with configurable dataloaders and cheaper logging."""

    def __init__(self, *args, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config or {}
 
    def get_train_dataloader(self):
        """Return DataLoader using values from the config."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,

            num_workers=self.config.get("pre_train", {}).get("num_workers", 4),
            pin_memory=self.config.get("pre_train", {}).get("pin_memory", True),
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.config.get("pre_train", {}).get("num_workers", 4),
            pin_memory=self.config.get("pre_train", {}).get("pin_memory", True),
        )

    '''
    def training_step(self, model, inputs, num_samples):
        """Detach loss and only convert to scalar on logging steps."""
        loss = super().training_step(model, inputs, num_samples)

        if (
            self.args.logging_steps > 0
            and (self.state.global_step + 1) % self.args.logging_steps == 0
        ):
            # Accessing the scalar triggers a sync, so do it sparingly
            self.loss_scalar = loss.detach().item()

        return loss
    '''

def main():
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Configs are now divided into "model" and "pre_train" sections. Fail fast if
    # these aren't present so that outdated configs cause a clear error.
    assert "model" in config and "pre_train" in config, "Config missing required sections"

    # Initialize the optional stats dictionary so later assignments don't fail.
    if "stats" not in config:
        config["stats"] = {}

    # Default max_steps for profiling; keeps runs short unless overridden
    config["pre_train"]["max_steps"] = 200 #config["pre_train"].get("max_steps", 50)

    print("Transformers version:", transformers.__version__)  # Helpful sanity check

    # Set random seed for reproducibility
    set_seed(config["pre_train"].get("seed", 42))

    # Setup Weights & Biases
    # Use offline mode if the API key isn't in the environment variables.
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"
    
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    
    # Use the standard BERT tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config["model"]["vocab_size"] = tokenizer.vocab_size
    
    
    dataset = load_dataset(
        config["pre_train"]["dataset_name"],
        config["pre_train"]["dataset_config"]
    )
    

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=config["pre_train"]["max_seq_length"]
        )

    tokenized = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"] # Comment this
    )

    # This DataCollator:
    # - Pads examples in a batch to the same length
    # - Applies BERT's Masked Language Modeling, replacing random tokens with 
    #   [MASK]. Enabled by setting `mlm_probability`.
    # - Returns input_ids, labels, and attention_mask for MLM training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=config["pre_train"]["mlm_probability"]  # Default is 15%
    )

    #model = CustomBertForMaskedLM.from_config(config)

    bert_config = SubspaceBertConfig(
        **config["model"]
    )

    model = SubspaceBertForMaskedLM(bert_config)
    
    model


    print("\n======== Config File ========")
    print(json.dumps(config, indent=2))
    print("=============================\n")
  
    """# Review

    ### Parameter Tally
    """

    print("\n======== Parameters ========")

    ## Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The model has {:} different named parameters.\n'.format(len(params)))

    # =====================
    #     Review Params
    # =====================

    total_params = 0
    for p_name, p in params:
        total_params += p.numel()

    config["stats"]["total_elements"] = format_size(total_params)

    print(f"Total elements: {config['stats']['total_elements']}\n")

    # Print out final config for quick verification
    #for k, v in config["model"].items():
    #    print(f"{k:>25}: {v:>10}")

    print("=============================\n")

    # Display a full parameter breakdown using the shared utility
    summarize_parameters(model)


    # ========================================
    #   Format Settings for WandB Run Name
    # ========================================
    
    # Format the learning rate as a scientific notation string like 5e-4
    lr_str = '{:.0e}'.format(config['pre_train']['learning_rate'])

    # Attention configuration
    if config['model'].get("use_mla"):
        dense_str = str(config['model'].get("num_dense_layers")) + "mha + "

        # If no output subspace is used, the dimension will show as -1.
        attn_str = (
            dense_str
            + "mla."
            + str(config['model'].get("q_lora_rank"))
            + "."
            + str(config['model'].get("kv_lora_rank"))
            + "."
            + str(config['model'].get("o_lora_rank"))
        )
    else:
        attn_str = "mha"

    # MLP Configuration
    if config['model'].get("ffn_decompose"):
        # Specify the number of dense mlps and their size.
        dense_str = (
            str(config['model'].get("num_dense_layers"))
            + "mlp."
            + str(config['model'].get("intermediate_size"))
            + " + "
        )

        # Specify the neuron count and latent dimension of the decomposed mlps.
        mlp_str = (
            dense_str
            + "dcmp."
            + str(config['model'].get("intermediate_size"))
            + "."
            + str(config['model'].get("ffn_rank"))
    )
    else:
        mlp_str = "mlp." + str(config['model'].get("intermediate_size"))


    run_name = (
        f"{config['stats']['total_elements']} - {attn_str} - {mlp_str} - "
        f"h{config['model']['hidden_size']} - l{config['model']['num_hidden_layers']} - "
        f"bs{config['pre_train']['train_batch_size']} - lr{lr_str} - "
        f"seq{config['pre_train']['max_seq_length']}"
    )

    config['pre_train']["run_name"] = run_name
    model.config.run_name = run_name

    print(run_name)





    wandb.init(
        project="encoder-pretrain-profiler",
        name=run_name,
        config=config
    )

    # ===============================
    #       Training Arguments
    # ===============================

    training_args = TrainingArguments(
        output_dir=config["pre_train"]["output_dir"],

        per_device_train_batch_size=config["pre_train"]["train_batch_size"],
        per_device_eval_batch_size=config["pre_train"]["eval_batch_size"],

        fp16=config["pre_train"].get("fp16", False),

        learning_rate=config["pre_train"]["learning_rate"],
        num_train_epochs=config["pre_train"]["num_train_epochs"],
        # Limit steps for profiling so execution finishes quickly
        max_steps=config["pre_train"].get("max_steps", 10),
        
        # Evaluate every xx steps
        # Recent versions changed from 'evaluation_strategy'
        eval_strategy="steps", 
        eval_steps=config["pre_train"].get("eval_steps", 500),  # Default to 500
        
        # Let's sanity check that this is actually the problem.
        #logging_steps=100,
        logging_steps=1000000,
        
        # Checkpoint saving
        save_steps=500,
        save_total_limit=2,           # Optional: keeps last 2 checkpoints
        save_strategy="steps",
        report_to=["wandb"],
        run_name=run_name,
        remove_unused_columns=False,  # Optional: avoid dropping custom model inputs
    )

    # ===============================
    #           Trainer
    # ===============================
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        
        # New argument, allows for other modalities.
        processing_class=tokenizer,
        
        data_collator=data_collator,
    )
    """

    
    # Use CustomTrainer to avoid per-step loss.item() calls which force
    # cudaStreamSynchronize and to use dataloader settings from the config.
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer, # New argument name, allows for other modalities.
        data_collator=data_collator,
        config=config,
    )

    # =====================
    #     Run Training
    # =====================

    try:
        # Profile the training run to collect performance information.
        # The context records all operations executed by `trainer.train`.
        with torch_profiler.profile(
            activities=[torch_profiler.ProfilerActivity.CPU, torch_profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            trainer.train()

        # Display a summary of the profiling results.
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        metrics = trainer.evaluate()
        wandb.log(metrics)
    
    finally:
        wandb.finish()
        trainer.save_model(config["pre_train"]["output_dir"])
    

if __name__ == "__main__":
    main()
