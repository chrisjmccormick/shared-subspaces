print("Importing Packages...\n")

import json
import argparse
from pathlib import Path
import sys
import os

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

print("\n===== DEBUG: File Context =====")
print("Current Working Directory:", os.getcwd())
print("Script __file__:", __file__)
print("sys.path:")
for p in sys.path:
    print("  ", p)

print("\nList of files in CWD:")
for f in os.listdir():
    print("  ", f)

print("\nList of files in 'models/' relative to CWD:")
models_path = os.path.join(os.getcwd(), "models")
if os.path.isdir(models_path):
    for f in os.listdir(models_path):
        print("  ", f)
else:
    print("  (models/ directory not found)")

# This file exists in the 'scripts' subdirectory, go up a level to find 'models'.
from models.custom_bert import SubspaceBertForMaskedLM, SubspaceBertConfig
from utils import summarize_parameters, format_size

# Make sure we can import modules from the encoder-pretrain package
PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("PROJECT_ROOT", PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Temp - Using base class for the moment.
#from models.custom_bert import CustomBertForMaskedLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config")
    return parser.parse_args()

"""Helper utilities are imported from utils."""



def main():
    args = parse_args()
    
    # Load the config file.    
    with open(args.config) as f:
        config = json.load(f)

    # Initialize the optional stats dictionary so later assignments don't fail.
    # Configuration files in this repo don't define a "stats" key yet.
    if "stats" not in config:
        config["stats"] = {}

    # Strict key check on the model configuration.
    valid_keys = SubspaceBertConfig.__init__.__code__.co_varnames
    valid_keys = set(valid_keys) - {"self", "kwargs"}
    extra_keys = set(config["model"]) - valid_keys
    if extra_keys:
        raise ValueError(f"Unknown keys in config: {sorted(extra_keys)}")


    print("Transformers version:", transformers.__version__)  # Helpful sanity check

    # Set random seed for reproducibility
    set_seed(config['pre_train'].get("seed", 42))

    # Setup Weights & Biases
    # Use offline mode if the API key isn't in the environment variables.
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"
    
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    
    # Use the standard BERT tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    assert config['model']["vocab_size"] == tokenizer.vocab_size    
    
    dataset = load_dataset(
        config['pre_train']["dataset_name"], 
        config['pre_train']["dataset_config"]
    )
    

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=config['pre_train']["max_seq_length"]
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
        mlm_probability=config['pre_train']["mlm_probability"] # Default is 15%
    )


    # Will raise TypeError, by design, if required args are missing
    bert_config = SubspaceBertConfig(**config["model"])
    
    model = SubspaceBertForMaskedLM(bert_config)

    # ================================
    #       Review Configuration
    # ================================

    # Display architecture
    model

    print("\n======== Model ========")
    print(json.dumps(config["model"], indent=2))
    
    print("\n======== Pre-Train ========")
    print(json.dumps(config["pre_train"], indent=2))

    # Print out final config for quick verification
    #for k, v in config['pre_train'].items():
    #    print(f"{k:>25}: {v:>10}")

  
    print("=============================\n")
  
    print("\n======== Parameters ========")

    ## Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The model has {:} different named parameters.\n'.format(len(params)))

    total_params = 0
    for p_name, p in params:
        total_params += p.numel()

    config["stats"]["total_elements"] = format_size(total_params)

    print(f"Total elements: {config['stats']['total_elements']}\n")

    # Display a full parameter breakdown using the shared utility
    summarize_parameters(model)

    # ========================================
    #   Format Settings for WandB Run Name
    # ========================================

    # Format the cfg learning rate as a scientific notation string like 5e-4
    lr_str = '{:.0e}'.format(config['pre_train']['learning_rate'])

    # Attention configuration
    if bert_config.use_mla:
        dense_str = str(bert_config.num_dense_layers) + "mha + "

        if bert_config.output_subspace:
            o_str = "." + str(bert_config.o_lora_rank)
        else:
            o_str = ""

        # If no output subspace is used, the dimension will show as -1.
        attn_str = (
            dense_str
            + "mla."
            + str(bert_config.q_lora_rank)
            + "."
            + str(bert_config.kv_lora_rank)
            + o_str
        )
    else:
        attn_str = "mha"

    # MLP Configuration
    if bert_config.ffn_decompose:
        dense_str = (
            str(bert_config.num_dense_layers)
            + "mlp."
            + str(bert_config.intermediate_size)
            + " + "
        )

        mlp_str = (
            dense_str
            + "dcmp."
            + str(bert_config.intermediate_size)
            + "."
            + str(bert_config.ffn_rank)
        )
    else:
        mlp_str = "mlp." + str(bert_config.intermediate_size)

    # Final run name
    run_name = (
        f"{config['stats']['total_elements']} - {attn_str} - {mlp_str} - "
        f"h{bert_config.hidden_size} - l{bert_config.num_hidden_layers} - "
        f"bs{config['pre_train']['train_batch_size']} - lr{lr_str} - "
        f"seq{config['pre_train']['max_seq_length']}"
    )

    config['pre_train']["run_name"] = run_name
    
    model.config.run_name = run_name

    print(run_name)

    wandb.init(
        project="encoder-pretrain", 
        name=run_name,
        config=config
    )

    # ===============================
    #       Training Arguments
    # ===============================

    training_args = TrainingArguments(
        output_dir=config['pre_train']["output_dir"],
        
        per_device_train_batch_size=config['pre_train']["train_batch_size"],
        per_device_eval_batch_size=config['pre_train']["eval_batch_size"],
        
        fp16=config['pre_train'].get("fp16", False),
        
        learning_rate=config['pre_train']["learning_rate"],
        num_train_epochs=config['pre_train']["num_train_epochs"],
        
        # Evaluate every xx steps
        # Recent versions changed from 'evaluation_strategy'
        eval_strategy="steps", 
        eval_steps=config['pre_train'].get("eval_steps", 500), # Default to 500
        
        logging_steps=50,
        
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        
        # New argument, allows for other modalities.
        processing_class=tokenizer,
        
        data_collator=data_collator,
    )

    # =====================
    #     Run Training
    # =====================

    try:
        trainer.train()
        metrics = trainer.evaluate()
        wandb.log(metrics)
    
    finally:
        wandb.finish()
        trainer.save_model(config['pre_train']["output_dir"])
    

if __name__ == "__main__":
    main()
