1\. Repository Organization
---------------------------

*   Continue working inside /encoder-pretrain/.
    
*   Create folders for:
    
    *   data/ – to store or download pretraining data.
        
    *   configs/ – experiment configuration files.
        
    *   models/ – custom model definitions (e.g., baseline BERT, MLA modifications, output subspace, decomposed MLP).
        
    *   scripts/ – training and evaluation scripts.
        

2\. Baseline Setup
------------------

1.  **Environment**
    
    *   Use Python 3.10+ with PyTorch and Hugging Face Transformers (latest stable versions).
        
    *   Include datasets and accelerate for efficient loading/training.
        
2.  **Model Definition**
    
    *   Start with a small BERT-like architecture (e.g., bert-base with reduced parameters suitable for a single 40GB A100).
        
    *   Use the existing BertModel implementation from Transformers as the baseline.
        
3.  **Pretraining Data**
    
    *   Use a small text corpus or subset of a public dataset (e.g., WikiText or OpenWebText).
        
    *   Set up data processing with the datasets library (tokenization, streaming if necessary).
        
4.  **Training Script**
    
    *   Implement a basic MLM pretraining script with Trainer or a custom loop.
        
    *   Include logging, checkpointing, and evaluation metrics (loss, perplexity).
        
5.  **Experiment Tracking**
    
    *   Save logs and metrics (consider wandb or TensorBoard).
        

3\. Multihead Latent Attention (MLA) Variant
--------------------------------------------

1.  **Modify Attention Module**
    
    *   Extend the Transformer’s self-attention layer to use DeepSeekV3Attention (already in Transformers).
        
    *   Adjust to insert a shared output latent projection (shared among heads).
        
2.  **Configuration Flags**
    
    *   Add a config option to toggle MLA and output subspace size.
        
3.  **Training/Evaluation**
    
    *   Run the same pretraining loop with the MLA model.
        
    *   Compare loss/accuracy to the baseline.
        

4\. Output Subspace & Decomposed MLP Variants
---------------------------------------------

1.  **Shared Output Projection**
    
    *   Implement the output subspace as described in reference/background.md.
        
    *   Ensure per-head matrices project into a shared latent, followed by a shared projection back to model dimension.
        
2.  **Decomposed MLP**
    
    *   Factor the FFN (feed-forward network) layers with low-rank decompositions.
        
    *   Provide config options for rank and dimensions.
        
3.  **Experiments**
    
    *   Train variants incrementally: baseline + MLA, MLA + output subspace, MLA + output subspace + decomposed MLP.
        
    *   Collect metrics and compare.
        

5\. Multi‑GPU Scaling (Later Phase)
-----------------------------------

1.  **Distributed Training**
    
    *   Use accelerate or DeepSpeed for multi-GPU scaling.
        
    *   Prepare scripts that can switch between single- and multi-GPU environments.
        
2.  **Hyperparameter Sweeps**
    
    *   Set up batch sizes, learning rates, and subspace dimensions for exploration.
        
    *   Automate with shell scripts or simple scheduler.
        

6\. Documentation and Reproducibility
-------------------------------------

*   Document all commands and config options in encoder-pretrain/README.md.
    
*   Include sample config files showing how to enable each variant.
    
*   Provide instructions for loading checkpoints and evaluating.
    

This plan should serve as a roadmap to implement and test each of the architecture variations while keeping everything organized in /encoder-pretrain/.