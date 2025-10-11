# SparseMoE Integration Plan

## Roadmap Overview
1. **Create MoE Package Skeleton**
   - Mirror the `subspace_decoder` layout inside `moe/` (`configs/`, `layers/`, `models/`, `utils.py`).
   - Stub `__init__` files and docstrings so modules can be imported standalone.
2. **Document Component Contracts**
   - Within this README, spell out responsibilities and tensor contracts for each component before coding them.
   - Keep the documentation synchronized with implementation choices (e.g., expert capacity math, routing outputs).
3. **Configuration Layer**
   - Port `SharedSpaceDecoderConfig` into `moe/models/` and extend it with Sparse MoE knobs:
     - `num_experts`, `router_top_k`, `capacity_factor`, `router_noise`.
     - Validation helpers and derived convenience properties (e.g., expert capacity per layer).
4. **MLA Attention Pathway**
   - Recreate MLA attention modules under `moe/layers/mla.py` unchanged aside from import paths.
   - Preserve shared/private decomposition logic and extensive inline commentary.
5. **Expert SwiGLU Module**
   - Implement `ExpertSwiGLU` inside `moe/layers/feedforward.py`:
     - Dense path for early layers, decomposed path mirroring `subspace_decoder` for later layers.
     - Shared projections down to rank `R`, SwiGLU activation, projection back to model dim.
6. **Router + Sparse Dispatch**
   - Build a Noisy-TopK router with reproducible noise and explicit tensor shape annotations.
   - Implement scatter/gather helpers that enforce expert capacity and drop overflow tokens gracefully.
7. **SparseMoE Block Integration**
   - Compose router and experts into `SparseMoEFeedForward` that replaces the vanilla FFN step.
   - Provide a dense fallback used when `num_experts == 1` or MoE is disabled in config.
   - Maintain residual pathways and dropout semantics consistent with the decoder stack.
8. **Model Wiring**
   - Implement a `SharedSpaceSparseDecoder` inside `moe/models/shared_space_decoder.py` that:
     - Uses MLA attention modules from Step 4.
     - Swaps the FFN stage with the MoE block from Step 7.
     - Exposes generation helpers and load-bearing utility methods.
9. **Utilities & Testing Hooks**
   - Port any helper utilities needed for configs/tests into `moe/utils.py`.
   - Sketch test entry points mirroring `subspace_decoder/tests/` to ease future validation.

## Component Blueprint

- **Config (`models/shared_space_config.py`)**
  - Construction of model hyper-parameters and MoE knobs with validation.
  - Derived attributes for capacity (`tokens_per_batch`, `expert_capacity`).

- **MLA Layers (`layers/mla.py`)**
  - Shared/private projections for Q, K, V, and optional output latent spaces.
  - Heavy commentary on tensor shapes and decomposition rationale.

- **Expert SwiGLU (`layers/feedforward.py`)**
  - `ExpertSwiGLU` dense vs. decomposed paths, matching Subspace FFN structure.
  - `SparseMoEFeedForward` that orchestrates routing, dispatch, expert evaluation, and combining outputs.

- **Routers (`layers/feedforward.py`)**
  - `NoisyTopKRouter` generating sparse assignment weights.
  - Capacity mask utilities for top-k selection with overflow handling.

- **Model (`models/shared_space_decoder.py`)**
  - Transformer block definition using MLA attention + MoE FFN.
  - Encoder/decoder forward logic, weight initialization, and optional LM head tying.

- **Configs (`configs/*.json`)**
  - MoE-enabled variants of existing decoder configs showcasing new knobs.

- **Utils (`utils.py`)**
  - Helper functions shared across modules (e.g., norm factory, dtype helpers).

## Deliverables & Validation
- Provide inline comments mirroring the original verbose style for clarity.
- Outline how to extend tests for routing and MoE equivalence (dense vs. sparse) once harness is ready.
- Ensure README stays updated once implementation completes, summarizing any deviations discovered during coding.

## Implementation Status
- `models/shared_space_config.py` now exposes `SparseMoEDecoderConfig` with routing knobs (`num_experts`, `router_top_k`, `capacity_factor`, `router_noise_std`, `expert_dropout_prob`).
- `layers/feedforward.py` supplies `ExpertSwiGLU`, `NoisyTopKRouter`, and `SparseMoEFeedForward`, providing dense fallbacks and low-rank decompositions.
- `layers/mla.py` is ported verbatim from the subspace decoder with updated imports to the MoE config/utility modules.
- `models/shared_space_decoder.py` wires MLA attention with the new Sparse MoE feed-forward blocks across the transformer stack.
- `configs/gpt-2_sparse_moe_wiki103.json` demonstrates how to activate the Sparse MoE path for the GPT-2 MLA baseline.
