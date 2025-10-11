# SparseMoE Integration Plan

## Objectives
- Replace the vanilla SwiGLU feed-forward layers in the subspace decoder with a Sparse Mixture-of-Experts implementation.
- Preserve existing MLA attention pathways and their decomposition logic while extending decomposition to the expert networks.
- Keep the SparseMoE implementation lightweight by reusing routing components and enforcing explicit expert capacity constraints.

## Component Implementation Plan
1. **Configuration Updates**
   - Extend `SharedSpaceDecoderConfig` with SparseMoE-specific knobs (expert count, top-k selection, capacity scaling, noise).
   - Provide validation helpers and derived utility values so layers can determine when to activate MoE logic.
2. **Expert Feed-Forward Network**
   - Implement an expert MLP that mirrors the existing decomposed SwiGLU FFN structure (shared projections, per-expert SwiGLU core, shared output projection).
   - Support both dense and decomposed experts based on layer index and configuration flags.
3. **Routing Mechanism**
   - Build a Noisy Top-K router that generates sparse dispatch weights, capacity masks, and top-k expert indices per token.
   - Incorporate configurable jitter/noise for exploration, deterministic seeding for reproducibility, and clear tensor shape documentation.
4. **Sparse MoE Core**
   - Combine the router and expert ensemble into a `SparseMoE` module that performs token-to-expert scattering, capacity enforcement, expert execution, and gather-back with weighted sums.
   - Ensure operations are batched-friendly, maintain gradient flow, and gracefully handle tokens dropped due to capacity overflow.
5. **Decoder Layer Integration**
   - Replace `SubspaceFeedForward` calls with the SparseMoE module while preserving residual connections and layer norms.
   - Provide fallbacks to dense FFN behavior when MoE is disabled via configuration.
6. **Documentation & Comments**
   - Mirror the descriptive commenting style in `subspace_decoder` files, annotating tensor shapes, processing stages, and rationale for design choices.

## Testing & Validation Plan
- Add targeted unit tests covering router behavior, capacity enforcement, and equivalence to dense FFN when `num_experts=1`.
- Run existing decoder smoke tests to ensure MLA pathways remain functional.
- Provide guidance for downstream users on enabling MoE in configuration JSON files.
