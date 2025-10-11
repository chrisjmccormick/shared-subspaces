
# I want you to adjust this expert network to have the Swiglu and decompositions as shown in FFN component in subspace_decoder/layers/feedforward.py
class Expert(nn.Module):
    """An MLP followed by a non-linear layer"""

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Here is the NoisyTopK router implementation    
class NoisyTopKRouter(nn.Module):
    """NoisyTopK Router module..."""

    def __init__(self, topk, n_embd, n_experts):
        super().__init__()

        self.topk = topk
        self.topkgate_linear = nn.Linear(n_embd, n_experts)
        self.noisy_linear = nn.Linear(n_embd, n_experts)

    def forward(self, mha_output):
        # Here x is the mha output(multihead attention output)
        logits = self.topkgate_linear(mha_output)
        noisy_logits = self.noisy_linear(mha_output) # (B, T, E)

        # Generate noisy by adding unit gaussian noise to the noisy logits
        noise = torch.randn_like(noisy_logits) * F.softplus(noisy_logits)
        noisy_logits = logits + noise

        # Let's now create sparse logits along the expert dimension
        zeros = torch.full_like(noisy_logits, float("-inf")) # (B, T, E)
        topkgate_logits, topkgate_indices = noisy_logits.topk(self.topk, dim=-1)
        sparse_logits = zeros.scatter(-1, topkgate_indices, topkgate_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, topkgate_indices
    

# Let's create SparseMOE class with expert capacity
class SparseMOE(nn.Module):
    """Sparse MOE module with expert capacity...

    return:
    """
    def __init__(self, topk, n_embd, n_experts, capacity_factor=1.0):
        super().__init__()

        self.topk = topk
        self.router = NoisyTopKRouter(topk, n_embd, n_experts)
        self.experts = nn.ModuleList([Expert(n_embd) for _ in range(n_experts)])
        self.capacity_factor = capacity_factor
        self.topk = topk
        self.n_experts = n_experts


    def forward(self, x):
        # Get the shape of x
        B, T, n_embd = x.shape
        # Generate router output
        # gating output (B, T, E)
        gating_output, topkgate_indices = self.router(x)
        final_output = torch.zeros_like(x) # (B, T, n_embd)

        # Flatten x and gating output
        flat_x = x.view(-1, x.size(-1)) # (B*T, n_embd)
        flat_gating_output = gating_output.view(-1, gating_output.size(-1)) # (B*T, E)

        # Compute expert capacity
        tokens_per_batch = B * T * self.topk
        expert_capacity = int(( tokens_per_batch / self.n_experts) * self.capacity_factor )

        # Create updates variable for updating token values efficiently
        updates = torch.zeros_like(flat_x)

        for expert_idx, current_expert in enumerate(self.experts):
          # Select indices of tokens routed to the current expert
          expert_mask = (topkgate_indices == expert_idx).any(dim=-1) # (B, T, k) -> (B, T)
          flat_mask = expert_mask.view(-1) # (B*T)
          selected_indices = flat_mask.nonzero(as_tuple=True)[0] # (Ts)
          limited_indices = selected_indices[:expert_capacity] if expert_capacity < len(selected_indices) else selected_indices

          if len(limited_indices) > 0:
            expert_input = flat_x[limited_indices] # (Ts, n_embd)
            expert_output = current_expert(expert_input) # (Ts, n_embd)

            # Let's now grab router weights
            router_weights = flat_gating_output[limited_indices, expert_idx].unsqueeze(1) # (Ts, 1)
            weighted_output = router_weights * expert_output # (Ts, n_embd)

            # Use scatter_add to do updates
            # Expand limited_indices to match the dimensions of updates for scatter_add_
            updates.scatter_add_(0, limited_indices.unsqueeze(-1).expand(-1, n_embd), weighted_output)

        # Reshape updates to match original dimension of x
        final_output = updates.view(B, T, -1)

        return final_output