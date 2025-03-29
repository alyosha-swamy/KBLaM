import torch

class LlamaLinearScalingRotaryEmbedding(torch.nn.Module):
    """Rotary position embeddings with linear scaling.
    
    Args:
        dim (int): Dimension of the model.
        max_position_embeddings (int, optional): Maximum position. Defaults to 2048.
        base (int, optional): Base for computing theta. Defaults to 10000.
        device (torch.device, optional): Device for computation. Defaults to None.
        scaling_factor (float, optional): Linear scaling factor. Defaults to 1.0.
    """
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build position embeddings
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor  # Scale positions by the scaling factor
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # If seq_len is None, infer from x
        if seq_len is None:
            seq_len = x.shape[2]
        
        # If seq_len is a tensor, use its shape for the sequence length
        if isinstance(seq_len, torch.Tensor):
            if len(seq_len.shape) > 0:  # It's a tensor with dimensions
                seq_len = seq_len.shape[0]
            else:  # It's a scalar tensor
                seq_len = seq_len.item()
                
        # Now check if we need to update cache
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            t = t / self.scaling_factor  # Scale positions by the scaling factor
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos(), persistent=False)
            self.register_buffer("sin_cached", emb.sin(), persistent=False)
        
        return (
            self.cos_cached[:seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:seq_len, ...].to(dtype=x.dtype),
        )

class LlamaDynamicNTKScalingRotaryEmbedding(torch.nn.Module):
    """Implement Llama's DyanmicNTK extrapolation method, thereby broadening the model support context.
    Args:
        dim (int): Characteristic dimension of each self-attentional head.
        max_position_embeddings (int, optional): Model's training length. Defaults to 2048.
        base (int, optional): The rotation position encodes the rotation Angle base number. Defaults to 10000.
        device (Any, optional): Running device. Defaults to None.
        scaling_factor (float, optional): NTK method extrapolation coefficient. Defaults to 1.0.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim
        self.base = base
        self.scaling_factor = scaling_factor

        # Build here to make `torch.jit.trace` work.
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _update_cached(self, x, seq_len=None):
        # If seq_len is a tensor, use its shape for the sequence length
        if isinstance(seq_len, torch.Tensor):
            if len(seq_len.shape) > 0:  # It's a tensor with dimensions
                seq_len = seq_len.shape[0]
            else:  # It's a scalar tensor
                seq_len = seq_len.item()
                
        self.max_seq_len_cached = max(seq_len, self.max_position_embeddings)
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(x.device) / self.dim))
        else:
            inv_freq = self.inv_freq
        t = torch.arange(self.max_seq_len_cached, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # If seq_len is None, infer from x
        if seq_len is None:
            seq_len = x.shape[2]
        
        # If seq_len is a tensor, use its shape for the sequence length
        if isinstance(seq_len, torch.Tensor):
            if len(seq_len.shape) > 0:  # It's a tensor with dimensions
                seq_len = seq_len.shape[0]
            else:  # It's a scalar tensor
                seq_len = seq_len.item()
        
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len <= self.max_position_embeddings:
            # Reset the tables if the sequence length has changed,
            if self.max_seq_len_cached > self.max_position_embeddings:
                self._update_cached(x, seq_len)
        else:
            self._update_cached(x, seq_len)

        return (
            self.cos_cached[:seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:seq_len, ...].to(dtype=x.dtype),
        ) 