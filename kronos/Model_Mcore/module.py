import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(torch.nn.Module):
    """ Mirroring LLamaRMSNorm """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Note: MCore LLamaRMSNorm calculates variance on float32 hidden_states
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x):
        input_dtype = x.dtype
        hidden_states_fp32 = x.float() # Cast input to float32
        variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True) # Compute variance in float32
        normed_x_fp32 = hidden_states_fp32 * torch.rsqrt(variance + self.eps) # Normalize in float32

        # Multiply weight (cast to float) with float32 normed_x, then cast back
        # This matches the MCore LLamaRMSNorm NOTE about TE implementation
        output = (self.weight.float() * normed_x_fp32).to(input_dtype)
        return output

class FeedForward(nn.Module):
    """ Mirroring the SwiGLU MLP structure used in LLaMA/LLaMA3.1 (MCore local version) """
    def __init__(self, d_model: int, ff_dim: int, ffn_dropout_p: float = 0.0):
        super().__init__()
        # Corresponds to linear_fc1 combining gate_proj and up_proj in mcore local
        self.fc1 = nn.Linear(d_model, 2 * ff_dim, bias=False) # Combined gate & up projection
        # Corresponds to linear_fc2 in mcore
        self.fc2 = nn.Linear(ff_dim, d_model, bias=False) # down_proj
        self.ffn_dropout = nn.Dropout(ffn_dropout_p) # Corresponds to hidden_dropout in mcore args

    def forward(self, x) -> torch.Tensor:
        # Apply combined projection and split for SwiGLU
        gate_proj, up_proj = self.fc1(x).chunk(2, dim=-1)
        # F.silu(gate_proj) * up_proj corresponds to SwiGLU activation
        # Then apply fc2 and dropout
        return self.ffn_dropout(self.fc2(F.silu(gate_proj) * up_proj))

class RotaryPositionalEmbedding(nn.Module):
    """ Mirroring megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding,
        but expects inv_freq to be loaded via state_dict. """
    def __init__(self, dim: int): # Removed base
        super().__init__()
        self.dim = dim
        # Initialize the buffer with expected shape, dtype will be set by state_dict load
        self.register_buffer('inv_freq', torch.zeros(dim // 2), persistent=True)
        self.cos_cached = None
        self.sin_cached = None
        self.seq_len_cached = -1 # Initialize cache status

    # Removed _compute_inv_freq method

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """ Update the cache for a given sequence length, device, and dtype using loaded inv_freq. """
        # Use the loaded inv_freq buffer
        if self.inv_freq is None:
             raise ValueError("inv_freq buffer has not been loaded into RotaryPositionalEmbedding")

        # Ensure inv_freq is on the correct device for calculations
        inv_freq_on_device = self.inv_freq.to(device=device)

        self.seq_len_cached = seq_len
        t = torch.arange(self.seq_len_cached, device=device).type_as(inv_freq_on_device)

        # freqs = torch.einsum('i,j->ij', t, inv_freq) # Original MCore calculation
        # Need to ensure inv_freq has the right shape [dim/2] for einsum
        # Assuming self.inv_freq is stored correctly with shape [dim/2]
        freqs = torch.outer(t, inv_freq_on_device) # More explicit outer product

        # freqs shape: [seq_len, dim / 2]
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb shape: [seq_len, dim]

        # --- Calculate cos/sin on emb (likely float32 from inv_freq) --- 
        # Removed explicit cast to float32 before trig
        # emb_fp32 = emb.float()

        # Store cos/sin cache, potentially still in float32 or original inv_freq dtype
        # Casting to target dtype will happen in forward pass
        self.cos_cached = emb.cos() # Removed .to(dtype)
        self.sin_cached = emb.sin() # Removed .to(dtype)

    def forward(self, x: torch.Tensor, seq_dim=1) -> torch.Tensor:
        """ Apply rotary embeddings to the input tensor. """
        seq_len = x.shape[seq_dim]

        # Check if cache needs update (length, device, dtype mismatch or uninitialized inv_freq)
        # NOTE: We don't check dtype match anymore, as casting happens right before use.
        if (self.inv_freq is None or # Make sure inv_freq is loaded
            seq_len != self.seq_len_cached or
            self.cos_cached is None or self.cos_cached.device != x.device or
            self.sin_cached is None or self.sin_cached.device != x.device):
            # If inv_freq is loaded, update cache, otherwise error handled in update func
            self._update_cos_sin_cache(seq_len, device=x.device, dtype=x.dtype) # Pass target dtype for potential use in update?

        # Retrieve cached cos/sin, slicing if necessary
        cos = self.cos_cached[:seq_len, ...]
        sin = self.sin_cached[:seq_len, ...]

        # Reshape cos and sin for broadcasting based on input tensor dims
        # Assume x is [batch, seq_len, ..., dim] or [seq_len, batch, ..., dim]
        # We need cos/sin reshaped to broadcast correctly.

        if seq_dim == 1: # e.g., [batch, seq_len, n_heads, head_dim]
            # Target shape for broadcasting: [1, seq_len, 1, self.dim]
            cos = cos.view(1, seq_len, 1, self.dim)
            sin = sin.view(1, seq_len, 1, self.dim)
        elif seq_dim == 0: # e.g., [seq_len, batch, n_heads, head_dim] - typical MCore format
             # Target shape for broadcasting: [seq_len, 1, 1, self.dim]
             cos = cos.view(seq_len, 1, 1, self.dim)
             sin = sin.view(seq_len, 1, 1, self.dim)
        else:
             # Adapt for other seq_dim if necessary
             raise ValueError(f"Unsupported seq_dim for RoPE: {seq_dim}")

        # --- Cast cos/sin to input dtype right before use (aligns with MCore apply_rotary_pos_emb_bshd) ---
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        # Helper function for rotation
        def rotate_half(y):
            y1, y2 = y.chunk(2, dim=-1)
            return torch.cat((-y2, y1), dim=-1)

        # Apply rotation: (x * cos) + (rotate_half(x) * sin)
        return (x * cos) + (rotate_half(x) * sin)

class CoreAttention(nn.Module):
    """ Performs scaled dot-product attention. """
    def __init__(self, d_model: int, n_heads: int, attn_dropout_p: float = 0.0):
        super().__init__()
        self.head_dim = d_model // n_heads
        self.attn_dropout = nn.Dropout(attn_dropout_p)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor = None):
        # q, k, v: [batch, n_heads, seq_len, head_dim]
        batch_size, n_heads, seq_len, _ = q.shape

        # Calculate attention scores: (B, H, S, D) @ (B, H, D, S) -> (B, H, S, S)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask if no explicit mask is provided
        if attention_mask is None:
            # Create causal mask (True for positions to mask)
            # Needs to be broadcastable to attn_scores shape (B, H, S, S)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
            # Apply mask by adding -inf (or large negative number) where mask is True
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        # TODO: Add support for applying a custom attention_mask if provided

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention dropout
        dropout_p = self.attn_dropout.p if self.training else 0.0
        attn_weights = F.dropout(attn_weights, p=dropout_p)

        # Multiply weights by values: (B, H, S, S) @ (B, H, S, D) -> (B, H, S, D)
        attn_output = torch.matmul(attn_weights, v)

        return attn_output # Shape: [batch, n_heads, seq_len, head_dim]

class MultiHeadAttentionWithRoPE(nn.Module):
    """ Mirroring mcore SelfAttention with RoPE, GQA support (MCore local version) """
    def __init__(self, d_model: int, n_heads: int, num_query_groups: int, rotary_percent: float = 1.0, attn_dropout_p: float = 0.0, resid_dropout_p: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % num_query_groups == 0, "n_heads must be divisible by num_query_groups"

        self.d_model = d_model
        self.n_heads = n_heads # Number of Q heads
        self.num_query_groups = num_query_groups # Number of K/V groups/heads
        self.num_kv_heads = num_query_groups
        self.head_dim = d_model // n_heads
        self.kv_dim = self.num_kv_heads * self.head_dim # Dimension for K and V projections

        # Calculate the actual dimension for RoPE based on rotary_percent
        self.rotary_dim = int(self.head_dim * rotary_percent)

        # Combined QKV projection layer (mirrors MCore ColumnParallelLinear for QKV)
        self.qkv_proj = nn.Linear(d_model, d_model + 2 * self.kv_dim, bias=False)

        # Instantiate RoPE with the calculated rotary_dim
        self.rotary = RotaryPositionalEmbedding(self.rotary_dim)

        # Instantiate CoreAttention
        self.core_attn = CoreAttention(d_model, n_heads, attn_dropout_p)

        # Output projection and dropout
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        batch_size, seq_len, _ = x.shape

        # Combined QKV projection and split
        qkv = self.qkv_proj(x) # [batch, seq, d_model + 2 * kv_dim]
        q, k, v = qkv.split([self.d_model, self.kv_dim, self.kv_dim], dim=-1)

        # Reshape for multi-head attention / GQA
        # Native shape: [batch_size, seq_len, num_heads/num_kv_heads, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply Rotary Embeddings selectively based on self.rotary_dim
        if self.rotary_dim > 0:
            q_rot = q[..., :self.rotary_dim]
            q_pass = q[..., self.rotary_dim:]
            k_rot = k[..., :self.rotary_dim]
            k_pass = k[..., self.rotary_dim:]

            q_rot = self.rotary(q_rot, seq_dim=1)
            k_rot = self.rotary(k_rot, seq_dim=1)

            # Concatenate rotated and pass-through parts
            q = torch.cat((q_rot, q_pass), dim=-1)
            k = torch.cat((k_rot, k_pass), dim=-1)
        # else: RoPE is not applied if rotary_dim is 0

        # Transpose for attention calculation: [batch, n_heads/n_kv_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat K/V heads for GQA
        repeat_factor = self.n_heads // self.num_kv_heads
        if repeat_factor > 1:
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Core Attention computation
        attn_output = self.core_attn(q, k, v, attention_mask) # attn_output: [batch, n_heads, seq_len, head_dim]

        # Transpose back and reshape: [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Apply output projection and dropout
        return self.resid_dropout(self.out_proj(attn_output))

class TransformerBlock(nn.Module):
    """ Mirroring mcore TransformerLayer structure (local version) with external norms """
    def __init__(self, d_model: int, n_heads: int, num_query_groups: int, ff_dim: int, rotary_percent: float = 1.0, ffn_dropout_p: float = 0.0, attn_dropout_p: float = 0.0, resid_dropout_p: float = 0.0, norm_eps: float = 1e-6):
        super().__init__()
        # Layer norms (now external to sub-modules)
        self.input_layernorm = RMSNorm(d_model, eps=norm_eps)
        self.pre_mlp_layernorm = RMSNorm(d_model, eps=norm_eps)

        # Instantiate Attention and MLP blocks (without internal norms)
        self.self_attn = MultiHeadAttentionWithRoPE(
            d_model=d_model,
            n_heads=n_heads,
            num_query_groups=num_query_groups,
            rotary_percent=rotary_percent,
            attn_dropout_p=attn_dropout_p,
            resid_dropout_p=resid_dropout_p,
        )
        self.ffn = FeedForward(
            d_model=d_model,
            ff_dim=ff_dim,
            ffn_dropout_p=ffn_dropout_p,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        # Apply input norm, then self-attention block + residual
        residual = x
        normed_x = self.input_layernorm(x)
        attn_out = self.self_attn(normed_x, attention_mask=attention_mask)
        x = residual + attn_out

        # Apply pre-MLP norm, then MLP block + residual
        residual = x
        normed_x = self.pre_mlp_layernorm(x)
        ffn_out = self.ffn(normed_x)
        x = residual + ffn_out
        return x

