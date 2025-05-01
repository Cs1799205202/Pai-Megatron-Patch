import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .module import RMSNorm, TransformerBlock

# Configuration dataclass (optional, but good practice)
from dataclasses import dataclass

@dataclass
class KronosUConfig:
    vocab_size: int = 50257 # Example default
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    ff_dim: int = 3072 # Often 4 * d_model
    max_seq_len: int = 1024 # Still needed for potential checks, even if not used by RoPE directly
    norm_eps: float = 1e-5
    # rotary_base: float = 10000.0 # Removed
    num_query_groups: int = 12 # Default to MHA, should be passed from args
    rotary_percent: float = 1.0 # <<< ADDED: Percentage of head dim to apply RoPE
    # Dropout probabilities - align with mcore args if possible
    ffn_dropout_p: float = 0.0
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.0
    # Add other relevant params as needed

class KronosU_Mcore(nn.Module):
    """ Native PyTorch model mirroring the mcore LLaMA-like structure """
    def __init__(self, config: KronosUConfig):
        super().__init__()
        self.config = config

        # Word Embedding
        # Note: MCore GPTModel uses LanguageModelEmbedding which includes VocabParallelEmbedding
        # and optionally PositionEmbedding. This native model uses a simple nn.Embedding.
        # The conversion script will need to handle potential TP sharding of this weight.
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer Blocks
        self.transformer = nn.ModuleList(
            [TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                num_query_groups=config.num_query_groups,
                ff_dim=config.ff_dim,
                # rotary_base=config.rotary_base, # Removed
                rotary_percent=config.rotary_percent,
                ffn_dropout_p=config.ffn_dropout_p,
                attn_dropout_p=config.attn_dropout_p,
                resid_dropout_p=config.resid_dropout_p,
                norm_eps=config.norm_eps
            ) for _ in range(config.n_layers)]
        )

        # Final RMSNorm (corresponds to final_layernorm in mcore GPTModel)
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # Output Head (corresponds to output_layer in mcore GPTModel)
        # Note: MCore uses ColumnParallelLinear. Conversion needs to handle potential TP.
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (optional, depends on training config)
        # self.embedding.weight = self.head.weight

        # Weight initialization (optional, could mimic mcore's init_method)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # Example init
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass similar to mcore GPTModel but with batch-first convention typical in HF/native PyTorch.

        Args:
            input_ids (torch.Tensor): Shape [batch_size, seq_len]
            position_ids (Optional[torch.Tensor]): Shape [batch_size, seq_len]. MCore uses this for learned pos embeddings.
                                                    Not directly used here as RoPE relies on sequence position, but kept for interface compatibility.
            attention_mask (Optional[torch.Tensor]): Mask for attention. Shape depends on implementation (e.g., [batch, seq_len] or [batch, 1, seq_len, seq_len]).
                                                   Needs processing to match F.scaled_dot_product_attention requirements.

        Returns:
            torch.Tensor: Logits of shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # 1. Embedding
        h = self.embedding(input_ids) # [batch, seq_len, d_model]

        # MCore handles position embeddings within its embedding layer or via RoPE args.
        # This native model relies on RoPE within the attention block.

        # Prepare attention mask if provided (needs specific format for F.scaled_dot_product_attention)
        # MCore mask format is [1, 1, seq_len, seq_len] typically. Can be additive or boolean.
        # F.scaled_dot_product_attention needs broadcastable [batch, n_heads, q_len, k_len] boolean (True=mask) or additive.
        # We'll handle basic causal mask within the attention module if attention_mask is None.
        # If a mask is provided, it needs to be adapted here or inside the attention module.
        processed_attn_mask = attention_mask # Pass it along for now, attn module will try to adapt

        # 2. Transformer Blocks
        for layer in self.transformer:
            h = layer(h, attention_mask=processed_attn_mask)

        # 3. Final Norm
        h = self.norm(h)

        # 4. Output Head
        logits = self.head(h) # [batch, seq_len, vocab_size]

        return logits
