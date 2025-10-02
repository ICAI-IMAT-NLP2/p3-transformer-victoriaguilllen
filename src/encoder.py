import torch
import torch.nn as nn

try:
    from utils import MultiHeadAttention, FeedForward, Embeddings
except ModuleNotFoundError:
    from src.utils import MultiHeadAttention, FeedForward, Embeddings

class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer.

    This class implements a single layer of the Transformer encoder, consisting
    of a multi-head attention mechanism followed by a feed-forward neural network.
    Both sub-layers are surrounded by residual connections and layer normalization.

    Args:
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads in the multi-head attention mechanism.
        intermediate_size (int): The dimension of the feed-forward network's intermediate layer.

    Attributes:
        layer_norm_1 (nn.LayerNorm): Layer normalization applied before the multi-head attention.
        layer_norm_2 (nn.LayerNorm): Layer normalization applied before the feed-forward network.
        attention (MultiHeadAttention): Multi-head attention mechanism.
        feed_forward (FeedForward): Feed-forward neural network.
    """

    def __init__(self, d_model: int, num_attention_heads: int, intermediate_size: int):
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm_1 = torch.nn.LayerNorm(d_model, eps=1e-5, elementwise_affine=True)
        self.layer_norm_2 = torch.nn.LayerNorm(d_model, eps=1e-5, elementwise_affine=True)
        self.attention = MultiHeadAttention(d_model=d_model, num_attention_heads=num_attention_heads)
        self.feed_forward = FeedForward(d_model=d_model, intermediate_size=intermediate_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Mask tensor for encoder tensor due to padding (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Apply layer normalization and then apply multi-head attention
        B, S, _ = x.shape

        keep = None
        if mask is not None:
            if mask.dtype != torch.bool:
                pad = mask != 0
            else:
                pad = mask

            if pad.dim() == 2:
                # (B, S) -> (B, S_q, S_k) = (B, S, S)
                # True donde NO hay padding
                q_ok = ~pad  # (B, S)
                keep = q_ok.unsqueeze(1).expand(-1, S, -1)  # (B, S, S)
            elif pad.dim() == 3:
                # Si ya viene como (B, S, S) de padding, negamos para obtener keep
                keep = ~pad


        # Pre-LN + MHA (self-attention) 
        h = self.layer_norm_1(x)
        attn_out = self.attention(h, h, h, mask=keep)  # (B, S, d_model)
        y = x + attn_out

        # Pre-LN + FFN 
        h2 = self.layer_norm_2(y)
        ff_out = self.feed_forward(h2)
        x = y + ff_out
        return x
    
class TransformerEncoder(nn.Module):
    """Transformer Encoder.

    This class implements the encoder part of the Transformer model, consisting
    of an embeddings layer followed by a stack of Transformer encoder layers.

    Args:
        vocab_size (int): The size of the vocabulary.
        max_position_embeddings (int): The maximum number of positions for positional embeddings.
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads in the multi-head attention mechanism.
        intermediate_size (int): The dimension of the feed-forward network's intermediate layer.
        num_hidden_layers (int): The number of Transformer encoder layers to stack.

    Attributes:
        embeddings (Embeddings): Embeddings layer combining token and positional embeddings.
        layers (nn.ModuleList): List of Transformer encoder layers.
    """

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int,
                num_attention_heads: int, intermediate_size: int, num_hidden_layers: int
                ):
        super(TransformerEncoder, self).__init__()
        self.embeddings = Embeddings(vocab_size, max_position_embeddings, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_hidden_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            mask (torch.Tensor): Mask tensor for encoder tensor due to padding (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Apply embeddings layer
        x = self.embeddings(x)

        for layer in self.layers:
            x = layer(x, mask)  

        return x
    
