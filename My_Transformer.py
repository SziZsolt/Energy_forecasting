import math
import numpy as np
import torch
import torch.nn as nn

def positional_encoding(seq_length: int, d_model: int) -> torch.Tensor:
    """
    Compute sinusoidal positional encodings for each position in the sequence.

    Args:
        seq_length (int): sequence length
        d_model (int): embedding dimension

    Returns:
        pe (torch.Tensor): shape (seq_length, d_model)
    """
    pe = torch.zeros(seq_length, d_model)
    position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (seq_length, d_model)

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention layer with residual (skip) connection and LayerNorm.

    Args:
        d_model (int): input dimension
        out_features (int): output dimension
        seq_length (int): sequence length
        num_heads (int): number of attention heads
        dropout (float): dropout rate
    """

    def __init__(self, d_model: int, out_features: int, seq_length: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_model
        self.num_heads = num_heads
        self.seq_length = seq_length

        self.Q = nn.Linear(d_model, self.d_k * num_heads)
        self.K = nn.Linear(d_model, self.d_k * num_heads)
        self.V = nn.Linear(d_model, self.d_k * num_heads)
        self.linear = nn.Linear(self.d_k, out_features)

        self.skip_proj = nn.Linear(d_model, out_features)
        self.layernorm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, masked: bool = False) -> torch.Tensor:
        """
        Forward pass of multi-head attention.

        Args:
            x (torch.Tensor): (batch_size, seq_length, d_model)
            masked (bool): whether to apply causal masking (for decoder)

        Returns:
            out (torch.Tensor): (batch_size, seq_length, out_features)
        """
        skip_x = x
        query = self.Q(x)  # (batch_size, seq_length, d_k * num_heads)
        key = self.K(x)
        value = self.V(x)

        # reshape for multi-heads
        query = query.view(-1, self.num_heads, self.seq_length, self.d_k)
        key = key.view(-1, self.num_heads, self.seq_length, self.d_k)
        value = value.view(-1, self.num_heads, self.seq_length, self.d_k)

        # scaled dot-product attention
        attention = query @ key.permute(0, 1, -1, 2)  # (batch_size, num_heads, seq_length, seq_len)
        attention = attention / np.sqrt(self.d_k)

        if masked:
            mask = torch.tril(torch.ones(self.seq_length, self.seq_length))
            mask = mask.unsqueeze(0).expand(x.shape[0], -1, -1).unsqueeze(1)  # (batch_size, 1, seq_length, seq_length)
            attention = attention.masked_fill(mask == 0, -np.inf)

        attention = torch.softmax(attention, dim=-1)
        attention = attention @ value  # (batch_size, num_heads, seq_length, d_k)
        attention = torch.sum(attention, dim=1)  # (batch_size, seq_length, d_k)
        out = self.linear(attention)  # (batch_size, seq_length, out_features)
        out = self.dropout(out)

        out = self.layernorm(out + self.skip_proj(skip_x))
        return out  # (batch_size, seq_length, out_features)



class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention layer with residual (skip) connection and LayerNorm.

    Args:
        d_model (int): decoder input dimension
        enc_out_dim (int): encoder output dimension
        out_features (int): output dimension
        seq_length (int): sequence length
        num_heads (int): number of attention heads
        dropout (float): dropout rate
    """

    def __init__(self, d_model: int, enc_out_dim: int, out_features: int, seq_length: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_model
        self.num_heads = num_heads
        self.seq_length = seq_length

        self.Q = nn.Linear(enc_out_dim, self.d_k * num_heads)
        self.K = nn.Linear(enc_out_dim, self.d_k * num_heads)
        self.V = nn.Linear(d_model, self.d_k * num_heads)
        self.linear = nn.Linear(self.d_k, out_features)

        self.skip_proj = nn.Linear(d_model, out_features)
        self.layernorm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of cross-attention.

        Args:
            x (torch.Tensor): encoder output (batch_size, seq_length, enc_out_dim)
            y (torch.Tensor): decoder input (batch_size, seq_length, d_model)

        Returns:
            out (torch.Tensor): (batch_size, seq_length, out_features)
        """
        skip_y = y
        query = self.Q(x)
        key = self.K(x)
        value = self.V(y)

        query = query.view(-1, self.num_heads, self.seq_length, self.d_k) # (batch_size, num_heads, seq_length, d_k)
        key = key.view(-1, self.num_heads, self.seq_length, self.d_k) # (batch_size, num_heads, seq_length, d_k)
        value = value.view(-1, self.num_heads, self.seq_length, self.d_k) # (batch_size, num_heads, seq_length, d_k)

        attention = query @ key.permute(0, 1, 3, 2) # (batch_size, num_heads, seq_length, seq_length)
        attention = attention / np.sqrt(self.d_k)
        attention = torch.softmax(attention, dim=-1)
        attention = attention @ value # (batch_size, num_heads, seq_length, d_k)
        attention = torch.sum(attention, dim=1) # (batch_size, seq_length, d_k)
        out = self.linear(attention) # (batch_size, seq_length, out_features)
        out = self.dropout(out)

        out = self.layernorm(out + self.skip_proj(skip_y))
        return out  # (batch_size, seq_length, out_features)


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network with residual (skip) connection:
    Linear -> ReLU -> Dropout -> Linear -> Add & LayerNorm

    Args:
        d_model (int): input/output dimension
        d_ff (int): hidden layer dimension
        dropout (float): dropout rate
        use_layernorm (bool): if True, applies LayerNorm after residual add
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, use_layernorm: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection.

        Args:
            x (torch.Tensor): (batch_size, seq_length, d_model)

        Returns:
            out (torch.Tensor): (batch_size, seq_length, d_model)
        """
        skip_x = x 
        x = self.linear1(x) # (batch_size, seq_length, d_ff)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x) # (batch_size, seq_length, d_model)
        x = x + skip_x
        if self.use_layernorm:
            x = self.layernorm(x)
        return x # (batch_size, seq_length, d_model)

class AttentionMetaData:
    """Container for attention parameters (out_dim, num_heads)."""
    def __init__(self, out_dim: int, num_heads: int):
        self.out_dim = out_dim
        self.num_heads = num_heads


class EncoderMetaData:
    """Container for encoder configuration."""
    def __init__(self, in_dim: int, embedding_dim: int, seq_length: int, attention_metadata: AttentionMetaData, out_dim: int):
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.out_dim = out_dim
        self.attention_metadata = attention_metadata


class DecoderMetaData:
    """Container for decoder configuration."""
    def __init__(self, in_dim: int, embedding_dim: int, seq_length: int, encoder_dim: int,
                 attention_1_metadata: AttentionMetaData, attention_2_metadata: AttentionMetaData, out_dim: int):
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.encoder_dim = encoder_dim
        self.out_dim = out_dim
        self.attention_1_metadata = attention_1_metadata
        self.attention_2_metadata = attention_2_metadata

class TransformerEncoder(nn.Module):
    """
    Encoder block:
    Embedding + PosEnc + Self-Attn + FFN + residual (skip) connections + LayerNorm
    """

    def __init__(self, metadata: EncoderMetaData, pos_encoding: bool = False):
        super().__init__()
        self.embedding = nn.Linear(metadata.in_dim, metadata.embedding_dim)
        self.pos_encoding = positional_encoding(metadata.seq_length, metadata.embedding_dim)
        self.multihead_attention = MultiHeadAttention(metadata.embedding_dim, metadata.attention_metadata.out_dim,
                                                      metadata.seq_length, metadata.attention_metadata.num_heads)
        self.feedforward = FeedForwardNetwork(metadata.attention_metadata.out_dim, metadata.out_dim)
        self.pos_encoding = pos_encoding

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (batch_size, seq_length, in_dim)
        Returns:
            out (torch.Tensor): (batch_size, seq_length, out_dim)
        """
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        if self.pos_encoding:
          x = x + self.pos_encoding 
        x = self.multihead_attention(x) # (batch_size, seq_length, attention_out_dim)
        x = torch.relu(x)
        x = self.feedforward(x)  # (batch_size, seq_length, out_dim)
        return x # (batch_size, seq_length, out_dim)


class TransformerDecoder(nn.Module):
    """
    Decoder block:
    Masked self-attn + Cross-attn + FFN + residual (skip) connections + LayerNorm
    """

    def __init__(self, metadata: DecoderMetaData, pos_encoding: bool = False):
        super().__init__()
        self.embedding = nn.Linear(metadata.in_dim, metadata.embedding_dim)
        self.pos_encoding = positional_encoding(metadata.seq_length, metadata.embedding_dim)
        self.multihead_attention = MultiHeadAttention(metadata.embedding_dim, metadata.attention_1_metadata.out_dim,
                                                      metadata.seq_length, metadata.attention_1_metadata.num_heads)
        self.multihead_crossattention = MultiHeadCrossAttention(metadata.attention_1_metadata.out_dim,
                                                                metadata.encoder_dim, metadata.attention_2_metadata.out_dim,
                                                                metadata.seq_length, metadata.attention_2_metadata.num_heads)
        self.feedforward = FeedForwardNetwork(metadata.attention_2_metadata.out_dim, metadata.out_dim)
        self.pos_encoding = pos_encoding

    def forward(self, x: torch.Tensor, y: torch.Tensor, masked: bool = False):
        """
        Args:
            x (torch.Tensor): encoder output (batch_size, seq_length, encoder_dim)
            y (torch.Tensor): decoder input (batch_size, seq_length, in_dim)
        Returns:
            out (torch.Tensor): (batch_size, seq_length, out_dim)
        """
        y = self.embedding(y)  # (batch_size, seq_length, embedding_dim)
        if self.pos_encoding:
          y = y + self.pos_encoding
        y = self.multihead_attention(y, masked) # (batch_size, seq_length, attention_1_out_dim)
        y = torch.relu(y)
        y = self.multihead_crossattention(x, y)  # (batch_size, seq_length, attention_2_out_dim)
        y = torch.relu(y)
        y = self.feedforward(y) # (batch_size, seq_length, out_dim)
        y = torch.relu(y)
        return y # (batch_size, seq_length, out_dim)


class TransformerModel(nn.Module):
    """
    Full Transformer architecture wrapping multiple Encoder-Decoder layers.

    Args:
        num_layers (int): number of encoder-decoder layers
        input_dim (int): input feature dimension
        embedding_dim (int): embedding dimension for the first layer
        seq_length (int): sequence length
        hidden_dim (int): hidden dimension for feed-forward layers
        num_heads (int): number of attention heads per layer
        out_dim (int): output dimension for the final layer
    """

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        embedding_dim: int,
        seq_length: int,
        hidden_dim: int,
        num_heads: int,
        out_dim: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.seq_length = seq_length

        layers = []
        in_dim = input_dim
        use_pos_encoding = True
        for i in range(num_layers):                
            # Attention metadata
            attn_meta_enc = AttentionMetaData(out_dim=hidden_dim, num_heads=num_heads)
            attn_meta_dec1 = AttentionMetaData(out_dim=hidden_dim, num_heads=num_heads)
            attn_meta_dec2 = AttentionMetaData(out_dim=hidden_dim, num_heads=num_heads)

            # Encoder metadata
            enc_meta = EncoderMetaData(
                in_dim=in_dim,
                embedding_dim=embedding_dim,
                seq_length=seq_length,
                attention_metadata=attn_meta_enc,
                out_dim=hidden_dim,
            )

            # Decoder metadata
            dec_meta = DecoderMetaData(
                in_dim=in_dim,
                embedding_dim=embedding_dim,
                seq_length=seq_length,
                encoder_dim=hidden_dim,
                attention_1_metadata=attn_meta_dec1,
                attention_2_metadata=attn_meta_dec2,
                out_dim=hidden_dim,
            )

            encoder = TransformerEncoder(enc_meta, pos_encoding=use_pos_encoding)
            decoder = TransformerDecoder(dec_meta, pos_encoding=use_pos_encoding)
            layers.append(nn.ModuleDict({
                "encoder": encoder,
                "decoder": decoder
            }))
            in_dim = hidden_dim
            use_pos_encoding = False

        self.layers = nn.ModuleList(layers)
        self.final_linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Run the input through all encoder-decoder layers.

        Args:
            x (torch.Tensor): encoder input (batch, seq_length, input_dim)
            y (torch.Tensor): decoder input (batch, seq_length, input_dim)
            masked (bool): apply causal mask to self-attention in decoder

        Returns:
            out (torch.Tensor): (batch, seq_length, out_dim)
        """
        masked = True
        for layer in self.layers:
            encoder = layer["encoder"]
            decoder = layer["decoder"]
            enc_out = encoder(x)
            y = decoder(enc_out, y, masked=masked)
            x = enc_out 
            masked = False
        out = self.final_linear(y)
        return out

