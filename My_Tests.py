import torch
import pytest
import math
from My_Transformer import (
    positional_encoding,
    MultiHeadAttention,
    MultiHeadCrossAttention,
    FeedForwardNetwork,
    TransformerEncoder,
    TransformerDecoder,
    TransformerModel,
    AttentionMetaData,
    EncoderMetaData,
    DecoderMetaData
)

def test_positional_encoding_shape():
    seq_len, d_model = 10, 16
    pe = positional_encoding(seq_len, d_model)
    assert pe.shape == (seq_len, d_model), "Positional encoding has incorrect shape"


def test_positional_encoding_variation():
    seq_len, d_model = 10, 16
    pe = positional_encoding(seq_len, d_model)
    # Positions should not all be identical
    assert not torch.allclose(pe[0], pe[1]), "Different positions should have different encodings"


def test_multihead_attention_output_shape():
    batch, seq_len, d_model = 2, 8, 32
    num_heads, out_dim = 4, 64
    mha = MultiHeadAttention(d_model, out_dim, seq_len, num_heads)
    x = torch.randn(batch, seq_len, d_model)
    out = mha(x)
    assert out.shape == (batch, seq_len, out_dim), "MHA output shape mismatch"


def test_multihead_attention_masked():
    batch, seq_len, d_model = 2, 6, 16
    mha = MultiHeadAttention(d_model, 32, seq_len, 2)
    x = torch.randn(batch, seq_len, d_model)
    out = mha(x, masked=True)
    assert not torch.isnan(out).any(), "Masked attention should not produce NaN values"


def test_cross_attention_output_shape():
    batch, seq_len, d_model, enc_dim = 2, 6, 16, 32
    cross_attn = MultiHeadCrossAttention(d_model, enc_dim, 64, seq_len, 4)
    x = torch.randn(batch, seq_len, enc_dim)
    y = torch.randn(batch, seq_len, d_model)
    out = cross_attn(x, y)
    assert out.shape == (batch, seq_len, 64), "Cross-attention output shape mismatch"


def test_feedforward_output_shape():
    batch, seq_len, d_model, d_ff = 4, 10, 32, 128
    ffn = FeedForwardNetwork(d_model, d_ff)
    x = torch.randn(batch, seq_len, d_model)
    out = ffn(x)
    assert out.shape == (batch, seq_len, d_model), "FeedForward output shape mismatch"


def test_feedforward_residual_connection():
    batch, seq_len, d_model, d_ff = 1, 5, 16, 64
    ffn = FeedForwardNetwork(d_model, d_ff)
    x = torch.randn(batch, seq_len, d_model)
    out = ffn(x)
    # Output should roughly preserve input info (due to residual)
    assert torch.isfinite(out).all(), "FFN output contains NaNs or infs"


def test_encoder_forward_pass():
    attn_meta = AttentionMetaData(out_dim=64, num_heads=4)
    enc_meta = EncoderMetaData(in_dim=16, embedding_dim=32, seq_length=10, attention_metadata=attn_meta, out_dim=64)
    encoder = TransformerEncoder(enc_meta, pos_encoding=True)
    x = torch.randn(2, 10, 16)
    out = encoder(x)
    assert out.shape == (2, 10, 64), "Encoder output shape mismatch"


def test_decoder_forward_pass():
    attn1 = AttentionMetaData(out_dim=128, num_heads=2)
    attn2 = AttentionMetaData(out_dim=128, num_heads=2)
    dec_meta = DecoderMetaData(
        in_dim=16,
        embedding_dim=32,
        seq_length=10,
        encoder_dim=64,
        attention_1_metadata=attn1,
        attention_2_metadata=attn2,
        out_dim=128,
    )
    decoder = TransformerDecoder(dec_meta, pos_encoding=True)
    x = torch.randn(2, 10, 64)
    y = torch.randn(2, 10, 16)
    out = decoder(x, y)
    assert out.shape == (2, 10, 128), "Decoder output shape mismatch"


def test_full_transformer_model_forward():
    model = TransformerModel(
        num_layers=2,
        input_dim=16,
        embedding_dim=32,
        seq_length=10,
        hidden_dim=64,
        num_heads=2,
        out_dim=8,
    )
    x = torch.randn(2, 10, 16)
    y = torch.randn(2, 10, 16)
    out = model(x, y)
    assert out.shape == (2, 10, 8), "Full transformer output shape mismatch"


def test_transformer_gradients():
    model = TransformerModel(
        num_layers=1,
        input_dim=8,
        embedding_dim=16,
        seq_length=6,
        hidden_dim=32,
        num_heads=2,
        out_dim=4,
    )
    x = torch.randn(3, 6, 8, requires_grad=True)
    y = torch.randn(3, 6, 8)
    out = model(x, y)
    loss = out.mean()
    loss.backward()
    assert x.grad is not None, "Gradients did not backpropagate through the Transformer"


def test_no_nan_outputs():
    model = TransformerModel(
        num_layers=1,
        input_dim=8,
        embedding_dim=16,
        seq_length=6,
        hidden_dim=32,
        num_heads=2,
        out_dim=4,
    )
    x = torch.randn(2, 6, 8)
    y = torch.randn(2, 6, 8)
    out = model(x, y)
    assert torch.isfinite(out).all(), "Model output contains NaN or Inf values"
