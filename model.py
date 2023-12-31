#############################################
# Diego Calanzone
# Research Interest Demonstration
# University of Trento, Italy
#############################################

import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class EnglishDistinguisher(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()

        self.d_model = d_model
        self.linear = nn.Linear(d_hid, 1)

        # Embeddings
        self.embedding = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.seg_encoder = SegmentEmbedding(d_model)
        
        # Attn
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, seg_mask:Tensor, attn_mask: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        
        src = self.embedding(src) + self.seg_encoder(seg_mask)
        src = torch.transpose(src, 0, 1)    # S, N, E
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src, src_key_padding_mask=attn_mask)
        output = torch.transpose(output, 0, 1)

        return self.linear(output[:, 0]).squeeze(-1) # CLS
    

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html   
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(2, embed_size, padding_idx=0) # 0 is the CLS token