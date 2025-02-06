import torch
from torch import nn
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.layer_norm import LayerNorm
from models.layers.position_wise_feed_forward import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,drop_prob):
        super(DecoderLayer,self).__init__()
        self.self_attention = MultiHeadAttention(d_model,n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.cross_attention = MultiHeadAttention(d_model,n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionWiseFeedForward(d_model,ffn_hidden)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self,dec,enc,t_mask,s_mask):
        # 1. self-attention
        _x = dec
        x = self.self_attention(q = dec,k=dec,v=dec,mask= t_mask)

        # 2. add & norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. cross-attention
        if enc is not None:
            _x = x
            x = self.cross_attention(q = x,k = enc,v = enc,mask = s_mask)
            x = self.dropout2(x)
            x = self.norm2(x + _x)
        
        # 4. feed forward
        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x