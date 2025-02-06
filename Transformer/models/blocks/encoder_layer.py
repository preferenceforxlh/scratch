import torch
import torch.nn as nn
from models.layers.layer_norm import LayerNorm
from models.layers.position_wise_feed_forward import PositionWiseFeedForward
from models.layers.multi_head_attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_heads,drop_prob):
        super(EncoderLayer,self).__init__()
        self.attention = MultiHeadAttention(d_model,n_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.ffn = PositionWiseFeedForward(d_model,ffn_hidden)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)
    
    def forward(self,x,s_mask):
        #1. compute attention
        _x = x
        x = self.attention(q=x,k=x,v=x,mask=s_mask)
        # 2.add residul and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        # 3. feed forward
        _x = x
        x = self.ffn(x)
        # 4. add residual and norm
        x = self.dropout2(x)
        self.norm2(x + _x)
        return x
