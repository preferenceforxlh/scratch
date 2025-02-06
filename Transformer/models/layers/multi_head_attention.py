import torch
from torch import nn
from models.layers.scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_head):
        super(MultiHeadAttention,self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_concat = nn.Linear(d_model,d_model)
    
    def forward(self,q,k,v,mask=None):
        # 1. Do linear projection
        q,k,v = self.w_q(q),self.w_k(k),self.w_v(v)
        # 2. Split the input into multiple heads
        q,k,v = self.split(q),self.split(k),self.split(v)
        # 3. Do attention
        out,attention = self.attention(q,k,v,mask)
        # 4. Concatenate heads
        out = self.concat(out)
        # 5. Linear projection
        out = self.w_concat(out)
        return out
    
    def split(self,tensor):
        batch_size,length,d_model = tensor.shape
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size,length,self.n_head,d_tensor).transpose(1,2)
        return tensor
    
    def concat(self,tensor):
        batch_size,n_head,length,d_tensor = tensor.shape
        tensor = tensor.transpose(1,2).contiguous().view(batch_size,length,n_head*d_tensor)
        return tensor
