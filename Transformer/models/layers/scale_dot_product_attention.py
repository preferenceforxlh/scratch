import math
from torch import nn
import torch

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention,self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self,q,k,v,mask=None,e=1e-12):
        # input is 4 dimension tensor
        # q: (batch_size,head_num,seq_len_q,embedding_dim)
        batch_size,head_num,seq_len,d_tensor = q.shape
        # 1.dot product of q and k
        kT = k.transpose(2,3)
        score = (q @ kT) / math.sqrt(d_tensor)

        # 2. apply mash opt
        if mask is not None:
            score = score.masked_fill(mask == 0,float("-inf"))
        
        # 3. apply softmax
        score = self.softmax(score)

        # 4.multiply with v
        v = score @ v

        # 5. return v and attn_score
        return v,score
