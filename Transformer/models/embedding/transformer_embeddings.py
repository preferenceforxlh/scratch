import torch
from torch import nn
from models.embedding.position_embeddings import PositionEmbedding
from models.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    def __init__(self,vocab_size,d_model,max_len,drop_prob,device):
        super(TransformerEmbedding,self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size,d_model)
        self.pos_emb = PositionEmbedding(d_model,max_len)
        self.drop_out = nn.Dropout(drop_prob)
    
    def forward(self,x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb+pos_emb)