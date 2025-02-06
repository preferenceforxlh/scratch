import torch
from torch import nn
from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embeddings import TransformerEmbedding

class Encoder(nn.Module):
    def __init__(self,enc_voc_sise,max_len,d_model,ffn_hidden,n_head,n_layers,drop_prob,device):
        super(Encoder,self).__init__()
        self.emb = TransformerEmbedding(enc_voc_sise,d_model,max_len,drop_prob,device)
        self.layers = nn.ModuleList([EncoderLayer(d_model,ffn_hidden,n_head,drop_prob) for _ in range(n_layers)])

    def forward(self,x,s_mask):
        # 1. Embedding
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x,s_mask)
        return x