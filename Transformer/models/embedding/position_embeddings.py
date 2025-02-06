from torch import nn
import torch

class PositionEmbedding(nn.Module):

    def __init__(self,d_model,max_len,device):
        super(PositionEmbedding,self).__init__()
        # 创建embedding
        self.encoding = torch.zeros(max_len,d_model,device=device)
        self.encoding.requires_grad = True
        # 创建pos位置
        self.pos = torch.arange(0,max_len,device=device)
        self.pos = self.pos.float().unsqueeze(1)
        # 生成正弦和余弦函数
        _2i = torch.arange(0,d_model,step=2,device=device).float()
        self.encoding[:,0::2] = torch.sin(self.pos / (10000 ** (_2i / d_model)))
        self.encoding[:,1::2] = torch.cos(self.pos / (10000 ** (_2i / d_model)))
    
    def forward(self,x):
        bs,seq_len = x.shape
        return self.encoding[:seq_len,:]