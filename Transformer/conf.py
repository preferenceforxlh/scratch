import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model parameters setting
batch_size = 128
max_len = 256
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1
# optimizer parameters setting
init_lr = 1e-5
factor = 0.9
adam_eps = 1e-9
patience = 10
warmup = 100
epoch=100
clip=1.0
weight_decay=5e-4
inf = float("-inf")