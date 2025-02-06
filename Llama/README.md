### prepare train data
1. python3 prepare_shakespeare.py 

### pretrain

2. python3 train_shakespeare.py

### inference

3. python3 generate.py --checkpoint_path=./out/shakespeare/iter-000100-ckpt.pth --tokenizer_path=./data/shakespeare/tokenizer.model --prompt="life is" --top_k=5

### prepare alpaca data
1. python3 prepare_alpaca.py --tokenizer_path=./data/shakespeare/tokenizer.model

### full sft
2. python3 lora_sft.py