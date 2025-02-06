import torch
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class DataLoader:
    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')

    def yield_tokens(self, data_iter, tokenizer):
        for _, text in data_iter:
            yield tokenizer(text)

    def build_vocab(self, train_iter, tokenizer, min_freq):
        vocab = build_vocab_from_iterator(self.yield_tokens(train_iter, tokenizer), min_freq=min_freq, specials=[self.init_token, self.eos_token])
        vocab.set_default_index(vocab["<unk>"])
        return vocab

    def data_process(self, raw_text_iter, src_vocab, tgt_vocab):
        data = []
        for src_raw, tgt_raw in raw_text_iter:
            src_tensor = torch.tensor([src_vocab[token] for token in self.tokenize_de(src_raw)], dtype=torch.long)
            tgt_tensor = torch.tensor([tgt_vocab[token] for token in self.tokenize_en(tgt_raw)], dtype=torch.long)
            data.append((src_tensor, tgt_tensor))
        return data

    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(torch.cat([torch.tensor([self.source_vocab[self.init_token]]), src_sample, torch.tensor([self.source_vocab[self.eos_token]])]))
            tgt_batch.append(torch.cat([torch.tensor([self.target_vocab[self.init_token]]), tgt_sample, torch.tensor([self.target_vocab[self.eos_token]])]))
        src_batch = pad_sequence(src_batch, padding_value=self.source_vocab["<pad>"])
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.target_vocab["<pad>"])
        return src_batch, tgt_batch

    def make_dataset(self):
        train_iter, valid_iter, test_iter = Multi30k(split=('train', 'valid', 'test'), language_pair=self.ext)
        self.source_vocab = self.build_vocab(train_iter, self.tokenize_de, min_freq=2)
        self.target_vocab = self.build_vocab(train_iter, self.tokenize_en, min_freq=2)
        train_data = self.data_process(train_iter, self.source_vocab, self.target_vocab)
        valid_data = self.data_process(valid_iter, self.source_vocab, self.target_vocab)
        test_data = self.data_process(test_iter, self.source_vocab, self.target_vocab)
        return train_data, valid_data, test_data

    def make_iter(self, train, validate, test, batch_size, device):
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        valid_loader = DataLoader(validate, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        print('dataset initializing done')
        return train_loader, valid_loader, test_loader
    
    def get_vocab_size(self):
        source_vocab_size = len(self.source_vocab)
        target_vocab_size = len(self.target_vocab)
        return source_vocab_size, target_vocab_size
    
    def get_special_tokens_idx(self):
        src_pad_idx = self.source_vocab['<pad>']
        trg_pad_idx = self.target_vocab['<pad>']
        trg_sos_idx = self.target_vocab[self.init_token]
        return src_pad_idx, trg_pad_idx, trg_sos_idx

